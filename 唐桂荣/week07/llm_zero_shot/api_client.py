"""DeepSeek 官方 API：JSON 抽取、有限重试、用量和费用记录。"""

import os
import time

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"
MAX_TOKENS_DEFAULT = 1024
PRICE_SOURCE = "https://api-docs.deepseek.com/quick_start/pricing/"
PRICE_DATE = "2026-11-26"
# 官方美元峰时单价 / 百万 tokens；用于保守估算，实际扣费以平台账单为准。
PEAK_PRICES = {"cache_hit": 0.006, "cache_miss": 0.30, "output": 1.20}
USAGE_KEYS = ("prompt_tokens", "completion_tokens", "cache_hit_tokens", "cache_miss_tokens",
              "unknown_cache_tokens", "reasoning_tokens")


def build_api_client():
    """密钥只从环境读取；关闭 SDK 内置重试，由 call_api 统一记录重试次数。"""
    key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("LLM_API_KEY")
    if not key:
        raise EnvironmentError("请在运行脚本的终端先设置 DEEPSEEK_API_KEY 环境变量。")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError('当前 Python 环境缺少 openai SDK，请执行：python -m pip install "openai>=1,<3"') from exc
    return OpenAI(api_key=key, base_url=os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL),
                  timeout=60.0, max_retries=0)


def request_settings(model, max_tokens):
    """用于实际请求和断点续跑配置校验，保证两处参数一致。"""
    disabled = os.getenv("LLM_DISABLE_THINKING", "1") != "0"
    options = {"model": model, "max_tokens": max_tokens,
               "response_format": {"type": "json_object"},
               "extra_body": {"thinking": {"type": "disabled" if disabled else "enabled"}}}
    if disabled:
        options["temperature"] = 0.0
    return options


def empty_usage():
    return {key: 0 for key in USAGE_KEYS}


def add_usage(left, right):
    return {key: int(left.get(key, 0)) + int(right.get(key, 0)) for key in USAGE_KEYS}


def usage_of(usage):
    if usage is None:
        return empty_usage()
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    hit = getattr(usage, "prompt_cache_hit_tokens", None)
    miss = getattr(usage, "prompt_cache_miss_tokens", None)
    if hit is None:
        hit = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", None)
    if hit is None and miss is None:
        hit, miss, unknown = 0, 0, prompt
    else:
        hit = int(hit if hit is not None else max(0, prompt - int(miss)))
        miss = int(miss if miss is not None else max(0, prompt - hit))
        unknown = max(0, prompt - hit - miss)
    return {"prompt_tokens": prompt, "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "cache_hit_tokens": hit, "cache_miss_tokens": miss, "unknown_cache_tokens": unknown,
            "reasoning_tokens": int(getattr(getattr(usage, "completion_tokens_details", None),
                                             "reasoning_tokens", 0) or 0)}


def cost_estimate(usage):
    """含已返回用量的重试；未知缓存按未命中计，峰时价格估算而非实际账单。"""
    amount = (usage.get("cache_hit_tokens", 0) * PEAK_PRICES["cache_hit"]
              + (usage.get("cache_miss_tokens", 0) + usage.get("unknown_cache_tokens", 0)) * PEAK_PRICES["cache_miss"]
              + usage.get("completion_tokens", 0) * PEAK_PRICES["output"]) / 1e6
    return {"usd_peak_estimate": round(amount, 8), "prices_usd_per_million": PEAK_PRICES,
            "price_date": PRICE_DATE, "price_source": PRICE_SOURCE,
            "note": "按官方峰时价格估算；实际以账单为准。超时等未返回 usage 的调用无法精确计费。"}


def call_api(client, messages, model, max_tokens, retries=3):
    """只重试空正文、限流和暂时错误；鉴权/参数错误标为 fatal，由主流程立即停止。"""
    start = time.perf_counter()
    usage = empty_usage()
    history = []
    final = {"text": "", "error": "", "fatal": False, "response_model": model,
             "response_id": "", "finish_reason": None, "hit_max_new_tokens": False}
    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(messages=messages, **request_settings(model, max_tokens))
            usage = add_usage(usage, usage_of(getattr(response, "usage", None)))
            choice = response.choices[0]
            content = (choice.message.content or "").strip()
            final.update(text=content, response_model=response.model, response_id=response.id,
                         finish_reason=choice.finish_reason, hit_max_new_tokens=choice.finish_reason == "length")
            if content:
                final["error"] = ""
                history.append({"attempt": attempt, "status": "ok", "finish_reason": choice.finish_reason})
                break
            final["error"] = "API 返回了空正文。"
            history.append({"attempt": attempt, "status": "empty_reply", "finish_reason": choice.finish_reason})
        except Exception as error:
            status = getattr(error, "status_code", None)
            # 不把 SDK 的完整请求/响应对象写入日志，避免带入凭据。
            final["error"] = f"{type(error).__name__}" + (f" (HTTP {status})" if status else "")
            final["fatal"] = status in (400, 401, 402, 403, 404, 422)
            history.append({"attempt": attempt, "status": "error", "http_status": status,
                            "error_type": type(error).__name__})
            if final["fatal"]:
                break
        if attempt < retries:
            time.sleep(min(2 ** (attempt - 1), 4))
    return {**final, "api_usage": usage, "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"], "api_attempts": len(history),
            "attempt_history": history, "latency_s": time.perf_counter() - start}
