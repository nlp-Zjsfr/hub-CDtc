# -*- coding: utf-8 -*-
"""Qwen3-0.6B-Base zero/few-shot 事件抽取作业。

在 week7作业 目录运行：
  python llm_zero_shot/llm_extract.py --mode zero --n 60 --device mps
  python llm_zero_shot/llm_extract.py --mode few --n 60 --device mps
  python llm_zero_shot/llm_extract.py --backend api --n 5 --resume  # 官方 DeepSeek 试跑
  python llm_zero_shot/llm_extract.py --backend api --mode few --n 60 --resume
  python llm_zero_shot/llm_extract.py --dry_run --n 20
  python llm_zero_shot/llm_extract.py --mode few --print_prompt
  python llm_zero_shot/llm_extract.py --from_raw outputs/llm/raw_qwen_base_zero_n60.jsonl --n 60

复用原课程 processed 数据；本地加载权重；输出独立写入 week7作业/outputs。
API 后端需显式 --backend api；默认不调用 API。模拟模式不能作为实验结论。
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # Windows 上 torch 与 numpy 的 OpenMP 冲突

import argparse
import hashlib
import tempfile
from contextlib import nullcontext
import json
import random
import subprocess
import sys
import time
from pathlib import Path

# 让 `python llm_zero_shot/llm_extract.py` 直接跑也能 import 同目录模块

ROOT = Path(__file__).resolve().parents[1]   # 项目根目录
sys.path.insert(0, str(ROOT / "common"))        # 公共模块
sys.path.insert(0, str(ROOT / "llm_zero_shot"))  # 生成式链路共用：event_data / event_prompt / local_generate
sys.path.insert(0, str(Path(__file__).resolve().parent))  # 本目录的 Qwen model_paths.py


from log_utils import EvalLogger  # noqa: E402
from event_data import (  # noqa: E402
    OUTPUT_LLM_DIR, ROOT, SEED, DATA_DIR, canonical_order, load_schema, load_split,
    make_pred_row, sample_rows, uid, write_eval_ids, write_gold_subset,
)
from event_prompt import (  # noqa: E402
    FEW_SHOT_EXAMPLES, build_messages, build_system_prompt, parse_model_output,
)
from local_generate import generate_batch, load_causal_lm, resolve_torch_dtype, select_device  # noqa: E402

from api_client import (DEFAULT_BASE_URL, DEFAULT_MODEL, MAX_TOKENS_DEFAULT, build_api_client,
                        call_api, request_settings, empty_usage, add_usage, cost_estimate)
from model_paths import resolve_model_path

MODEL_PATH_DEFAULT = None


def write_raw_checkpoint(path, records, rows):
    """原子替换最新回复表，保留尚未遍历的已完成样本，避免续跑中断丢失回复。"""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            for row in rows:
                record = records.get(uid(row))
                if record is not None:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 后端 2：本地 Qwen3-0.6B（实现在 local_generate.py：批量解码 + 左 padding）
# ══════════════════════════════════════════════════════════════════════════════
# 为什么单独抽一个模块：llm_extract.py（zero-shot）和 evaluate_sft.py（SFT 后评测）
# 都要用同一套生成逻辑，抽出来才能保证"生成参数、thinking 开关、批量策略"三处一致。

# ══════════════════════════════════════════════════════════════════════════════
# 后端 3：mock（--dry-run，离线演示）
# ══════════════════════════════════════════════════════════════════════════════

def mock_generate(row: dict, idx: int, seed: int) -> str:
    """
    假模型：不联网不占 GPU，按固定概率"模拟"真实模型的各种毛病。

    为什么 mock 要故意制造解析失败：
      只有这样才能验证「解析修复 + 统计」这条工程链路真的在工作
      （剥围栏、括号配平、截断补全、空回复计数）。
      注意：**mock 跑出来的 F1 没有任何意义**，只用于打通流程 / 课堂演示，
      绝不能写进 PPT 的对比表。
    """
    rng = random.Random(seed * 100000 + idx)
    event = row.get("event") or (row.get("events") or [None])[0]
    if not event:
        return json.dumps({"events": []})
    n_args = len(event["args"])
    keep = max(0, n_args // 2)              # 故意只抽一半论元 → F1 适中
    args = [{"role": a["role"], "argument": a["text"]} for a in event["args"][:keep]]
    obj = {"events": [{"type": event["type"],
                       "trigger": {"text": event["trigger"]["text"]},
                       "arguments": args}]}
    body = json.dumps(obj, ensure_ascii=False)
    roll = rng.random()
    if roll < 0.05:                          # 5% 空回复（模拟 max_tokens 被思考吃光）
        return ""
    if roll < 0.10:                          # 5% 输出被截断（模拟 max_tokens 不够）
        return body[: int(len(body) * 0.8)]
    if roll < 0.20:                          # 10% 包了 markdown 围栏（最常见）
        return f"```json\n{body}\n```"
    if roll < 0.25:                          # 5% 前后夹带解释文字
        return f"好的，抽取结果如下：\n{body}\n（如需调整请告诉我）"
    if roll < 0.30:                          # 5% 类型抽错
        types = [t for t in SCHEMA["event_types"] if t != event["type"]]
        obj["events"][0]["type"] = rng.choice(types) if types else event["type"]
        return json.dumps(obj, ensure_ascii=False)
    return body

SCHEMA = {}  # 全局 schema（main 里加载；mock_generate 会用到）

# ══════════════════════════════════════════════════════════════════════════════
# 评测（统一口径：调用 common/evaluate_events.py，绝不自己算一套）
# ══════════════════════════════════════════════════════════════════════════════

def run_shared_evaluation(gold_path: Path, pred_path: Path, metrics_path: Path) -> dict:
    """
    跑项目的统一评测脚本，返回其 JSON 结果。

    为什么要 subprocess 调脚本而不是 import 函数：
      保证"报告里的数字 = 命令行 `python common/evaluate_events.py ...` 跑出来的数字"，
      学生复现时不会因为调用方式不同得到略有差异的指标。多花一次进程启动很值。
    """
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-B", str(ROOT / "common" / "evaluate_events.py"),
           "--gold", str(gold_path), "--pred", str(pred_path),
           "--json", str(metrics_path), "--quiet"]
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)

def _pct(a: int, b: int) -> float:
    return round(100.0 * a / b, 2) if b else 0.0

def _print_metrics(res: dict, report: dict) -> None:
    """把评测结果 + 工程指标打印成一张人看得懂的表（PPT 直接抄）。"""
    print("\n" + "=" * 78)
    print(f"结果：{report['tag']}（{report['backend']}/{report['mode']}，"
          f"{report['n']} 条）")
    print("=" * 78)
    print(f"  trigger F1   : {res['trigger']['f1']:.4f}"
          f"   (P={res['trigger']['precision']:.4f} R={res['trigger']['recall']:.4f})")
    print(f"  argument F1  : {res['argument']['f1']:.4f}"
          f"   (P={res['argument']['precision']:.4f} R={res['argument']['recall']:.4f})")
    print(f"  event  F1    : {res['event_exact']['f1']:.4f}")
    print(f"  type_macro_f1: {res['type_macro_f1']:.4f}")
    print(f"  JSON 合法率  : {report['json_valid_rate']}%（结构性修复 {report['repaired']} 条，"
          f"围栏/废话剥离 {report['noise_stripped']} 条）")
    print(f"  解析失败     : {report['parse_fail']} 条 | 空回复 {report['empty_reply']} 条")
    print(f"  幻觉论元     : {report['arg_not_in_source']} 次 | "
          f"触发词找不到 {report['trigger_not_in_source']} 次")
    print(f"  平均延迟     : {report['latency_mean_s']}s/条（p50={report['latency_p50_s']}s）")
    print(f"  token        : prompt {report['prompt_tokens']} + completion "
          f"{report['completion_tokens']} = {report['total_tokens']}")

# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════

def check_gpu_idle(threshold_mib: int = 1500) -> None:
    """
    本地推理前先看一眼显存占用（8GB 卡要和同学/其它脚本共用，必须排队）。

    只做"提示"不做"拦截"：显存被别人占着时程序仍能启动（可能 OOM），
    但至少能让人立刻意识到"不是模型的问题，是卡被占了"。
    """
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        used = int(out.splitlines()[0].split(",")[0])
        print(f"[gpu] 当前显存占用 {used} MiB（阈值 {threshold_mib} MiB）")
        if used > threshold_mib:
            print("[gpu][警告] 显存占用较高，可能有其它 GPU 任务在跑；"
                  "8GB 卡同时跑两个任务很容易 OOM。")
    except Exception:  # noqa: BLE001 —— 没装 nvidia-smi 就跳过，不阻塞主流程
        print("[gpu] 未获取到显存信息（nvidia-smi 不可用），跳过检查")

def parse_args():
    p = argparse.ArgumentParser(description="LLM zero-shot / few-shot 事件抽取")
    p.add_argument("--backend", default="local", choices=["api", "local", "mock"],
                   help="api=OpenAI 兼容 API；local=本地 Qwen3-0.6B；mock=离线假模型")
    p.add_argument("--mode", default="zero", choices=["zero", "few"],
                   help="zero=不给示例；few=给 6 个难类型示例")
    p.add_argument("--n", default=200, type=int, help="评测样本数（默认 200），0 使用全量")
    p.add_argument("--seed", default=SEED, type=int)
    p.add_argument("--data", default=None, help="金标 JSONL（默认 data/processed/dev_extract.jsonl）")
    p.add_argument("--tag", default="", help="输出文件后缀，默认 {backend}_{mode}_n{n}")
    p.add_argument("--out", default="", help="预测输出路径，默认 outputs/llm/pred_{tag}.jsonl")
    p.add_argument("--model", default=os.getenv("LLM_MODEL", DEFAULT_MODEL))
    p.add_argument("--model_path", default=MODEL_PATH_DEFAULT,
                   help="本地模型目录（默认 week7作业/qwen3-0.6b-base）")
    p.add_argument("--max_tokens", default=MAX_TOKENS_DEFAULT, type=int,
                   help="API 正文输出 token 上限，默认 1024")
    p.add_argument("--max_new_tokens", default=384, type=int, help="本地模型最大新 token 数")
    p.add_argument("--gen_batch", default=None, type=int,
                   help="生成批量；默认 MPS=1、CUDA=8、CPU=1")
    p.add_argument("--device", default="", help="cuda / mps / cpu；留空自动检测")
    p.add_argument("--schema", default="", help="事件 schema 路径；默认优先使用数据文件同目录的 schema")
    p.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"],
                   help="本地模型精度；auto = 有 CUDA 用 bf16")
    p.add_argument("--max_length", default=4096, type=int, help="完整 prompt 的 token 上限，不含新生成 token")
    p.add_argument("--retries", default=3, type=int, help="API 失败重试次数")
    p.add_argument("--dry_run", "--dry-run", dest="dry_run", action="store_true",
                   help="离线演练：不联网不占 GPU，用假模型把整条流水线跑通（数字无意义）")
    p.add_argument("--print_prompt", "--print-prompt", dest="print_prompt",
                   action="store_true", help="只打印提示词全文然后退出")
    p.add_argument("--no_eval", action="store_true", help="只生成预测，不跑统一评测")
    p.add_argument("--resume", action="store_true", help="API 断点续跑：复用成功回复，只调用未完成或失败的样本")
    p.add_argument("--from_raw", default="",
                   help="不调模型，只把已有的 raw_*.jsonl 用**当前解析器**重新解析一遍"
                        "（改了容错逻辑后不必重新花 API 成本/GPU 时间，语义与重跑完全一致）")
    args = p.parse_args()
    if args.n < 0 or any(value < 1 for value in [args.max_new_tokens, args.max_length,
                                                args.max_tokens, args.retries]):
        p.error("n 应为非负数，token 上限、长度和重试次数必须大于 0")
    if args.gen_batch is not None and args.gen_batch < 1:
        p.error("gen_batch 必须大于 0")
    if args.resume and (args.backend != "api" or args.from_raw or args.dry_run):
        p.error("--resume 仅用于 API 推理；重解析使用 --from_raw")
    return args

def main():
    global SCHEMA
    args = parse_args()
    if args.dry_run:
        args.backend = "mock"
    data_path = Path(args.data).expanduser().resolve() if args.data else DATA_DIR / "dev_extract.jsonl"
    schema_path = Path(args.schema).expanduser().resolve() if args.schema else data_path.with_name("event_schema.json")
    if not schema_path.exists() and not args.schema:
        schema_path = DATA_DIR / "event_schema.json"
    SCHEMA = load_schema(schema_path)
    if args.print_prompt:
        print(json.dumps(build_messages("待抽取公告原文", SCHEMA, few_shot=args.mode == "few"),
                         ensure_ascii=False, indent=2))
        return
    with data_path.open(encoding="utf-8") as stream:
        all_rows = [json.loads(line) for line in stream if line.strip()]
    rows = sample_rows(all_rows, args.n or len(all_rows), args.seed)
    if not rows:
        raise ValueError("评测数据为空。")
    if len({uid(row) for row in rows}) != len(rows):
        raise ValueError("评测样本存在重复的 id|sent_id。")
    if args.tag:
        tag = args.tag
    elif args.from_raw:
        tag = Path(args.from_raw).stem.removeprefix("raw_") + "_reparse"
    else:
        prefix = "qwen_base" if args.backend == "local" else ("deepseek_v41" if args.backend == "api" and args.model == "deepseek-flash" else args.backend)
        tag = f"{prefix}_{args.mode}_n{len(rows)}"
    if tag in (".", "..") or any(char in tag for char in ("/", "\\")):
        raise ValueError("tag 应为文件名前缀，不能包含路径分隔符。")
    args.tag = tag
    args.data = str(data_path)
    args.schema = str(schema_path)
    raw_path = OUTPUT_LLM_DIR / f"raw_{tag}.jsonl"
    pred_path = Path(args.out).expanduser().resolve() if args.out else OUTPUT_LLM_DIR / f"pred_{tag}.jsonl"
    gold_subset = OUTPUT_LLM_DIR / f"gold_{tag}.jsonl"
    eval_ids_path = OUTPUT_LLM_DIR / f"eval_ids_{tag}.json"
    report_path = OUTPUT_LLM_DIR / f"report_{tag}.json"
    metrics_path = ROOT / "outputs" / "metrics" / "llm" / f"{tag}.json"
    raw_map, source_report = {}, {}
    if args.from_raw:
        source = Path(args.from_raw).expanduser().resolve()
        if source == raw_path.resolve():
            raise ValueError("重解析不能覆盖原 raw 文件，请使用新的 --tag。")
        with source.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    record = json.loads(line)
                    key = uid(record)
                    if key in raw_map:
                        raise ValueError(f"raw 中样本键重复：{key}")
                    raw_map[key] = record
        missing = [uid(row) for row in rows if uid(row) not in raw_map]
        if missing:
            raise ValueError(f"raw 缺少本次采样中的 {len(missing)} 条样本，例如 {missing[0]}。"
                             "请保持 --data、--n、--seed 与原推理一致。")
        source_report_path = source.with_name(source.name.replace("raw_", "report_", 1)).with_suffix(".json")
        if source_report_path.is_file():
            source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
        args.from_raw = str(source)
    protected_paths = {data_path.resolve(), raw_path.resolve(), gold_subset.resolve()}
    if args.from_raw:
        protected_paths.add(Path(args.from_raw))
    if pred_path.resolve() in protected_paths:
        raise ValueError("预测输出不能覆盖数据、金标子集或 raw 文件。")

    cached_api = {}
    client = None
    reused_samples = new_api_attempts = 0
    api_usage = empty_usage()
    run_manifest = OUTPUT_LLM_DIR / f"request_{tag}.json"
    if args.backend == "api" and not args.from_raw:
        configuration = {"request": request_settings(args.model, args.max_tokens), "mode": args.mode,
                         "seed": args.seed, "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
                         "base_url": os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL),
                         "sample_ids": [uid(row) for row in rows],
                         "messages_sha256": hashlib.sha256(json.dumps(
                             [build_messages(row["text"], SCHEMA, args.mode == "few") for row in rows],
                             ensure_ascii=False, sort_keys=True).encode()).hexdigest()}
        if raw_path.exists():
            if not args.resume:
                raise FileExistsError(f"API 回复已存在：{raw_path}。请用 --resume 续跑或使用新的 --tag。")
            if not run_manifest.is_file() or json.loads(run_manifest.read_text()) != configuration:
                raise ValueError("断点记录的样本、模型、提示词或请求参数与本次不同，请使用新的 --tag。")
            for line in raw_path.read_text().splitlines():
                if line.strip():
                    record = json.loads(line)
                    if uid(record) in cached_api:
                        raise ValueError("API 断点记录存在重复样本键。")
                    cached_api[uid(record)] = record
            if set(cached_api) - {uid(row) for row in rows}:
                raise ValueError("API 断点记录包含当前评测集以外的样本。")
        reusable = {key for key, record in cached_api.items() if record.get("raw", "").strip() and not record.get("error")}
        if len(reusable) < len(rows):
            client = build_api_client()  # 缺密钥/SDK 时先报错，不写出结果，也不清空日志。
    source_is_mock = source_report.get("is_mock", any(raw_map.get(uid(row), {}).get("is_mock") for row in rows))
    device = None
    if args.backend == "local" and not args.from_raw:
        device = select_device(args.device)
        args.device = str(device)
        args.model_path = str(resolve_model_path(args.model_path))
        if args.gen_batch is None:
            args.gen_batch = 8 if device.type == "cuda" else 1
    if args.gen_batch is None:
        args.gen_batch = 1
    OUTPUT_LLM_DIR.mkdir(parents=True, exist_ok=True)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    write_eval_ids(eval_ids_path, rows, args.seed)
    write_gold_subset(gold_subset, rows)

    if args.backend == "api" and not args.from_raw:
        run_manifest.write_text(json.dumps(configuration, ensure_ascii=False, indent=2), encoding="utf-8")
    with EvalLogger(method=tag, args=vars(args), root=ROOT,
                    notes=f"LLM {args.backend}/{args.mode} 事件抽取；复用课程 prompt 与统一评测") as eval_logger:
        print(f"数据：{data_path} → 固定采样 {len(rows)} 条（seed={args.seed}）")
        print(f"输出：{pred_path}")
        local_model = local_tok = None
        if args.from_raw:
            print(f"[from_raw] 仅重解析已有 {len(raw_map)} 条回复，不调用模型")
        elif args.backend == "local":
            dtype = resolve_torch_dtype(device, args.dtype)
            print(f"[local] 模型={args.model_path} 设备={device} dtype={dtype} batch={args.gen_batch}")
            print("保留完整提示词；关闭思考；贪心解码；不做微调。")
            if device.type == "cuda":
                check_gpu_idle()
            local_model, local_tok = load_causal_lm(args.model_path, device, dtype)
        elif args.backend == "api":
            print(f"[api] model={args.model} max_tokens={args.max_tokens} JSON 模式，复用 {len(reusable)} 条回复")
        else:
            print("[mock] 只验证流程，使用模拟回复；指标不能用于实验结论。")

        keys = ("empty_reply", "json_ok", "repaired", "noise_stripped", "type_invalid",
                "role_invalid", "trigger_not_in_source", "arg_not_in_source")
        stats = {key: 0 for key in keys}
        latencies, errors = [], []
        prompt_tokens = completion_tokens = capped = 0
        step = args.gen_batch if args.backend == "local" and not args.from_raw else 1
        started = time.perf_counter()
        with pred_path.open("w", encoding="utf-8") as f_pred, (nullcontext(None) if args.backend == "api" and not args.from_raw else raw_path.open("w", encoding="utf-8")) as f_raw:
            for start in range(0, len(rows), step):
                chunk = rows[start:start + step]
                messages = [build_messages(row["text"], SCHEMA, few_shot=args.mode == "few") for row in chunk]
                if args.from_raw:
                    outs = [{"text": raw_map[uid(row)].get("raw", ""),
                             **{key: raw_map[uid(row)].get(key, default) for key, default in
                                [("prompt_tokens", 0), ("completion_tokens", 0), ("latency_s", 0.0),
                                 ("error", ""), ("hit_max_new_tokens", False)]}} for row in chunk]
                elif args.backend == "local":
                    outs = generate_batch(local_model, local_tok, messages, device,
                                          max_new_tokens=args.max_new_tokens, batch_size=args.gen_batch,
                                          max_length=args.max_length)
                elif args.backend == "api":
                    outs = []
                    for row, message in zip(chunk, messages):
                        previous = cached_api.get(uid(row), {})
                        if uid(row) in reusable:
                            reused_samples += 1
                            out = {"text": previous["raw"], **previous, "_cached": True}
                        else:
                            out = call_api(client, message, args.model, args.max_tokens, args.retries)
                            new_api_attempts += out["api_attempts"]
                            if previous:
                                out["api_usage"] = add_usage(previous.get("api_usage", {}), out["api_usage"])
                                out["api_attempts"] += previous.get("api_attempts", 0)
                                out["attempt_history"] = previous.get("attempt_history", []) + out["attempt_history"]
                                out["latency_s"] += previous.get("latency_s", 0)
                                out["prompt_tokens"] = out["api_usage"]["prompt_tokens"]
                                out["completion_tokens"] = out["api_usage"]["completion_tokens"]
                        outs.append(out)
                else:
                    outs = [{"text": mock_generate(row, start + k + 1, args.seed),
                             "prompt_tokens": 0, "completion_tokens": 0, "latency_s": 0.0}
                            for k, row in enumerate(chunk)]
                if len(outs) != len(chunk):
                    raise RuntimeError("生成结果条数与输入样本数不一致。")
                for row, out in zip(chunk, outs):
                    if out.get("error"):
                        errors.append({"uid": uid(row), "error": out["error"]})
                    events, info = parse_model_output(out["text"], row["text"], SCHEMA)
                    for key in keys:
                        stats[key] += int(info.get(key, 0) or 0)
                    latencies.append(float(out.get("latency_s") or 0.0))
                    prompt_tokens += int(out.get("prompt_tokens") or 0)
                    completion_tokens += int(out.get("completion_tokens") or 0)
                    capped += int(out.get("hit_max_new_tokens", False))
                    f_pred.write(json.dumps(make_pred_row(row, events), ensure_ascii=False) + "\n")
                    record = {"id": row.get("id"), "sent_id": row.get("sent_id"),
                              "raw": out["text"], "events": events,
                              "latency_s": float(out.get("latency_s") or 0.0),
                              "prompt_tokens": int(out.get("prompt_tokens") or 0),
                              "completion_tokens": int(out.get("completion_tokens") or 0),
                              "hit_max_new_tokens": bool(out.get("hit_max_new_tokens", False)),
                              "error": out.get("error", ""),
                              "is_mock": source_is_mock if args.from_raw else args.backend == "mock"}
                    if args.backend == "api" and not args.from_raw:
                        for field in ("response_model", "response_id", "finish_reason", "api_usage",
                                      "api_attempts", "attempt_history"):
                            record[field] = out.get(field)
                        api_usage = add_usage(api_usage, out.get("api_usage", {}))
                        cached_api[uid(row)] = record
                        write_raw_checkpoint(raw_path, cached_api, rows)
                        if out.get("fatal"):
                            raise RuntimeError(f"{out['error']}：请检查 API 密钥、余额或请求参数；已保存断点。")
                    else:
                        f_raw.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_pred.flush()
                if f_raw is not None:
                    f_raw.flush()
                done = min(start + step, len(rows))
                memory_info = ""
                if device is not None and device.type == "mps":
                    import torch
                    memory_info = f"，MPS 当前分配 {torch.mps.current_allocated_memory()/1024**2:.0f} MiB"
                print(f"  [{done:3d}/{len(rows)}] JSON 合法 {stats['json_ok']} 条，"
                      f"平均 {(time.perf_counter() - started) / done:.2f}s/条{memory_info}", flush=True)
        runtime = time.perf_counter() - started
        n = len(rows)
        model_name = args.model_path if args.backend == "local" else (args.model if args.backend == "api" else "mock")
        is_mock = args.backend == "mock"
        if args.from_raw:
            is_mock = source_is_mock
            model_name = source_report.get("model", "unknown_from_raw")
        report = {
            "tag": tag, "backend": "from_raw" if args.from_raw else args.backend,
            "mode": source_report.get("mode", args.mode) if args.from_raw else args.mode,
            "n": n, "model": model_name, "seed": args.seed, "is_mock": is_mock,
            "few_shot_examples": len(FEW_SHOT_EXAMPLES) if (source_report.get("mode", args.mode) == "few") else 0,
            "json_valid_rate": _pct(stats["json_ok"], n), "parse_fail": n - stats["json_ok"],
            **{key: stats[key] for key in keys if key != "json_ok"},
            "latency_mean_s": round(sum(latencies) / n, 3),
            "latency_p50_s": round(sorted(latencies)[n // 2], 3),
            "total_s": round(runtime, 2), "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens,
            "hit_max_new_tokens": capped, "api_errors": len(errors), "errors": errors,
            "pred_path": str(pred_path), "gold_subset": str(gold_subset), "eval_ids": str(eval_ids_path),
            "data": str(data_path), "schema": str(schema_path), "device": args.device,
            "dtype": args.dtype, "max_length": args.max_length, "max_new_tokens": args.max_new_tokens,
            "gen_batch": args.gen_batch, "prompt_truncated": 0,
            "latency_note": "批量生成耗时摊销到每句；不包含模型加载",
        }
        if args.backend == "api" and not args.from_raw:
            report.update(api_usage=api_usage, max_tokens=args.max_tokens,
                          generation_total_s=round(sum(latencies), 3),
                          api_attempts=sum(r.get("api_attempts", 0) for r in cached_api.values()),
                          new_api_attempts=new_api_attempts, reused_samples=reused_samples,
                          response_models=sorted({r.get("response_model", "") for r in cached_api.values()}),
                          cost=cost_estimate(api_usage), request_manifest=str(run_manifest))
        if args.from_raw:
            report.update(source_raw=args.from_raw, parse_runtime_s=round(runtime, 2),
                          generation_total_s=sum(latencies),
                          token_counts_available=all("prompt_tokens" in raw_map[uid(row)] for row in rows))
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"统计 → {report_path}")
        metrics = {}
        if not args.no_eval:
            result = run_shared_evaluation(gold_subset, pred_path, metrics_path)
            result.update(method=tag, gold=str(gold_subset), pred=str(pred_path),
                          meta={"is_mock": is_mock, "model": model_name, "mode": report["mode"],
                                "num_samples": n, "report": str(report_path), "seed": args.seed})
            metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            _print_metrics(result, report)
            print(f"统一评测 → {metrics_path}")
            metrics = {key: result[key]["f1"] for key in ("trigger", "argument", "event_exact")}
        eval_logger.finalize(metrics=metrics, report=report, runtime_s=runtime)


if __name__ == "__main__":
    main()
