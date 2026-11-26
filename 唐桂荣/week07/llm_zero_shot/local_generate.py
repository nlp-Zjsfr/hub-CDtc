"""本地 Qwen 批量贪心生成，供 zero/few-shot 和后续 LoRA 评测共用。"""

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria


def select_device(name="") -> torch.device:
    """自动选择 CUDA、Apple MPS、CPU，也可显式指定。"""
    if name:
        device = torch.device(name)
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("指定了 MPS，但当前 PyTorch 环境无法使用 MPS。")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("指定了 CUDA，但当前 PyTorch 环境无法使用 CUDA。")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_torch_dtype(device: torch.device, dtype: str = "auto") -> torch.dtype:
    """CUDA 自动使用支持的半精度，MPS/CPU 默认 fp32；支持显式覆盖。"""
    if dtype == "auto":
        if device.type == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]


def load_causal_lm(model_path: str, device: torch.device, dtype: torch.dtype = torch.float32):
    """加载完整本地模型；decoder-only 生成采用左 padding 和显式 attention_mask。"""
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    if not tokenizer.chat_template:
        raise ValueError("模型目录缺少 chat template，请保留作业提供的 chat_template.jinja。")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=dtype, local_files_only=True,
    ).to(device).eval()
    return model, tokenizer


def batch_messages(tokenizer, messages_list):
    """训练和推理共用 chat 格式，关闭 Qwen3 的思考段。"""
    return [tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    ) for messages in messages_list]


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


class _MpsCacheCleanup(StoppingCriteria):
    """长回复生成中也周期性释放未使用缓存，保持输出停止条件不变。"""

    def __init__(self):
        self.steps = 0
        self.max_allocated = 0
        self.max_driver = 0

    def __call__(self, input_ids, scores, **kwargs):
        self.steps += 1
        self.max_allocated = max(self.max_allocated, torch.mps.current_allocated_memory())
        self.max_driver = max(self.max_driver, torch.mps.driver_allocated_memory())
        if self.steps % 32 == 0:
            torch.mps.empty_cache()
        return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)


@torch.inference_mode()
def generate_batch(model, tokenizer, messages_list, device, max_new_tokens=384,
                   batch_size=1, max_length=4096) -> list[dict]:
    """完整保留 prompt，批量生成，返回逐句原文、token 数和摊销延迟。

    max_length 只限制提示词 token 数，不扣除 max_new_tokens。
    超长时明确报错，避免静默删除 schema/few-shot 示例后仍把结果标为原实验。
    """
    if batch_size < 1 or max_new_tokens < 1 or max_length < 1:
        raise ValueError("batch_size、max_new_tokens、max_length 必须大于 0。")
    eos_ids = model.generation_config.eos_token_id
    eos_ids = [eos_ids] if isinstance(eos_ids, int) else list(eos_ids or [])
    if tokenizer.eos_token_id is not None:
        eos_ids.append(tokenizer.eos_token_id)
    # Base 的 EOS 是 endoftext；聊天格式还需在 im_end 处停止，避免续写下一轮对话。
    im_end = tokenizer.get_vocab().get("<|im_end|>")
    if im_end is not None:
        eos_ids.append(im_end)
    eos_ids = sorted(set(eos_ids))
    results = []
    for start in range(0, len(messages_list), batch_size):
        chunk = messages_list[start:start + batch_size]
        prompts = batch_messages(tokenizer, chunk)
        enc = tokenizer(prompts, return_tensors="pt", padding=True,
                        add_special_tokens=False, truncation=False)
        prompt_length = enc["input_ids"].shape[1]
        if prompt_length > max_length:
            raise ValueError(f"提示词长 {prompt_length} tokens，超过 --max_length {max_length}；"
                             "请调大上限，提示词未被截断。")
        context_limit = getattr(model.config, "max_position_embeddings", None)
        if context_limit and prompt_length + max_new_tokens > context_limit:
            raise ValueError(f"提示词与输出上限合计 {prompt_length + max_new_tokens}，"
                             f"超过模型上下文上限 {context_limit}。")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        _synchronize(device)
        start_time = time.perf_counter()
        cleanup = _MpsCacheCleanup() if device.type == "mps" else None
        generated = model.generate(
            input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_ids or None, use_cache=True,
            **({"stopping_criteria": [cleanup]} if cleanup else {}),
        )
        _synchronize(device)
        latency = time.perf_counter() - start_time
        paths = generated[:, prompt_length:].cpu().tolist()
        lengths = attention_mask.sum(-1).cpu().tolist()
        for path, n_prompt in zip(paths, lengths):
            stop = next((i for i, token in enumerate(path) if token in eos_ids), None)
            # 不把批内其他句子尚未结束时补的 PAD 算入 token。
            token_ids = path if stop is None else path[:stop + 1]
            results.append({
                "text": tokenizer.decode(token_ids, skip_special_tokens=True).strip(),
                "latency_s": latency / len(chunk),
                "prompt_tokens": int(n_prompt), "completion_tokens": len(token_ids),
                "hit_max_new_tokens": stop is None and len(path) >= max_new_tokens,
                "observed_mps_allocated_mb": round(cleanup.max_allocated / 1024**2, 1) if cleanup else None,
                "observed_mps_driver_mb": round(cleanup.max_driver / 1024**2, 1) if cleanup else None,
            })
        # MPS 的缓存可能跨不同长度的生成批次累积；每批结束后释放不再使用的张量和缓存。
        del generated, input_ids, attention_mask, enc
        if device.type == "mps":
            torch.mps.empty_cache()
    return results
