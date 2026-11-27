import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM

from common import ROOT


def base_model(cfg, device):
    return AutoModelForCausalLM.from_pretrained(
        cfg["model_id"], revision=cfg["revision"], local_files_only=True,
        trust_remote_code=False, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        attn_implementation="sdpa").to(device)


def trainable_model(cfg, device):
    model = get_peft_model(base_model(cfg, device), LoraConfig(
        task_type="CAUSAL_LM", r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"], target_modules=cfg["lora_target_modules"], bias="none", revision=cfg["revision"]))
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    assert all("lora_" in n for n, p in model.named_parameters() if p.requires_grad)
    return model


def inference_model(cfg, device):
    return PeftModel.from_pretrained(base_model(cfg, device), ROOT / "models" / "best_adapter", is_trainable=False).eval()
