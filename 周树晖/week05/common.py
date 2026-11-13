import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
ROOT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8-sig").splitlines() if s.strip()]


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in rows), encoding="utf-8")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def config():
    return read_json(ROOT / "config.json")


def load_tokenizer(cfg=None):
    from transformers import AutoTokenizer
    cfg = cfg or config()
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], revision=cfg["revision"], local_files_only=True, trust_remote_code=False)
    tokenizer.chat_template = (ROOT / "chat_template.jinja").read_text(encoding="utf-8")
    tokenizer.pad_token = "<|endoftext|>"
    return tokenizer
