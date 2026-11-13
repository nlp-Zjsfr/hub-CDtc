from common import ROOT, config, write_json
from pathlib import Path
import shutil


def download():
    from huggingface_hub import snapshot_download
    cfg = config()
    snapshot = snapshot_download(cfg["model_id"], revision=cfg["revision"],
                      allow_patterns=["config.json", "generation_config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "LICENSE"], max_workers=2)
    shutil.copyfile(Path(snapshot) / "LICENSE", ROOT / "models" / "LICENSE-base.txt")
    write_json(ROOT / "models" / "base_model.json", {"model_id": cfg["model_id"], "revision": cfg["revision"],
               "license": "Apache-2.0", "url": "https://huggingface.co/" + cfg["model_id"],
               "storage": "Hugging Face cache outside the homework directory"})
    print("Base model cache ready:", cfg["model_id"], cfg["revision"], flush=True)


if __name__ == "__main__":
    download()
