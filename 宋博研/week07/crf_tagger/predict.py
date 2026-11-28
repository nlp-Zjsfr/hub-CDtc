"""
用训练好的 RoBERTa+CRF 对任意句子做 NER（演示用）。

  python crf_tagger/predict.py "浙商银行企业信贷部叶老桂博士则从另一个角度进行了解读。"
  python crf_tagger/predict.py --hard "..." "..."
"""
import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "crf_tagger"))
from train import CharEncoder, RobertaCRF, load_vocab, make_batches, pick_device, predict  # noqa: E402
from shared.data import LABEL_ZH  # noqa: E402
from shared.paths import CKPT_DIR, ROBERTA_DIR  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("texts", nargs="+")
    ap.add_argument("--ckpt", default=str(CKPT_DIR / "crf_tagger" / "best.pt"))
    ap.add_argument("--model_dir", default=str(ROBERTA_DIR))
    ap.add_argument("--hard", action="store_true", help="硬约束维特比")
    ap.add_argument("--device", default="")
    a = ap.parse_args()

    device = pick_device(a.device)
    model = RobertaCRF(a.model_dir).to(device)
    model.load_state_dict(torch.load(a.ckpt, map_location=device))
    rows = [{"id": str(i), "text": t} for i, t in enumerate(a.texts)]
    preds, _, _ = predict(model, make_batches(rows, CharEncoder(load_vocab(a.model_dir)), 32, False), device, hard=a.hard)
    for p in preds:
        print(p["text"])
        for e in p["entities"]:
            print(f"  [{LABEL_ZH[e['label']]}] {e['text']}  ({e['start']},{e['end']})")


if __name__ == "__main__":
    main()
