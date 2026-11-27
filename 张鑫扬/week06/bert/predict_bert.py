"""
方法三（推理/评测）：加载 RoBERTa 领域分类 checkpoint，在 **test 切分** 上评测

训练目标：中文文本领域分类（18 个国民经济行业大类）
数据：train_data/zh.test.csv → common/domain_data.py 按 8:1:1 分层切分
      train 训练 / val 选模型 / test 只在最终评测时用一次。

教学重点：
  1. 最终指标必须报在**没参与训练和选模型**的 test 上；val 上的分数是调参用的，
     拿 val 分数当最终成绩等于自己给自己判卷。
  2. 18 类均衡数据上 Accuracy 和 Macro-F1 应该接近；两者差很多说明某些类被吃掉了。
  3. 顺手测延迟：单条 vs batch 差几倍——「模型多大」不是成本，「能不能批处理」才是。
  4. 保存逐条预测，方便看混淆矩阵、做错误分析。

用法：
  python bert/predict_bert.py                        # 默认 outputs/roberta_domain_ckpt.pt，评 test
  python bert/predict_bert.py --split val            # 改评验证集
  python bert/predict_bert.py --n 500                # 固定 seed 抽样 500 条
  python bert/predict_bert.py --ckpt path/to/ckpt.pt --out outputs/foo.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.domain_data import DOMAIN_LABELS, load_domain_splits, sample_subset
from common.metrics import classification_report, expected_calibration_error, to_markdown_table
from common.model_paths import roberta_path
from train_bert import RobertaClassifier

CODE_ROOT = Path(__file__).resolve().parent.parent


@torch.no_grad()
def predict(model, tok, texts, device, max_length=64, batch_size=32):
    """返回 (preds, probs, 每样本毫秒数)"""
    preds, probs = [], []
    t0 = time.time()
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], truncation=True, max_length=max_length,
                  padding="max_length", return_tensors="pt")
        logits = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        p = torch.softmax(logits.float(), dim=-1)
        preds.extend(p.argmax(-1).cpu().tolist())
        probs.extend(p.cpu().tolist())
    ms = (time.time() - t0) * 1000 / max(1, len(texts))
    return preds, probs, round(ms, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(CODE_ROOT / "outputs" / "roberta_domain_ckpt.pt"))
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--n", type=int, default=0, help="抽样条数，0=全部")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_length", type=int, default=64)
    ap.add_argument("--out", default="outputs/roberta_domain_pred.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)  # torch 2.6+ 默认 weights_only=True 会拒载字典
    labels = ckpt.get("labels") or DOMAIN_LABELS
    if list(labels) != list(DOMAIN_LABELS):
        print(f"[warn] checkpoint 标签与当前 DOMAIN_LABELS 不一致：{labels}", flush=True)

    from transformers import AutoTokenizer
    model_dir = Path(ckpt["model_dir"]) if ckpt.get("model_dir") and Path(ckpt["model_dir"]).is_dir() else roberta_path()
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = RobertaClassifier(model_dir, num_labels=len(labels), pool=ckpt.get("pool", "cls"))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    train, val, test = load_domain_splits()
    samples = {"train": train, "val": val, "test": test}[args.split]
    samples = sample_subset(samples, args.n)
    texts = [d["text"] for d in samples]
    preds, probs, ms = predict(model, tok, texts, device,
                               max_length=args.max_length, batch_size=args.batch_size)

    y_true = [d["label"] for d in samples]
    report = classification_report(y_true, preds, num_classes=len(labels), label_names=labels)
    report.update({
        "method": "roberta_domain_finetune",
        "model": ckpt.get("base_model", "roberta-base-chinese"),
        "pool": ckpt.get("pool"),
        "split": args.split,
        "ckpt_val_f1": ckpt.get("val_macro_f1"),
        "ms_per_sample": ms,
        "ece": expected_calibration_error(y_true, probs),
    })
    meta = {"n_eval": len(samples), "split": args.split, "labels": labels,
            "n_train": len(train), "n_val": len(val), "n_test": len(test)}

    out = CODE_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "report": report,
                               "preds": [{"text": t, "true": y, "pred": p, "true_name": labels[y],
                                          "pred_name": labels[p]}
                                         for t, y, p in zip(texts, y_true, preds)]},
                              ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"评测 {len(samples)} 条（{args.split} 切分）| {ms} ms/条 | ECE={report['ece']}")
    print(to_markdown_table({"RoBERTa 领域分类": report}))

    # 逐类召回：直接看出哪几个行业最难分
    print("\n| 领域 | Recall |\n|---|---|")
    for name, r in report["per_class_recall"].items():
        print(f"| {name} | {'-' if r is None else r} |")
    print(f"\n→ 结果写入 {out}")


if __name__ == "__main__":
    main()
