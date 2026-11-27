import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

DATASET_NAME = "sookiemonster/asrs-narratives-rebalance"
SPLIT = "test"

# 获取项目根目录 (homework/)
ROOT_DIR = Path(__file__).resolve().parent.parent


class BertWithPooling(nn.Module):
    def __init__(self, encoder, num_labels, pooling="cls", dropout=0.1):
        super().__init__()
        assert pooling in {"cls", "mean", "max"}
        self.encoder = encoder
        self.pooling = pooling
        hidden = encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state
        if self.pooling == "cls":
            x = h[:, 0]
        elif self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            x = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        else:
            mask = attention_mask.unsqueeze(-1).bool()
            x = h.masked_fill(~mask, torch.finfo(h.dtype).min).max(1).values
        return self.classifier(self.dropout(x))


# ================= 与训练一致的滑动窗口 Dataset =================
class AsrsDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length, stride=128):
        self.input_ids = []
        self.attention_masks = []
        self.labels = []
        self.acns = []
        
        texts = dataset["text"]
        labels = dataset["label"]
        acns = dataset["acn"]

        for text, label, acn in zip(texts, labels, acns):
            # 开启滑动窗口
            enc = tokenizer(
                text, 
                truncation=True, 
                max_length=max_length,
                stride=stride, 
                return_overflowing_tokens=True
            )
            for j in range(len(enc["input_ids"])):
                self.input_ids.append(enc["input_ids"][j])
                self.attention_masks.append(enc["attention_mask"][j])
                self.labels.append(label)
                self.acns.append(acn)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return {
            "input_ids": self.input_ids[i],
            "attention_mask": self.attention_masks[i],
            "label": self.labels[i],
            "acn": self.acns[i]
        }


def collate_fn(batch, pad_token_id):
    max_len = max(len(b["input_ids"]) for b in batch)
    ids, mask, labels = [], [], []
    acns = []
    for b in batch:
        pad = max_len - len(b["input_ids"])
        ids.append(b["input_ids"] + [pad_token_id] * pad)
        mask.append(b["attention_mask"] + [0] * pad)
        labels.append(b["label"])
        acns.append(b["acn"])
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "attention_mask": torch.tensor(mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "acn": acns
    }


def autocast_ctx(enabled):
    if not enabled:
        from contextlib import nullcontext
        return nullcontext()
    try:
        return torch.autocast(device_type="cuda", enabled=True)
    except AttributeError:
        return torch.cuda.amp.autocast(enabled=True)


# ================= 与训练一致的聚合预测函数 =================
@torch.no_grad()
def predict_aggregated(model, loader, device, use_fp16):
    model.eval()
    all_acns, all_labels, all_logits = [], [], []
    
    for batch in tqdm(loader, desc="模型推理中"):
        acn = batch.pop("acn")
        labels = batch["labels"]
        batch = {k: v.to(device) for k, v in batch.items()}
        
        with autocast_ctx(use_fp16 and device.type == "cuda"):
            logits = model(batch["input_ids"], batch["attention_mask"])
            
        all_acns.extend(acn)
        all_labels.extend(labels.tolist())
        all_logits.append(logits.cpu())
        
    all_logits = torch.cat(all_logits, dim=0)
    
    # 按 acn 分组软投票
    acn_logits_dict = defaultdict(list)
    acn_labels_dict = {}
    
    for i, acn in enumerate(all_acns):
        acn_logits_dict[acn].append(all_logits[i])
        acn_labels_dict[acn] = all_labels[i]
        
    final_preds, final_trues = [], []
    
    for acn, logits_list in acn_logits_dict.items():
        avg_logits = torch.stack(logits_list).mean(dim=0)
        pred = avg_logits.argmax().item()
        
        final_preds.append(pred)
        final_trues.append(acn_labels_dict[acn])
        
    return final_trues, final_preds


def evaluate(y_true, y_pred, num_classes):
    n = len(y_true)
    unparsed = sum(1 for p in y_pred if p is None or p < 0)
    conf = [[0] * num_classes for _ in range(num_classes)]
    for t, p in zip(y_true, y_pred):
        if p is not None and p >= 0:
            conf[t][p] += 1
    correct = sum(conf[i][i] for i in range(num_classes))
    parsed = n - unparsed
    acc = correct / parsed if parsed else 0.0
    acc_incl = correct / n if n else 0.0
    
    per_recall, f1s, supports = {}, [], []
    for i in range(num_classes):
        tp = conf[i][i]
        row = sum(conf[i])
        col = sum(conf[r][i] for r in range(num_classes))
        recall = tp / row if row else 0.0
        precision = tp / col if col else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_recall[str(i)] = round(recall, 4)
        f1s.append(f1)
        supports.append(row)
        
    macro_f1 = sum(f1s) / num_classes if num_classes else 0.0
    weighted_f1 = sum(f * s for f, s in zip(f1s, supports)) / n if n else 0.0
    counts = {}
    for p in y_pred:
        key = str(p) if (p is not None and p >= 0) else "-1"
        counts[key] = counts.get(key, 0) + 1
    used = sum(1 for k, v in counts.items() if k != "-1" and v > 0)
    
    return {
        "n": n,
        "unparsed": unparsed,
        "unparsed_rate": round(unparsed / n, 4) if n else 0.0,
        "acc": round(acc, 4),
        "acc_incl_unparsed": round(acc_incl, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class_recall": per_recall,
        "confusion_matrix": conf,
        "pred_distribution": {"used_classes": used, "counts": counts},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="outputs/bert_model/best",
                    help="相对 homework/ 的模型目录（best 或 last）")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--n_test", type=int, default=-1)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--out", default="outputs/bert_pred.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[设备] {device}")

    model_dir = ROOT_DIR / args.model_dir
    if not (model_dir / "model.pt").exists():
        raise FileNotFoundError(f"未找到模型: {model_dir/'model.pt'}，请先运行 bert_train_set.py")

    # 读取配置，自动同步 max_length 和 stride
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    pooling, num_classes = cfg["pooling"], cfg["num_labels"]
    label_names = cfg["label_names"]
    max_length = cfg.get("max_length", 512)
    # 从保存的 train_args 中提取 stride，如果没找到默认用 128
    stride = cfg.get("train_args", {}).get("stride", 128)
    
    print(f"[加载模型] {model_dir} | pooling={pooling} | "
          f"best_val_macro_f1={cfg.get('best_val_macro_f1')}")
    print(f"[滑动窗口配置] max_length={max_length}, stride={stride} (自动从 config 读取)")

    encoder_cfg = AutoConfig.from_pretrained(model_dir / "encoder_config")
    encoder = AutoModel.from_config(encoder_cfg)
    model = BertWithPooling(encoder, num_classes, pooling=pooling)
    state = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer")

    print(f"[加载] {DATASET_NAME} [{SPLIT}] ...")
    ds = load_dataset(DATASET_NAME)
    test = ds[SPLIT]
    n_test = len(test) if args.n_test < 0 else min(args.n_test, len(test))
    test = test.select(range(n_test))

    # 构建滑动窗口测试集
    print(f"[预处理] 正在切分测试集 ...")
    test_dataset = AsrsDataset(test, tokenizer, max_length, stride)
    print(f"[预处理] 测试集样本切片: {len(test)} -> {len(test_dataset)}")
    
    loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size, shuffle=False, num_workers=0,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
    )

    t0 = time.time()
    # 使用聚合预测
    trues, preds = predict_aggregated(model, loader, device, args.fp16)
    dt = time.time() - t0

    report = evaluate(trues, preds, num_classes)
    report["method"] = f"bert_{pooling}_sliding_window"
    report["sec_per_item"] = round(dt / n_test, 4) if n_test else 0.0

    meta = {
        "dataset": DATASET_NAME,
        "split": SPLIT,
        "n_train": len(ds["train"]),
        "n_val": len(ds["validation"]),
        "n_test": len(ds["test"]),
        "n_eval": n_test,
        "seed": cfg["train_args"].get("seed", 42),
        "model_dir": str(model_dir),
        "window_max_length": max_length,
        "window_stride": stride,
    }

    # 将原始文本、真实标签和预测标签对齐。由于 test 里是按 acn 排列的，可以直接 zip
    preds_out = [
        {"text": t, "true": int(yt), "pred": int(yp)}
        for t, yt, yp in zip(test["text"], trues, preds)
    ]

    out = ROOT_DIR / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"meta": meta, "report": report, "preds": preds_out},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n评测 {n_test} 条 | acc={report['acc']:.4f} | "
          f"macro_f1={report['macro_f1']:.4f} | weighted_f1={report['weighted_f1']:.4f}")
    print(f"覆盖类别: {report['pred_distribution']['used_classes']}/{num_classes}")
    print(f"→ 结果写入 {out}")


if __name__ == "__main__":
    main()