# -*- coding: utf-8 -*-
"""
BERT + CRF + 硬约束 事件抽取验证脚本（对应 bert_crf_hard_train.py）

与 bert_crf_valid.py 的区别：
  - 加载的是 output/bert_crf_hard/bert_crf_hard_model.pt；
  - 恢复权重后会重新钉一次 BIO 硬约束，保证 Viterbi 解码不会产生非法序列
    （例如 O -> I-质押方、B-质押方 -> I-质押物）。

其余（数据源、BIO 解码、指标口径）与 bert_crf_valid.py 完全一致，
因此 bert / bert_crf / bert_crf_hard / bilstm 的数字可直接横向对比。
"""

import os
import json
from collections import defaultdict, Counter

import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import BertTokenizerFast, BertModel

MAX_LEN = 512
BATCH_SIZE = 32
DROPOUT = 0.1
NEG = -1e4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================= 硬约束 =================
def build_constraints(id2tag):
    """与 bert_crf_hard_train.py 完全相同：起始不能是 I-*；转移到 I-X 必须来自 B-X 或 I-X。"""
    start_forbidden = [j for j, t in id2tag.items() if t.startswith("I-")]

    trans_forbidden = []
    for i, from_tag in id2tag.items():
        for j, to_tag in id2tag.items():
            if not to_tag.startswith("I-"):
                continue
            role = to_tag[2:]
            if from_tag not in (f"B-{role}", f"I-{role}"):
                trans_forbidden.append((i, j))
    return start_forbidden, trans_forbidden


class BertCrfHardForTokenClassification(nn.Module):
    def __init__(self, bert: BertModel, id2tag, dropout: float = DROPOUT):
        super().__init__()
        num_labels = len(id2tag)
        self.bert = bert
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(bert.config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        self.num_labels = num_labels

        self.start_forbidden, self.trans_forbidden = build_constraints(id2tag)
        self.apply_constraints()

    @torch.no_grad()
    def apply_constraints(self):
        dev = self.crf.transitions.device
        if self.start_forbidden:
            idx = torch.as_tensor(self.start_forbidden, dtype=torch.long, device=dev)
            self.crf.start_transitions[idx] = NEG
        if self.trans_forbidden:
            rows = torch.as_tensor([a for a, _ in self.trans_forbidden], dtype=torch.long, device=dev)
            cols = torch.as_tensor([b for _, b in self.trans_forbidden], dtype=torch.long, device=dev)
            self.crf.transitions[rows, cols] = NEG

    def emissions(self, input_ids, attention_mask):
        seq_out = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.classifier(self.dropout(seq_out))

    def forward(self, input_ids, attention_mask, labels=None):
        emissions = self.emissions(input_ids, attention_mask)
        mask = attention_mask.bool()
        if labels is not None:
            tags = labels.clone()
            tags[tags < 0] = 0
            log_likelihood = self.crf(emissions, tags, mask=mask, reduction="sum")
            return -log_likelihood / mask.sum().clamp(min=1)
        return self.crf.decode(emissions, mask=mask)


# ================= 解码 / 拼事件 / 指标（与 bert_crf_valid.py 相同） =================
def decode_entities(text, preds, offsets, id2tag):
    ents, cur = [], None
    for i, pid in enumerate(preds):
        s, e = int(offsets[i][0]), int(offsets[i][1])
        if e <= s:
            if cur:
                ents.append(cur)
                cur = None
            continue
        tag = id2tag.get(int(pid), "O")
        if tag.startswith("B-"):
            if cur:
                ents.append(cur)
            cur = {"type": tag[2:], "start": s, "end": e}
        elif tag.startswith("I-"):
            if cur is not None and cur["type"] == tag[2:]:
                cur["end"] = e
            else:
                if cur:
                    ents.append(cur)
                cur = {"type": tag[2:], "start": s, "end": e}
        else:
            if cur:
                ents.append(cur)
                cur = None
    if cur:
        ents.append(cur)
    for x in ents:
        x["text"] = text[x["start"]:x["end"]]
    return ents


def build_events(ents):
    triggers = [e for e in ents if e["type"] == "Trigger"]
    args = [e for e in ents if e["type"] != "Trigger"]
    if not triggers:
        return []
    if len(triggers) == 1:
        groups = [(triggers[0], args)]
    else:
        buckets = {id(t): [] for t in triggers}
        for a in args:
            t = min(triggers, key=lambda t: abs(t["start"] - a["start"]))
            buckets[id(t)].append(a)
        groups = [(t, buckets[id(t)]) for t in triggers]

    events = []
    for t, t_args in groups:
        events.append({
            "type": "Unknown",
            "trigger": {"text": t["text"], "start": t["start"], "end": t["end"]},
            "args": [{"role": a["type"], "text": a["text"], "start": a["start"], "end": a["end"]}
                     for a in t_args],
        })
    return events


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return round(p, 4), round(r, 4), round(f1, 4)


# ================= 主流程 =================
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)

    model_dir = os.path.join(root_dir, "output", "bert_crf_hard")
    ckpt_path = os.path.join(model_dir, "bert_crf_hard_model.pt")
    eval_path = os.path.join(root_dir, "data", "processed", "dev_extract_clean.jsonl")

    output_dir = os.path.join(root_dir, "output", "bilstm_crf")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "bert_crf_hard.jsonl")

    if not os.path.exists(ckpt_path):
        print(f"❌ 找不到模型: {ckpt_path}，请先运行 bert_crf_hard_train.py")
        return

    print(f"✅ 当前使用的设备: {DEVICE}")
    print("1. 加载模型与字典...")
    tokenizer = BertTokenizerFast.from_pretrained(model_dir)

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    with open(os.path.join(model_dir, "tag_dict.json"), "r", encoding="utf-8") as f:
        tag_dict = json.load(f)
    tag2id = tag_dict["tag2id"]
    id2tag = {int(k): v for k, v in tag_dict["id2tag"].items()}

    bert_path = ckpt.get("bert_path", model_dir)
    bert = BertModel.from_pretrained(bert_path)
    model = BertCrfHardForTokenClassification(bert, id2tag).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.apply_constraints()          # ★ 恢复权重后再钉一次，确保解码受约束
    model.eval()

    start_forbidden, trans_forbidden = model.start_forbidden, model.trans_forbidden
    total_trans = len(id2tag) ** 2
    print(f"   标签数: {len(id2tag)}，模型参数: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    print(f"   硬约束: 禁止 {len(start_forbidden)} 个 I-* 起点；"
          f"禁止 {len(trans_forbidden)}/{total_trans} 条非法转移")

    # 校验非法项确实被钉住
    with torch.no_grad():
        s_ok = (model.crf.start_transitions > NEG / 2).sum().item()
        t_ok = (model.crf.transitions > NEG / 2).sum().item()
    if s_ok == len(id2tag) - len(start_forbidden) and t_ok == total_trans - len(trans_forbidden):
        print("   约束检查: ✅ 已生效")
    else:
        print(f"   约束检查: ⚠ 异常（合法起点 {s_ok}/{len(id2tag) - len(start_forbidden)}，"
              f"合法转移 {t_ok}/{total_trans - len(trans_forbidden)}）")

    print("2. 加载验证集...")
    rows = []
    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    gt_by_sent = defaultdict(list)
    for r in rows:
        if r.get("event"):
            gt_by_sent[r["sent_id"]].append(r["event"])
    n_gt_events = sum(len(v) for v in gt_by_sent.values())
    n_gt_args = sum(len(g.get("args", [])) for v in gt_by_sent.values() for g in v)
    print(f"   句子数: {len(rows)} | 含事件句子: {len(gt_by_sent)} | GT事件: {n_gt_events} | GT论元: {n_gt_args}")

    print("3. 开始推理与计算指标...")
    stats = {"trig": [0, 0, 0], "arg": [0, 0, 0], "event": [0, 0, 0]}
    results = []

    with torch.no_grad():
        for i in range(0, len(rows), BATCH_SIZE):
            batch = rows[i:i + BATCH_SIZE]
            enc = tokenizer(
                [r["text"] for r in batch], max_length=MAX_LEN, truncation=True,
                padding=True, return_offsets_mapping=True, return_tensors="pt",
            )
            paths = model(
                enc["input_ids"].to(DEVICE),
                enc["attention_mask"].to(DEVICE),
            )
            offsets = enc["offset_mapping"].numpy()

            for k, r in enumerate(batch):
                text = r["text"]
                preds = paths[k]
                offs = offsets[k][:len(preds)]

                ents = decode_entities(text, preds, offs, id2tag)
                pred_events = build_events(ents)
                true_events = gt_by_sent.get(r["sent_id"], [])

                # ---- Trigger ----
                gt_trig = Counter((g["trigger"]["text"], g["trigger"]["start"]) for g in true_events)
                pr_trig = Counter((e["trigger"]["text"], e["trigger"]["start"]) for e in pred_events)
                tp = sum((gt_trig & pr_trig).values())
                stats["trig"][0] += tp
                stats["trig"][1] += sum(pr_trig.values()) - tp
                stats["trig"][2] += sum(gt_trig.values()) - tp

                # ---- Argument ----
                gt_args = Counter((a["role"], a["text"], a["start"])
                                  for g in true_events for a in g.get("args", []))
                pr_args = Counter((a["role"], a["text"], a["start"])
                                  for e in pred_events for a in e.get("args", []))
                tp = sum((gt_args & pr_args).values())
                stats["arg"][0] += tp
                stats["arg"][1] += sum(pr_args.values()) - tp
                stats["arg"][2] += sum(gt_args.values()) - tp

                # ---- Event ----
                def key(e):
                    return (e["trigger"]["text"], e["trigger"]["start"],
                            frozenset((a["role"], a["text"]) for a in e.get("args", [])))
                gt_ev = Counter(key(g) for g in true_events)
                pr_ev = Counter(key(e) for e in pred_events)
                tp = sum((gt_ev & pr_ev).values())
                stats["event"][0] += tp
                stats["event"][1] += sum(pr_ev.values()) - tp
                stats["event"][2] += sum(gt_ev.values()) - tp

                results.append({
                    "sent_id": r["sent_id"],
                    "text": text,
                    "pred_event": pred_events[0] if pred_events else None,
                    "pred_events": pred_events,
                    "true_events": true_events,
                })

    t_p, t_r, t_f1 = prf(*stats["trig"])
    a_p, a_r, a_f1 = prf(*stats["arg"])
    e_p, e_r, e_f1 = prf(*stats["event"])

    metrics = {
        "Trigger": {"Precision": t_p, "Recall": t_r, "F1": t_f1},
        "Argument": {"Precision": a_p, "Recall": a_r, "F1": a_f1},
        "Event": {"Precision": e_p, "Recall": e_r, "F1": e_f1},
    }

    with open(output_path, "w", encoding="utf-8") as f_out:
        f_out.write(json.dumps({"global_metrics": metrics}, ensure_ascii=False) + "\n")
        for res in results:
            f_out.write(json.dumps(res, ensure_ascii=False) + "\n")

    print("\n---- 预测示例 ----")
    shown = 0
    for res in results:
        if not res["true_events"] or shown >= 2:
            continue
        print(f"[{shown + 1}] {res['text'][:50]}")
        print(f"    GT  : {res['true_events'][0]['trigger']['text']!r} "
              f"args={[(a['role'], a['text']) for a in res['true_events'][0].get('args', [])][:5]}")
        pe = res["pred_event"]
        if pe:
            print(f"    PRED: {pe['trigger']['text']!r} "
                  f"args={[(a['role'], a['text']) for a in pe.get('args', [])][:5]}")
        else:
            print("    PRED: <无>")
        shown += 1

    print("\n" + "=" * 50)
    print("✅ BERT+CRF+硬约束 验证完成！全局指标如下：")
    print(f" - Trigger F1 : {t_f1:.4f} (P: {t_p:.4f}, R: {t_r:.4f})")
    print(f" - Argument F1: {a_f1:.4f} (P: {a_p:.4f}, R: {a_r:.4f})")
    print(f" - Event F1   : {e_f1:.4f} (P: {e_p:.4f}, R: {e_r:.4f})")
    print(f"\n📁 详细结果已保存至: {output_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
