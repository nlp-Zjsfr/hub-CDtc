# -*- coding: utf-8 -*-
"""
BiLSTM+CRF 验证脚本 —— 与 bert_ori_valid.py 完全同一评估口径

修复的旧版问题（旧版 0.3064 的成因）：
  1. 【数据源】旧版用 dev_detect_clean.jsonl(4821 条，混了 1173 条标题记录) 作为输入文本，
     却用 dev_extract_clean.jsonl(920 条) 当标准答案 —— sent_id 对不上，
     绝大多数句子 true_events 为空，预测全被记成 FP。
  2. 【长度】旧版 MAX_LEN=256 截断，而 bilstm_train.py 是按 MAX_LEN=512 训练的。
  3. 【统计】旧版一句只保留最后一个 trigger、arg 的 FN 只在“没预测出事件”时才累加、
     trigger 的 FP 按句计 —— 与 BERT 版口径不一致，无法公平对比。
"""

import os
import json
from collections import defaultdict, Counter

import torch
import torch.nn as nn
from torchcrf import CRF

EMBEDDING_DIM = 128
HIDDEN_DIM = 256
MAX_LEN = 512                      # 与 bilstm_train.py 保持一致
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag2id):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM, padding_idx=0)
        self.lstm = nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM // 2, num_layers=1,
                            bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(HIDDEN_DIM, len(tag2id))
        self.crf = CRF(len(tag2id), batch_first=True)

    def forward(self, x, tags=None, mask=None):
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        emissions = self.hidden2tag(lstm_out)
        if tags is not None:
            return -self.crf(emissions, tags, mask=mask, reduction='mean')
        return self.crf.decode(emissions, mask=mask)


# ================= 解码 / 拼事件 / 指标（与 bert_ori_valid.py 相同） =================
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


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)

    model_dir = os.path.join(root_dir, "output", "bilstm_crf")
    model_path = os.path.join(model_dir, "bilstm_crf_model.pt")
    # 文本与标注同源
    eval_path = os.path.join(root_dir, "data", "processed", "dev_extract_clean.jsonl")
    output_path = os.path.join(model_dir, "bilstm.jsonl")

    if not os.path.exists(model_path):
        print(f"❌ 找不到模型: {model_path}，请先运行 bilstm_train.py")
        return

    print(f"✅ 当前使用的设备: {DEVICE}")
    print("1. 加载模型与字典...")
    ckpt = torch.load(model_path, map_location=DEVICE)
    char2id = ckpt["char2id"]
    tag2id = ckpt["tag2id"]
    id2tag = ckpt["id2tag"]

    model = BiLSTM_CRF(len(char2id), tag2id).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"   标签数: {len(tag2id)}，词表大小: {len(char2id)}")

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
    n_long = sum(1 for r in rows if len(r["text"]) > MAX_LEN)
    print(f"   句子数: {len(rows)} | 含事件句子: {len(gt_by_sent)} | GT事件: {n_gt_events} | GT论元: {n_gt_args}")
    if n_long:
        print(f"   ⚠ 其中 {n_long} 条文本长度 > {MAX_LEN}，按模型训练上限截断")

    print("3. 开始推理与计算指标...")
    stats = {"trig": [0, 0, 0], "arg": [0, 0, 0], "event": [0, 0, 0]}
    results = []

    with torch.no_grad():
        for r in rows:
            text = r["text"]
            if len(text) > MAX_LEN:
                text = text[:MAX_LEN]

            x_ids = [char2id.get(c, char2id["<UNK>"]) for c in text]
            x_tensor = torch.tensor([x_ids]).to(DEVICE)
            mask = torch.ones((1, len(x_ids)), dtype=torch.bool).to(DEVICE)

            pred_tags = model(x_tensor, mask=mask)[0]
            offsets = [(i, i + 1) for i in range(len(x_ids))]

            ents = decode_entities(text, pred_tags, offsets, id2tag)
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
    print("✅ BiLSTM+CRF 验证完成！全局指标如下：")
    print(f" - Trigger F1 : {t_f1:.4f} (P: {t_p:.4f}, R: {t_r:.4f})")
    print(f" - Argument F1: {a_f1:.4f} (P: {a_p:.4f}, R: {a_r:.4f})")
    print(f" - Event F1   : {e_f1:.4f} (P: {e_p:.4f}, R: {e_r:.4f})")
    print(f"\n📁 详细结果已保存至: {output_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
