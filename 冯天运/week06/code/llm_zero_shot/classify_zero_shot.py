"""
方法四：大模型零样本（L3，不训练）

【这个脚本在干什么】
用 Qwen3-0.6B 直接按 prompt 分类，**不训练**任何东西，
用它来展示「零样本为什么在这个任务上塌缩」。

教学重点：
  1. 零样本 = 把分类当成「生成」来做：写一句 prompt，让模型直接吐类别名。
     不用数据、不用训练，冷启动最快——这是它的唯一优势。
  2. 但 15 类中文新闻标题上它**基本不可用**：实测 Macro-F1 0.098，
     相对随机基线（1/15=0.067）只高一点。原因是**预测塌缩**：
     15 类里只输出过 5 类，「文化」一个类就占了七成。
  3. Qwen3 必须传 enable_thinking=False。不传的话模型会先吐一段
     <think>...</think> 推理链，把「只输出类别名」的格式彻底打乱。
     这是 Qwen2 → Qwen3 迁移时最容易踩的坑。

用法：
  python llm_zero_shot/classify_zero_shot.py              # 默认 200 条
  python llm_zero_shot/classify_zero_shot.py --n 500
"""

import argparse                                # 命令行参数
import json                                   # 读写 JSON
import re                                     # 正则（保留 import，解析里可能用到）
import sys                                    # 模块搜索路径
import time                                   # 计时
from pathlib import Path                     # 路径处理

import torch                                  # PyTorch（本地加载 LLM）

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# ★把 code/ 加进 sys.path

from common.data import load_clean_val                     # 统一评测集
from common.labels import LABELS                          # 15 类
from common.metrics import classification_report, pred_distribution, to_markdown_table
from common.model_paths import qwen3_path                 # 本地 Qwen3 目录

CODE_ROOT = Path(__file__).resolve().parent.parent

# Prompt 三件套：角色 + 类别定义 + 输出约束
# 类别名不在 prompt 里列全，模型就会自己发明标签（实测会答「新闻」「其它」）
SYSTEM = ("你是一个新闻标题分类助手。请将给定的新闻标题分类到以下类别之一，"
          "只输出类别名称，不要输出任何其他内容。\n可选类别：" + "、".join(LABELS))
# ↑ 15 个类别名必须**全部列出**，否则模型会自由发挥


def parse_label(text: str) -> int:
    """
    从生成文本里抽出类别名；抽不到返回 -1（记为解析失败）。

    为什么用「包含」而不是「等于」：
      模型经常多吐标点或一句解释（"类别：财经。"），严格相等会误判为失败；
      用包含 + 取最长匹配，能容忍这些噪声，同时保留失败样本可统计。
    """
    text = text.strip()
    hits = [name for name in LABELS if name in text]   # 哪些类别名出现在输出里
    if not hits:
        return -1                                     # 一个都没有 → 解析失败
    return LABELS.index(max(hits, key=len))           # 取最长的那个（避免「文」被「文化」抢走）


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)            # 默认只跑 200 条（LLM 很慢）
    ap.add_argument("--max_new_tokens", type=int, default=8)  # 生成上限（只要一个类别名，8 个够）
    ap.add_argument("--out", default="outputs/zero_shot_pred.json")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(str(qwen3_path()))
    model = AutoModelForCausalLM.from_pretrained(
        str(qwen3_path()), dtype=torch.float16 if device == "cuda" else torch.float32
        # ↑ GPU 上用 fp16 省显存；CPU 上 fp16 太慢，用 fp32
    ).to(device).eval()

    samples, meta = load_clean_val(args.n)         # 与其它方法同一批样本
    preds, raws, t0 = [], [], time.time()
    for i, d in enumerate(samples):
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": d["sentence"]}]
        # enable_thinking=False 必须在模板阶段传：Qwen3 会在 assistant 前插入空 think 块
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        enc = tok(text, return_tensors="pt").to(device)
        with torch.no_grad():                       # 不建计算图
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                 do_sample=False,        # ★贪心解码：结果可复现，不要采样
                                 pad_token_id=tok.eos_token_id)
        # 只解码新生成的部分（切掉输入 prompt 的长度）
        raw = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        raws.append(raw)                           # 原始文本也留着（查失败样本用）
        preds.append(parse_label(raw))             # 解析成类别 id（-1 = 失败）
        if i % 50 == 0:
            print(f"  {i}/{len(samples)} {time.time()-t0:.0f}s", flush=True)   # 进度

    y_true = [d["label"] for d in samples]
    report = classification_report(y_true, preds)  # 统一口径（解析失败会算进 acc_incl_unparsed）
    report["method"] = "qwen3_zero_shot"
    report["sec_per_item"] = round((time.time() - t0) / len(samples), 2)   # 单条耗时（很贵）
    report["pred_distribution"] = pred_distribution(preds)   # 塌缩的证据

    out = CODE_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "report": report,
                               "preds": [{"text": d["sentence"], "true": d["label"], "pred": p, "raw": r}
                                         for d, p, r in zip(samples, preds, raws)]},
                              ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"评测 {meta['n_eval']} 条 | {report['sec_per_item']}s/条")
    print(to_markdown_table({"Qwen3 zero-shot": report}))
    print(f"预测覆盖类别数：{report['pred_distribution']['used_classes']}/{len(LABELS)}")  # 塌缩程度
    print(f"→ 结果写入 {out}")


if __name__ == "__main__":
    main()
