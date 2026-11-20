"""
方法三（推理）：加载 BERT checkpoint，在干净验证集上评测
================================================================
【这个脚本在干什么】
  训练阶段产出的是"权重文件"（.pt），这个脚本负责：
    权重 → 加载模型 → 拿一批验证样本 → 逐条预测 → 算指标 → 存结果 json

【三个必须记住的点】
  1. 评测集必须是**去重后的同一批样本**（load_clean_val），否则和 TF-IDF / LLM 的数字不可比；
  2. 推理要**批处理**（batch=32），单条调用比批量慢 3 倍多（实测 394ms → 121ms/条）；
  3. 保存**逐条预测**，后面才能和规则/TF-IDF/LLM 做三方对照与错误分析。

用法：
  python bert/predict_bert.py                       # 默认 outputs/bert_ckpt.pt
  python bert/predict_bert.py --ckpt path/to/ckpt.pt --n 500
"""

# argparse：命令行参数
import argparse
# json：读写预测结果
import json
# statistics：统计工具（预留）
import statistics
# sys：调整 import 路径
import sys
# time：测推理延迟
import time
# pathlib：拼路径
from pathlib import Path

# torch：深度学习框架（张量运算、no_grad 上下文）
import torch

# 让本文件能 import 到同级目录 code/ 下的 common、bert 包
# （本文件在 code/bert/ 里，父目录才是 code/）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.data import load_clean_val        # 取"去重后的干净验证集"（同口径的关键）
from common.labels import LABELS              # 15 个类别的名字与顺序
from common.metrics import (classification_report, expected_calibration_error,
                            to_markdown_table)  # 统一口径的指标计算（七路方法共用）
from common.model_paths import bert_path      # 定位预训练模型权重（受 WEEK6_BERT_DIR 控制）
from train_bert import BertClassifier, TnewsDataset   # 复用训练时的模型定义（保证结构一致）

CODE_ROOT = Path(__file__).resolve().parent.parent   # = code/ 目录


@torch.no_grad()          # ★推理模式：不建计算图、不存梯度 → 省内存、更快
def predict(model, tok, texts, device, max_length=64, batch_size=32):
    """
    批量预测。返回 (preds, probs, 每条样本耗时毫秒)
    为什么必须分批？因为一次一条要重复 8,901 次前向，每次都要把模型"叫醒"，
    批处理能把权重读进 cache 的次数降下来，实测快 3 倍多。
    """
    preds, probs = [], []                       # preds=预测类别 id；probs=每条样本的 15 类概率
    t0 = time.time()                            # 计时起点
    for i in range(0, len(texts), batch_size):  # 按 batch_size 切块遍历
        enc = tok(texts[i:i + batch_size], truncation=True, max_length=max_length,
                  padding="max_length", return_tensors="pt")
        # truncation=True：超长截断；padding="max_length"：短句补齐到定长（否则 batch 内形状不一）
        # return_tensors="pt"：直接给 PyTorch 张量，不用自己转
        logits = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        # attention_mask：告诉模型哪些位置是 [PAD]，池化时要用
        p = torch.softmax(logits.float(), dim=-1)   # logits → 概率（转 float32 避免精度问题）
        preds.extend(p.argmax(-1).cpu().tolist())    # 概率最大的类别 = 预测标签
        probs.extend(p.cpu().tolist())               # 保留完整概率分布（算 ECE 要用）
    ms = (time.time() - t0) * 1000 / max(1, len(texts))   # 平均每条多少毫秒
    return preds, probs, round(ms, 1)


def main():
    # === 作用：加载训练好的权重 → 在"去重后的干净验证集"上评测 → 落盘结果 JSON ===
    # 这份 JSON 是后面所有横向对比（汇总分组指标 / 混淆矩阵 / 报告表格）的唯一数据来源，
    # 所以评测口径必须与其它方法完全一致：同一份 load_clean_val、同一套 metrics。
    ap = argparse.ArgumentParser()          # 命令行参数解析器
    ap.add_argument("--ckpt", default=str(CODE_ROOT / "outputs" / "bert_ckpt.pt"))  # 权重文件路径
    ap.add_argument("--n", type=int, default=0, help="抽样条数，0=全部干净样本")   # 0 表示全量评测
    ap.add_argument("--out", default="outputs/bert_pred.json")                     # 结果 JSON 输出路径
    args = ap.parse_args()                    # 把命令行参数读进 args

    device = "cuda" if torch.cuda.is_available() else "cpu"   # 有 GPU 用 GPU，否则 CPU
    # 读权重：map_location="cpu" 先在 CPU 上反序列化（避免 GPU 显存不够时直接崩）
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)   # torch 2.6 默认 weights_only=True，会拒载字典
    from transformers import AutoTokenizer
    # ★关键：分词器必须和"训练时用的那个预训练模型"完全一致，
    #   否则词表对不上，出来的预测全是垃圾。bert_path() 读的就是环境变量 WEEK6_BERT_DIR。
    tok = AutoTokenizer.from_pretrained(str(bert_path()))
    # 重建模型骨架：结构（num_labels=15、pool 方式）必须和训练时一致，才能 load_state_dict
    model = BertClassifier(bert_path(), num_labels=len(LABELS), pool=ckpt.get("pool", "cls"))
    model.load_state_dict(ckpt["state_dict"])   # 灌入训练好的参数（严格匹配 shape，不匹配会直接报错）
    model.to(device).eval()                     # .to(device) 搬到设备；.eval() 关闭 dropout/BN 随机性

    samples, meta = load_clean_val(args.n or None)   # 取干净评测集（已去重，避免"训练集泄漏"虚高）
    texts = [d["sentence"] for d in samples]          # 只要文本
    preds, probs, ms = predict(model, tok, texts, device)  # 批量推理：返回预测标签、概率、单条耗时

    y_true = [d["label"] for d in samples]        # 真值标签（整数 id）
    report = classification_report(y_true, preds)  # 统一指标口径：acc / Macro-F1 / 每类 P-R-F1
    report.update({
        "method": "bert_finetune",                          # 方法名，写进 JSON 供后续脚本识别
        "pool": ckpt.get("pool"),                           # 池化方式（cls / mean），用于核对两次实验是否同设置
        "ckpt_val_f1": ckpt.get("val_macro_f1"),            # 训练过程中记录的自测 F1（可与本次复测对照）
        "ms_per_sample": ms,                                # 单条推理耗时(ms)：横向对比里的速度维度
        "ece": expected_calibration_error(y_true, probs),  # 期望校准误差：置信度是否等于真实正确率
    })

    out = CODE_ROOT / args.out        # 相对 code/ 目录定位输出文件
    out.parent.mkdir(parents=True, exist_ok=True)   # 确保父目录存在（outputs/ 可能还没建）
    # 三段式落盘：meta(评测集元信息) + report(指标) + preds(逐条明细，供画混淆矩阵/查错分样本)
    out.write_text(json.dumps({"meta": meta, "report": report,
                               "preds": [{"text": t, "true": y, "pred": p}
                                         for t, y, p in zip(texts, y_true, preds)]},
                              ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"评测 {meta['n_eval']} 条（干净集）| {ms} ms/条 | ECE={report['ece']}")  # 一行摘要
    print(to_markdown_table({"BERT 微调": report}))   # 指标表格（直接可贴进报告）
    print(f"→ 结果写入 {out}")


if __name__ == "__main__":
    main()
