"""
把换预训练模型对照实验的结果回填进《对比实验报告.md》的占位符。

【这个脚本在干什么】
读 outputs/model_swap_experiment.json，把两个模型的 acc / Macro-F1 / ECE / 耗时
填进报告里的 {{BERT_*}}、{{ROB_*}}、{{DELTA_*}}、{{VERDICT}} 占位符，
避免手工抄数字抄错。

用法（在作业目录下执行）：
  python 回填对比结果.py

数据来源：code/outputs/model_swap_experiment.json
被更新：  对比实验报告.md 里的 {{BERT_*}} / {{ROB_*}} / {{DELTA_*}} / {{VERDICT}}
"""
import json                       # 读实验结果
from pathlib import Path           # 路径处理

HERE = Path(__file__).resolve().parent          # 作业目录
SRC = HERE / "outputs" / "model_swap_experiment.json"   # 实验结果（由 换模型对照实验.py 生成）
REPORT = HERE / "对比实验报告.md"                       # 要更新的报告

KEYS = ["acc", "macro_f1", "ece", "ms_per_sample"]    # 需要从结果里取的指标
TAGS = {"bert-base-chinese": "BERT", "chinese-roberta-wwm-ext": "ROB"}   # 占位符前缀


def fmt(v, nd=4):
    """格式化一个指标：None → '-';浮点保留 nd 位小数"""
    return "-" if v is None else (round(v, nd) if isinstance(v, float) else v)


def main():
    if not SRC.exists():
        print("no result file:", SRC)      # 没结果就明说，不静默退出
        return
    d = json.loads(SRC.read_text(encoding="utf-8"))
    by = {r["model"]: r for r in d.get("results", [])}   # 按模型名索引
    if len(by) < 2:
        print("result file incomplete:", list(by))        # 两个模型没跑齐就别填
        return

    def ev(model):
        """取某个模型的评测指标 + 训练耗时（分钟）"""
        r = by[model]
        e = r.get("eval") or {}                 # 评测指标字典
        sec = (r.get("train") or {}).get("seconds") or 0   # 训练耗时（秒）
        return {
            "acc": e.get("acc"),
            "f1": e.get("macro_f1"),
            "ece": e.get("ece"),
            "ms": e.get("ms_per_sample"),
            "min": (round(sec / 60.0, 1) if sec else None),   # 秒 → 分钟
        }

    a = ev("bert-base-chinese")               # 对照组
    b = ev("chinese-roberta-wwm-ext")         # 实验组

    def diff(x, y):
        """差值 = 实验组 - 对照组；任一缺失则返回 None"""
        if x is None or y is None:
            return None
        return round(y - x, 4)

    d_acc = diff(a["acc"], b["acc"])
    d_f1 = diff(a["f1"], b["f1"])

    # verdict: roberta - bert-base（差值分档，给出定性结论）
    if d_f1 is None:
        verdict = "-"
        d_abs = "-"
    else:
        d_abs = round(abs(d_f1), 4)
        if d_f1 > 0.01:
            verdict = "roberta 明显更优"        # 差 > 1 个点
        elif d_f1 > 0.002:
            verdict = "roberta 略优"            # 差 > 0.2 个点
        elif d_f1 < -0.01:
            verdict = "bert-base 明显更优"
        elif d_f1 < -0.002:
            verdict = "bert-base 略优"
        else:
            verdict = "两者基本持平"            # 差在 0.2 个点以内 → 视为噪声

    # 占位符 → 实际数值的映射表
    rep = {
        "{{BERT_ACC}}": fmt(a["acc"]), "{{BERT_F1}}": fmt(a["f1"]),
        "{{BERT_ECE}}": fmt(a["ece"]), "{{BERT_MS}}": fmt(a["ms"], 1),
        "{{BERT_MIN}}": (str(a["min"]) + " 分钟" if a["min"] else "-"),
        "{{ROB_ACC}}": fmt(b["acc"]), "{{ROB_F1}}": fmt(b["f1"]),
        "{{ROB_ECE}}": fmt(b["ece"]), "{{ROB_MS}}": fmt(b["ms"], 1),
        "{{ROB_MIN}}": (str(b["min"]) + " 分钟" if b["min"] else "-"),
        "{{DELTA_ACC}}": fmt(d_acc), "{{DELTA_F1}}": fmt(d_f1),
        "{{DELTA_ABS}}": fmt(d_abs), "{{VERDICT}}": verdict,
    }

    txt = REPORT.read_text(encoding="utf-8")     # 读报告全文
    left = 0
    for k, v in rep.items():
        if k in txt:                            # 只有占位符还在才替换（避免重复运行时二次替换错位）
            txt = txt.replace(k, str(v))
            left += 1
    REPORT.write_text(txt, encoding="utf-8")    # 写回（全文替换）
    print("filled %d placeholders -> %s" % (left, REPORT.name))
    print("bert  :", a)
    print("roberta:", b)
    print("delta_f1:", d_f1, "verdict:", verdict)


if __name__ == "__main__":
    main()
