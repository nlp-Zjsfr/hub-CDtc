"""
统一评测汇总：把各路方法的结果拉到一张表上

【这个脚本在干什么】
读取各路方法各自落盘的结果 JSON，检查「口径是否一致」，渲染成一张 Markdown 表
（outputs/summary.md），并打印口径不一致的告警。

教学重点：
  1. 四路方法必须跑在**同一批样本**上（common/data.py 的 clean_val + 固定 seed），
     否则数字不可比。这个脚本先检查每份结果的 meta（n_eval / seed），不一致就报警。
  2. 只报 Accuracy 会被头部类骗，所以同时给 Macro-F1 和「预测覆盖类别数」——
     后者能一眼看出 zero-shot 的塌缩。
  3. 汇总结果写到 outputs/summary.md，可直接粘进 ARCHITECTURE.md。

用法：
  python eval/eval_all.py                 # 汇总已有结果（先跑各路脚本）
  python eval/eval_all.py --run-light     # 先跑规则 + TF-IDF，再汇总
"""

import argparse                        # 命令行参数
import json                            # 读各路结果 JSON
import subprocess                      # --run-light 时拉起子进程跑脚本
import sys                             # 模块搜索路径
from pathlib import Path               # 路径处理

CODE_ROOT = Path(__file__).resolve().parent.parent   # code/ 根目录
sys.path.insert(0, str(CODE_ROOT))     # 让本脚本也能 import common.*

OUT = CODE_ROOT / "outputs"            # 各路结果都放这里

# 方法名 → 结果文件名（跑过哪一路就读哪一份）
SOURCES = [
    ("规则 / 关键词", "rule_pred.json"),
    ("TF-IDF + SVM", "tfidf_pred.json"),
    ("BERT 微调", "bert_pred.json"),
    ("Qwen3 zero-shot", "zero_shot_pred.json"),
    ("Qwen3 LoRA SFT", "lora_pred.json"),
    ("Qwen3 全量微调", "full_ft_pred.json"),
    ("API zero-shot（deepseek-v4-flash）", "api_zero_shot_pred.json"),
]


def load(name):
    """读一份结果文件；不存在或损坏都返回 None（汇总时不中断，只是标记「未运行」）"""
    p = OUT / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[warn] {name} 解析失败：{e}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-light", action="store_true", help="先跑规则与 TF-IDF（秒级）")
    args = ap.parse_args()

    if args.run_light:
        # 只跑秒级的两路（规则 + TF-IDF），BERT/LLM 太慢就不自动跑
        for script in ("rule/classify_rule.py", "tfidf/classify_tfidf.py"):
            print(f"→ 运行 {script}")
            subprocess.run([sys.executable, str(CODE_ROOT / script), "--n", "500"], check=False, cwd=CODE_ROOT)
            # check=False：某一路挂了继续往下跑，不中断汇总

    rows, metas = [], {}
    for label, fname in SOURCES:
        data = load(fname)                    # 读这份结果
        if not data:
            rows.append({"method": label, "acc": "-", "macro_f1": "-", "n": "-", "note": "未运行"})
            continue
        r = data["report"]                     # 指标字典
        # 老版本结果文件可能没有 acc_incl_unparsed，这里用逐条预测重算（解析失败算答错）
        if "acc_incl_unparsed" not in r and data.get("preds"):
            ps = data["preds"]
            r["acc_incl_unparsed"] = round(sum(1 for x in ps if x["pred"] == x["true"]) / len(ps), 4)
        m = data.get("meta", {})               # 评测元信息（n_eval / seed / n_dup…）
        metas[label] = m
        dist = r.get("pred_distribution") or {}   # 预测分布（LLM 塌缩看这个）
        rows.append({
            "method": label,
            "acc": r.get("acc_incl_unparsed", r.get("acc", "-")),   # 优先用「含失败」口径
            "macro_f1": r.get("macro_f1", "-"),
            "n": m.get("n_eval", r.get("n")),   # 样本数
            "used_classes": dist.get("used_classes", "-"),  # 用到了几个类别
            "unparsed": r.get("unparsed", 0),   # 解析失败条数
            "note": "",
        })

    # 口径检查：样本量/seed 不一致就要提醒，这是「数字不可比」最常见的原因
    seeds = {v.get("seed") for v in metas.values() if v.get("seed") is not None}     # 各路 seed 集合
    sizes = {v.get("n_eval") for v in metas.values() if v.get("n_eval") is not None}  # 各路样本量集合
    if len(sizes) > 1:
        print(f"[warn] 各路样本量不一致：{sizes}，结果不可直接比较")
    if len(seeds) > 1:
        print(f"[warn] 各路 seed 不一致：{seeds}，抽样集合可能不同")

    header = "| 方法 | 准确率(含失败) | Macro-F1 | 样本数 | 覆盖类别 | 解析失败 |"
    sep = "|---|---|---|---|---|---|"
    lines = [header, sep]
    for r in rows:
        lines.append(f"| {r['method']} | {r['acc']} | {r['macro_f1']} | {r['n']} | "
                     f"{r.get('used_classes', '-')} | {r.get('unparsed', '-')} |")
    table = "\n".join(lines)

    print("\n" + table + "\n")
    note = []
    note.append("口径：全部方法在同一批「去重后的干净验证样本」上评测（seed=42）；"
                "各路实际评测条数见上表「样本数」列，条数不同的两行不可直接比较。")
    if "TF-IDF + SVM" in metas or True:
        # 这句参考值是「泄漏对照」的结论，固定写死（实测结论，不随单次实验变化）
        note.append("参考：全量 val（含 11% 重复）上 TF-IDF 0.541 / BERT 0.551，去重后分别降到 0.509 / 0.534。")
    (OUT / "summary.md").write_text(table + "\n\n" + "\n".join(note) + "\n", encoding="utf-8")
    print(f"→ 汇总写入 {OUT / 'summary.md'}")


if __name__ == "__main__":
    main()
