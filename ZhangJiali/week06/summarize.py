"""汇总各方法结果并生成 Markdown 实验报告。"""

from __future__ import annotations

import json
from pathlib import Path

from common import OUTPUT_DIR, ROOT


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    baseline_path = OUTPUT_DIR / "baseline_results.json"
    roberta_path = OUTPUT_DIR / "roberta_result.json"
    if not baseline_path.exists() or not roberta_path.exists():
        raise FileNotFoundError("请先运行 run_baselines.py 和 train_roberta.py")

    baseline = read_json(baseline_path)
    roberta = read_json(roberta_path)
    rows = []
    for name, result in baseline["results"].items():
        rows.append((name, result))
    rows.append(("轻量中文 RoBERTa 微调", roberta["result"]))
    rows.sort(key=lambda item: item[1]["macro_f1"], reverse=True)

    table = [
        "| 方法 | Accuracy | Macro-F1 | Weighted-F1 | 训练时间/s | 推理时间/s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, result in rows:
        table.append(
            f"| {name} | {result['accuracy']:.4f} | {result['macro_f1']:.4f} | "
            f"{result['weighted_f1']:.4f} | {result['train_seconds']:.2f} | {result['inference_seconds']:.2f} |"
        )

    data = baseline["data"]
    best_name, best = rows[0]
    svm = baseline["results"]["TF-IDF + 线性SVM"]
    mac = roberta["result"]
    report = f"""# 中文短文本分类方法对比实验报告

## 一、实验任务

本实验使用 TNEWS 中文新闻标题数据集完成 15 类文本分类，对多数类基线、TF-IDF 传统模型和预训练语言模型微调进行横向比较。预训练模型更换为 `uer/chinese_roberta_L-2_H-128`。该模型采用 2 层 Transformer 和 128 维隐藏表示，在保留预训练语义表示的同时降低了训练与部署成本。

## 二、数据处理

训练集包含 {data['n_train']} 条样本，原始验证集包含 {data['n_val_raw']} 条样本。实验移除了验证集中与训练集完全相同的 {data['removed_duplicates']} 条文本，最终使用 {data['n_val_clean']} 条干净验证样本。所有方法均使用相同的训练集、验证集和随机种子 42。

文本标签共 15 类，包括财经、体育、科技、教育等新闻主题。传统模型采用字符级 1-2 gram TF-IDF，避免额外引入中文分词误差。RoBERTa 输入最大长度设为 {mac['max_length']}，微调 {mac['epochs']} 个 epoch，学习率为 {mac['learning_rate']}。

## 三、评价指标

- Accuracy：全部样本中预测正确的比例。
- Macro-F1：先计算每个类别的 F1，再对所有类别等权平均，能够反映小类别的识别情况。
- Weighted-F1：按类别样本量对各类别 F1 加权，更接近整体样本表现。
- 训练与推理时间：用于比较模型精度与计算成本。

## 四、实验结果

{chr(10).join(table)}

本次实验中，Macro-F1 最高的方法为 **{best_name}**，其 Accuracy 为 {best['accuracy']:.4f}，Macro-F1 为 {best['macro_f1']:.4f}。轻量中文 RoBERTa 相比 TF-IDF + 线性 SVM 的 Macro-F1 变化为 {(mac['macro_f1'] - svm['macro_f1']):+.4f}。

## 五、结果分析

多数类基线只能识别数据量最大的类别，Macro-F1 很低，说明仅依赖类别先验无法完成多类别新闻分类。TF-IDF + 逻辑回归能够利用字符统计特征，效果明显提升；线性 SVM 更适合高维稀疏文本特征，通常比逻辑回归获得更好的分类边界。

RoBERTa 能利用上下文语义表示，不只依赖局部字符是否出现，因此在标题表述变化、同词异义和类别边界较模糊时具有潜在优势。轻量模型的表达能力低于标准 12 层基座，但训练和部署成本更低。若应用强调快速训练、低资源部署，可优先选择线性 SVM；若需要语义表示并希望控制资源开销，可选择轻量 RoBERTa 微调。

Macro-F1 低于 Accuracy 时，说明类别不均衡仍影响小类别。后续可进一步尝试类别权重、分层采样或分析混淆矩阵，但这些改动应在统一数据切分上完成，避免因评测口径不同产生不可比结果。

## 六、结论

实验表明，传统线性模型是中文短文本分类中成本低且可靠的基线。本次使用的 2 层轻量 RoBERTa 虽然具备上下文表示能力，但受模型容量限制，指标未超过 TF-IDF 线性模型。这说明预训练模型并非越小越合适，压缩计算成本可能损失分类精度。模型选择不应只比较准确率，还应综合考虑 Macro-F1、训练成本、推理速度和部署环境。本实验通过统一数据、去除重复样本和固定随机种子，使不同方法的比较具有可复现性。

## 七、复现命令

```powershell
E:\\Anaconda3\\envs\\BDpy312\\python.exe run_baselines.py
E:\\Anaconda3\\envs\\BDpy312\\python.exe train_roberta.py --epochs {mac['epochs']} --batch-size {mac['batch_size']} --local-files-only
E:\\Anaconda3\\envs\\BDpy312\\python.exe summarize.py
```
"""
    report_path = ROOT / "实验报告.md"
    report_path.write_text(report, encoding="utf-8")
    (OUTPUT_DIR / "comparison_table.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    print(f"实验报告已生成：{report_path}")


if __name__ == "__main__":
    main()

