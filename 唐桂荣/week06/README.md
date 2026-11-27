# 第六周作业 商品名称分类

根据中文商品名 `product_name` 预测 `category`。作业比较字符 TF-IDF + LinearSVC、本地中文 BERT 和本地中文 RoBERTa，以 Macro-F1 为主指标，同时记录准确率、各类别指标、混淆矩阵和训练耗时。

## 最终结果

两套微调模型各完成 9 组实验，分别按验证集 Macro-F1 选出最佳配置。下表使用同一批 3,906 条去重测试样本。

| 方法 | 测试配置 | Macro-F1 | 准确率 |
|---|---|---:|---:|
| TF-IDF + LinearSVC | 字符 1–2 gram，`C=1.0` | **0.9832** | **98.80%** |
| BERT | exp2 · Mean 池化 | 0.9752 | 98.05% |
| RoBERTa | exp4 · CLS + Mean 池化 | 0.9705 | 97.72% |

TF-IDF 在本次测试中得分最高。商品名经常直接含有类别线索，例如「牙膏」和「奶茶」。两个 tokenizer 在长度 32 时均未截断清洗后的样本。将最大长度从 64 改为 32 后，本机两套模型的三轮训练耗时均减少约 43%。 [实验总结](outputs/summary.md) · [18 组验证结果](outputs/eval/validation.json) · [测试指标与混淆矩阵](outputs/eval/test_comparison.json)

| 图表 | 内容 |
|---|---|
| [类别分布](outputs/analysis/category_distribution.png) | 原始训练集和测试集的 10 类占比 |
| [商品名长度](outputs/analysis/product_name_lengths.png) | 字符长度和两个 tokenizer 的 token 长度 |
| [消融结果](outputs/figures/ablation_validation.png) | 两套模型的验证集 Macro-F1 及相对各自 exp1 的变化 |
| [最终测试](outputs/figures/final_test_metrics.png) | 三种方法的 Macro-F1 与准确率 |
| [训练耗时](outputs/figures/training_time.png) | exp1、长度 32 和 FGM 的三轮训练耗时 |
| [果汁误判](outputs/figures/juice_confusions.png) | 冷藏与非冷藏果汁的两个误判方向，附条数及类别内比例 |

## 数据与实验设置

原始数据包含 10 类，训练集 15,997 条，测试集 4,003 条。去重键使用 Unicode NFKC、大小写折叠与空白移除，模型接收原始商品名。同名同类保留首条，同名异类整组排除。测试集中有 86 条记录的商品名与原始训练集重叠，清洗时一并排除。

清洗后的训练数据以 `seed=42` 按类别分层划出 10% 作为验证集。最终切分为 **14,187 条训练、1,577 条验证、3,906 条测试**。TF-IDF 的词表与 IDF 只用训练子集拟合。 [数据读取与切分](common/data.py) · [数据审计](outputs/analysis/data_audit.json)

两套微调模型的 exp1 使用 CLS 池化、最大长度 64、编码器学习率 `2e-5`、分类头学习率 `1e-4`、10% warmup 和类别加权交叉熵。每组训练 3 轮，批大小为 64，模型随机种子为 42。类别加权是全部 18 组微调的共同设置，权重由训练子集计算。exp2–exp9 各自覆盖 exp1 的一项设置。 [九组实验配置](bert/experiments.py)

## 运行方式

在 `week6作业/` 目录下执行命令。本机使用 Conda 的 `pytorch` 环境，验证过的版本为 Python 3.11.15、PyTorch 2.13.0、Transformers 5.18.0、scikit-learn 1.9.1 和 Matplotlib 3.11.2。模型权重和 tokenizer 从 `model/bert-base-chinese/` 与 `model/roberta_base/` 离线读取。

### 使用已有结果

```bash
conda activate pytorch
python -m eval.plot_results
```

绘图脚本从 `outputs/eval/` 读取已保存的指标，并将 PNG 写到 `outputs/figures/`。如需指定中文字体，可传入 `--font /path/to/chinese-font.ttf`。

### 从头运行实验

在尚未生成训练结果的副本中运行以下命令。保留作业目录中的 `outputs/summary.md`，最终评测会更新其中的结果章节。本次运行用 Apple Silicon 的 MPS 和 FP32。若使用 CPU 或 CUDA，两条 exp1 命令和批量命令的 `--device` 应使用同一设备。

```bash
conda activate pytorch
python -m tfidf.classify_tfidf
python -m bert.train_bert --model bert-base-chinese --experiment exp1 --device mps
python -m bert.train_bert --model roberta_base --experiment exp1 --device mps
python -m bert.run_experiments --device mps --finalize
python -m eval.plot_results
```

批量脚本依次运行 exp6、exp2–exp5、exp7–exp9。再次执行会跳过已经完整保存的实验。每组的验证指标、逐条预测和最佳权重保存在 `outputs/bert/<模型>/<实验>/`。`--finalize` 校验两套模型的全部验证结果，锁定最佳配置，再运行最终测试。 [批量训练入口](bert/run_experiments.py) · [最终评测入口](eval/evaluate_experiments.py)
