# 中文短文本分类方法对比实验

## 1. 实验目的

本实验在 TNEWS 中文新闻标题数据集上比较四种分类方法：多数类基线、TF-IDF + 逻辑回归、TF-IDF + 线性 SVM、轻量中文 RoBERTa 微调。预训练模型选用 `uer/chinese_roberta_L-2_H-128`，与课程示例中的 BERT 基座不同。

实验重点是比较传统机器学习与预训练语言模型在相同数据、相同验证集和相同指标下的表现。验证集会先移除与训练集完全重复的文本，避免数据泄漏造成指标虚高。

## 2. 目录结构

```text
submission_macbert_comparison/
├── data/                    # 训练集与验证集
├── outputs/                 # 实验指标、预测结果和自动生成的报告
├── pretrained_model/       # 轻量 RoBERTa 原始预训练权重
├── common.py                # 数据读取、去重、指标计算与随机种子
├── run_baselines.py         # 三种非预训练基线
├── train_roberta.py         # 轻量中文 RoBERTa 微调与评测
├── summarize.py             # 汇总结果并生成实验报告
├── requirements.txt         # Python 依赖
└── 实验报告.md              # 实验过程、结果与分析
```

## 3. 运行环境

- Python 3.12
- PyTorch 2.14
- Transformers 5.22
- scikit-learn 1.5
- CUDA GPU（推荐，CPU 也可运行）

建议使用课程虚拟环境：

```powershell
E:\Anaconda3\envs\BDpy312\python.exe run_baselines.py
E:\Anaconda3\envs\BDpy312\python.exe train_roberta.py
E:\Anaconda3\envs\BDpy312\python.exe summarize.py
```

提交包已包含本实验使用的轻量预训练权重，可以离线复现。`--model-name` 参数也可指向其他兼容的本地模型目录。

## 4. 复现实验

在本目录依次执行：

```powershell
python run_baselines.py
python train_roberta.py --epochs 5 --batch-size 128 --learning-rate 1e-4
python summarize.py
```

轻量模型使用比标准 BERT 更高的学习率，以减少欠拟合。快速检查代码流程：

```powershell
python train_roberta.py --epochs 1 --max-train 512 --max-eval 200 --no-save-model
```

所有方法使用同一份去重验证集。默认输出 Accuracy、Macro-F1、Weighted-F1、训练耗时和推理耗时，详细结果保存在 `outputs/`。

