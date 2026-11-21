# BERT → ModelScope 中文 RoBERTa：领域分类（18 类）

本目录已从「TNEWS 15 类新闻标题分类」改造为
**「中文文本领域分类（国民经济行业 18 大类）」**，预训练模型换成了
ModelScope 的 **RoBERTa预训练模型-中文-base**。

## 1. 预训练模型

| 项 | 值 |
|---|---|
| ModelScope id | `iic/nlp_roberta_backbone_base_std`（等价 `damo/...`） |
| 中文名 | RoBERTa预训练模型-中文-base |
| 本地目录 | `common/pretrain_models/roberta-base-chinese` |
| 结构 | 12 层 / hidden 768 / 12 头 / 词表 21128 |
| 大小 | 约 400 MB（`pytorch_model.bin` + `vocab.txt` + `config.json`） |

下载方式（已下载过可跳过）：

```bash
python -c "from modelscope.hub.snapshot_download import snapshot_download; \
snapshot_download('iic/nlp_roberta_backbone_base_std', local_dir='common/pretrain_models/roberta-base-chinese')"
```

路径由 `common/model_paths.py` 的 `roberta_path()` 向上查找 `pretrain_models/roberta-base-chinese`，
找不到会明确报错，不会静默联网下载。

## 2. 数据与切分

| 项 | 值 |
|---|---|
| 原始数据 | `train_data/zh.test.csv.tgz`（解压得到 `zh.test.csv`，36,138 条） |
| 列 | `text`、`label:LABEL` |
| 类别 | 18 个，基本均衡（每类约 2,000 条） |
| 文本长度 | 平均 42 字 |

原始数据**只有测试集**，所以 `common/domain_data.py` 先按整句去重，再按类别分层切成
**8 : 1 : 1**（固定 `seed=42`），缓存到 `data/domain/`：

| 切分 | 条数 | 用途 |
|---|---|---|
| train | 28,908 | 训练 |
| val | 3,614 | 选最优 checkpoint（按 Macro-F1） |
| test | 3,614 | 最终评测，只在最后用一次 |

想重新切分（例如换 seed / 比例）：

```python
from common.domain_data import build_splits
build_splits(seed=42, rebuild=True)
```

## 3. 训练

```bash
# 默认：cls 池化、3 epoch、batch 32、max_len 64、lr 2e-5（主体）/ 5×（头）
python bert/train_bert.py

# 换池化 / 加轮数
python bert/train_bert.py --pool mean --epochs 5

# 训练技巧（消融项，本任务上不一定涨点）
python bert/train_bert.py --label_smoothing 0.1
python bert/train_bert.py --focal_gamma 2
python bert/train_bert.py --fgm
python bert/train_bert.py --no_class_weight

# 冒烟：只跑 20 步、不落盘（CPU 也能很快跑完）
python bert/train_bert.py --max_steps 20 --max_eval 200 --no_save

# CPU 调试：只用 5,000 条训练样本、1 epoch
python bert/train_bert.py --max_train 5000 --epochs 1
```

输出：

- `outputs/roberta_domain_ckpt.pt`（约 400 MB，含 `labels` / `pool` / `base_model`）
- `outputs/roberta_domain_train_log.json`（每个 epoch 的 train loss / val acc / val Macro-F1）

> CPU 训练提示：本机实测 batch=32、max_len=64 时约 **5.5 秒/步**，
> 28,908 条 ≈ 903 步/epoch，单 epoch 约 80~90 分钟。没有 GPU 时建议
> 用 `--max_steps` 或 `--epochs 1` 做流程验证；正式训练建议 GPU。

## 4. 评测

```bash
# 默认在 test 切分上评全部 3,614 条
python bert/predict_bert.py

# 评验证集 / 抽样
python bert/predict_bert.py --split val
python bert/predict_bert.py --n 500
```

输出 `outputs/roberta_domain_pred.json`，包含：

- `report`：Accuracy、Macro-F1、Weighted-F1、逐类 Recall、18×18 混淆矩阵、ECE
- `preds`：逐条 `text / true / pred / true_name / pred_name`，可直接做错误分析
- `meta`：切分、样本量、标签列表

## 5. 与项目其他方法的关系

`rule/`、`tfidf/`、`llm_*` 仍然跑在原来的 TNEWS 15 类数据上，`eval/eval_all.py`
汇总的也是那套任务；**RoBERTa 领域分类是独立任务**，指标不与它们横向比较。
若要把领域分类接入服务，需要另外定义 18 类的队列/相关性口径，不能直接复用
`common/labels.py` 里的 15 类 → 4 组映射。
