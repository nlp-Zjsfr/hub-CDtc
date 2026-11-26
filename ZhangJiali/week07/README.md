# 金融公告事件抽取方法对比

## 1. 实验目的

本实验在 DuEE-Fin 派生的句子级金融公告数据上比较三种事件抽取方法：规则基线、字符级 BiLSTM-CRF，以及 BERT+Linear 联合序列标注。预训练编码器采用本地 `bert-base-chinese`。

任务定义为：给定一句公告文本，抽取恰好一个结构化事件，包括事件类型、触发词原文片段，以及该类型下各角色对应的论元片段。三种方法使用同一训练集、同一验证集和同一套三级评测指标。

## 2. 目录结构

```text
week07/
├── data/                    # 句级抽取数据与事件 schema
├── pretrained_model/        # bert-base-chinese 本地权重
├── outputs/                 # 指标、预测、训练日志与对比图
├── common.py                # 数据、标签体系、模型封装与统一评测
├── run_rule.py              # 规则基线
├── train_bilstm.py          # 字符级 BiLSTM-CRF
├── train_bert.py            # BERT+Linear 联合抽取
├── summarize.py             # 汇总对比表与柱状图
├── requirements.txt
├── README.md
└── 实验报告.md
```

## 3. 运行环境

- Python 3.12
- PyTorch 2.14（CUDA 12.6）
- Transformers 5.22
- NVIDIA GeForce RTX 4090 Laptop GPU（16GB）

建议使用课程虚拟环境：

```powershell
E:\Anaconda3\envs\BDpy312\python.exe run_rule.py
E:\Anaconda3\envs\BDpy312\python.exe train_bilstm.py
E:\Anaconda3\envs\BDpy312\python.exe train_bert.py
E:\Anaconda3\envs\BDpy312\python.exe summarize.py
```

## 4. 预训练权重（提交仓库不含大文件）

`bert-base-chinese` 的 `model.safetensors` 约 400MB，无法推送到课程 fork，因此未纳入 Git。复现前请将权重放到 `pretrained_model/`，与本目录已有的 `config.json`、词表与分词器配置一并使用。

**方式一：huggingface-cli**

```powershell
pip install -U huggingface_hub
huggingface-cli download bert-base-chinese --local-dir pretrained_model
```

**方式二：Python**

```powershell
python -c "from huggingface_hub import snapshot_download; snapshot_download('bert-base-chinese', local_dir='pretrained_model')"
```

下载完成后，目录中应存在 `pretrained_model/model.safetensors`。

## 5. 复现实验

在本目录依次执行：

```powershell
python run_rule.py
python train_bilstm.py --epochs 3 --batch-size 32
python train_bert.py --epochs 3 --batch-size 16 --max-len 256
python summarize.py
```

快速检查代码流程：

```powershell
python run_rule.py --max-eval 20 --run-name rule_smoke
python train_bert.py --epochs 1 --max-train 64 --max-eval 32 --batch-size 4 --no-save-model --run-name bert_smoke
```

所有方法的预测均写成 `{id, sent_id, events}`，并由 `common.evaluate` 计算触发词 F1、论元 F1 与事件级精确匹配 F1。详细数字见 `outputs/metrics/` 与 `实验报告.md`。

BERT 微调后的完整 checkpoint 体积较大，默认不放入提交目录。复现时重新运行 `train_bert.py` 即可在 `outputs/checkpoints/` 生成。`pretrained_model/` 中提交的是配置与词表；权重按上一节下载后可离线加载。
