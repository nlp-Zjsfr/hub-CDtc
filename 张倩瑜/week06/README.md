# Week06 · Propaganda Technique Classification

用课件「数据接入与分流」的方法，做 **SemEval-2020 Task 11** 的技术分类（TC）：给定英文新闻里的一个宣传片段，判断它用了哪种手法。

基座模型：**Qwen2.5-0.5B-Instruct**。

## 数据

用的是 [`SemEvalWorkshop/sem_eval_2020_task_11`](https://huggingface.co/datasets/SemEvalWorkshop/sem_eval_2020_task_11) 的原文和金标。HF 仓库只有加载脚本，正文不在仓库里；公开可下的同一份语料是 Zenodo 上的 PTC v2（[3952415](https://zenodo.org/records/3952415)，CC-BY 4.0）。标签字符串与 HF 数据集的 14 个 ClassLabel 一致，没有改名。

- 脚本：`data/build_dataset.py`
- 一条样本 = 一条 TC 标注。`sentence` 是按字符偏移切出的 span，`label_name` 是原手法名
- 公开包里的 dev 只有待填的 `?`，没有金标。验证集是从训练文章里按文章持出的 15%（seed=42），同一篇文章不会同时出现在 train 和 val
- 同一 span 若标了多个手法，按标注各留一条
- 规模：6129 条标注 → train 5210 / val 919。`clean_val()` 去掉与训练集完全相同的片段后剩 820 条（重复 99 条）

重新拉取并切分：

```bash
python data/build_dataset.py
```

## 方法阶梯（对照课件 L0–L5）

| 层级 | 方法 | 脚本 |
|---|---|---|
| L0 | 规则 / 关键词 | `rule/classify_rule.py` |
| L1 | 词 TF-IDF + LinearSVC | `tfidf/classify_tfidf.py` |
| L2 | bert-base-uncased 微调 | `bert/train_bert.py` |
| L3 | Qwen2.5-0.5B zero-shot | `llm_zero_shot/classify_zero_shot.py` |
| L4 | Qwen2.5-0.5B + LoRA SFT | `llm_lora/train_lora.py` + `eval_lora.py` |
| L5 | 分流 CLI（置信度→队列） | `service/classify_cli.py` |

统一评测：同一批 `clean_val`（820 条）。对比指标是 **Micro-F1**（14 类上计算；预测都落在这 14 类里时，它等于准确率）。Macro-F1 仍保留，用来看小类有没有被照顾到。

TC 金标全是宣传手法，没有「无宣传」类。L5 的队列只有 `flag_bias`（高置信）和 `review`（低置信）。L5 的 Micro-F1 评的是路由输出的手法标签，不是队列名。

## 本机实测（clean_val，n=820，seed=42）

| 层级 | 方法 | Micro-F1 | Macro-F1 | 解析失败 |
|---|---|---|---|---|
| L0 | 规则 / 关键词 | 0.3329 | 0.1103 | 0 |
| L1 | TF-IDF + LinearSVC | 0.4195 | 0.3004 | 0 |
| L2 | BERT 微调 | 0.5341 | 0.2972 | 0 |
| L3 | Qwen2.5-0.5B zero-shot | 0.0398 | 0.0326 | 31 |
| L4 | Qwen2.5-0.5B + LoRA | 0.5878 | 0.4116 | 0 |
| L5 | 分流 CLI | 0.4305 | 0.3144 | 0 |

按 Micro-F1：L4 > L2 > L5 > L1 > L0 > L3。BERT 微调 3 个 epoch（`bert-base-uncased`，lr 2e-5，batch 16），Micro-F1 高于 TF-IDF，Macro-F1 与 TF-IDF 接近。LoRA 两类 F1 都最高。零样本仍塌缩，31 条输出对不上标签名。

L5 用的是 CLI 默认的 TF-IDF + SGD，所以和 L1 的 LinearSVC 不是同一个分类器。

## 五分钟跑通

```bash
pip install torch transformers peft scikit-learn tqdm

hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir pretrain_models/Qwen2.5-0.5B-Instruct
hf download bert-base-uncased --local-dir pretrain_models/bert-base-uncased

python data/build_dataset.py
python rule/classify_rule.py
python tfidf/classify_tfidf.py
python bert/train_bert.py --epochs 3

python llm_zero_shot/classify_zero_shot.py --n 820
python llm_lora/train_lora.py --epochs 3
python llm_lora/eval_lora.py
python service/eval_router.py

python eval/eval_all.py

python service/classify_cli.py --text "The next transmission could be more pronounced or stronger" --json
```

## 目录

```
week06/
├── common/          # 标签 / 数据切分 / 指标 / 模型路径
├── data/            # build_dataset.py + train/val.json
├── rule/            # L0
├── tfidf/           # L1
├── bert/            # L2
├── llm_zero_shot/   # L3
├── llm_lora/        # L4
├── service/         # L5
├── eval/            # 汇总表
├── outputs/         # 预测与指标
└── pretrain_models/ # Qwen2.5-0.5B-Instruct
```

## 标签表

| Label | 含义 |
|---|---|
| Appeal_to_Authority | 引用权威，不给可核对的证据 |
| Appeal_to_fear-prejudice | 唤起恐惧或成见 |
| Bandwagon,Reductio_ad_hitlerum | 人人都如此，或比作纳粹 |
| Black-and-White_Fallacy | 只有两种选择 |
| Causal_Oversimplification | 归结为单一原因 |
| Doubt | 质疑信誉或事实，无具体反证 |
| Exaggeration,Minimisation | 夸大或缩小 |
| Flag-Waving | 诉诸国家或群体认同 |
| Loaded_Language | 带强烈感情色彩的用词 |
| Name_Calling,Labeling | 用标签定性 |
| Repetition | 反复传达同一信息 |
| Slogans | 用口号代替论证 |
| Thought-terminating_Cliches | 用套话打断讨论 |
| Whataboutism,Straw_Men,Red_Herring | 那别人呢、歪曲观点或岔开话题 |

引用：Da San Martino et al., SemEval-2020 Task 11；语料来自 Da San Martino et al., EMNLP 2019。
