# Week6 分类任务 · 作业工作区

> 本目录是 week6「数据接入与分流」的**作业专用工作区**：代码、数据、权重、结果、课件、作业素材已全部拷齐，可脱离原目录独立运行。
> 源目录：week6-数据接入与分流/　｜　生成日期：2026-11-17，实验补跑：2026-11-18
> **先看这个**：`对比实验报告.md`（作业交付主文档）

---

## 零、先看懂这个作业（10 分钟路线图）

**一句话说清**：把一条新闻标题判成 15 类里的哪一类，再归并到 4 个业务组 / 相关与否，最后把它**路由到 research / review / drop 三个队列**；在这之上做三组对比实验：**换方法**、**换预训练模型**、**换任务口径**。

**按这个顺序读**（每一步都有“看到什么算懂了”的标志）：

| 步 | 读什么 | 看到什么算懂了 |
|---|---|---|
| 1 | `对比实验报告.md` 第一节（三线实验设计表） | 能说清三条线各自只动了一个变量 |
| 2 | `code/common/labels.py` | 知道 15 类 → 4 组 → 相关性 → 队列 的映射链 |
| 3 | `code/common/data.py` | 知道为什么评测要去掉 11% 的重复样本（泄漏） |
| 4 | `code/common/metrics.py` | 知道 acc / Macro-F1 / ECE 各自回答什么问题 |
| 5 | `code/rule/` → `code/tfidf/` → `code/bert/train_bert.py` → `predict_bert.py` | 能解释“为什么从弱到强”的原因 |
| 6 | `code/service/classify_cli.py` | 看懂“分类 → 动作”的接口契约（下游要的是 conf 和 queue） |
| 7 | 作业目录三个脚本：`换模型对照实验.py`、`汇总分组指标.py`、`plot_confusion.py` | 知道每个实验的输入 / 输出 / 控制变量 |

**自己动手：三条命令，从 5 分钟到 15 分钟**

```bash
cd /d "f:\冯天运\AI\AI大模型&Agent\八斗\主课程\week6-分类任务\作业"

# ① 5 分钟：先确认环境和数据都是好的（不训练、不耗电算力）
cd code
python common/model_paths.py                 # 应打印两个模型都解析到路径
python rule/classify_rule.py --n 500         # L0 基线：Macro-F1 约 0.07（就是随机水平）
python tfidf/classify_tfidf.py --n 500       # L1 基线：Macro-F1 约 0.47，且只用秒级
python eval/eval_all.py                       # 汇总到 code/outputs/summary.md（看七路表）
cd ..

# ② 10 分钟：看“分类变成动作”是什么样
python code/service/classify_cli.py --text "贵州茅台：2023 年年度业绩预告" --json
python code/service/demo_early_report.py      # 19 条消息被分成 research/review/drop 三桶

# ③ 15 分钟：作业 B 的诊断链（不训练只看控制变量）
python 换模型对照实验.py --steps 50 --models chinese-roberta-wwm-ext
python 汇总分组指标.py              # 同一份预测重算 15 类 / 4 组 / 相关性
python plot_confusion.py --group              # 出业务 4 组混淆矩阵
```

> 每跑完一条都看“**它输出了什么**”，不要只看是否报错：
> `rule/tfidf/bert` 三个脚本都会写一个 `outputs/*_pred.json`，里面有 `meta`（口径）+ `report`（指标）+ `preds`（逐条明细）。

**重点在哪（时间有限，只看这 5 处）**

1. `code/common/model_paths.py` 的 `BERT_DIR = os.getenv("WEEK6_BERT_DIR", "bert-base-chinese")` → **作业 B 的全部机关就在这一行**（换模型不改一行代码）
2. `code/common/data.py` 的 `clean_val` → **口径问题的根**（不去重就会虚高 3 个点）
3. `code/bert/train_bert.py` 里 `BertClassifier.forward` 的 **mask 池化** → 最容易写错的隐形 bug（Mean/Max 不 mask 会把 [PAD] 算进去）
4. `code/common/metrics.py` 的 `acc_incl_unparsed` → **生成式方法的诚信底线**（解析失败必须计入分母）
5. `换模型对照实验.py` 里 `env["WEEK6_BERT_DIR"] = model` 那一行 → **控制变量法的实现**（变量只有 backbone，其余全对齐）

**不要在这个作业里写的东西**（别去花时间冲流模）：一是“调 lr / 换池化 / 开 FGM”（课件已实测只动 1~2 个点）；二是“在本机跑微调”（CPU 上是十小时量级）。

---

## 一、目录结构

```
作业/
├── README_作业_说明.md        ← 本文件
├── 对比实验报告.md             ← 作业交付主文档：三条实验线（方法横向 / 换模型 / 换口径）
├── 作业素材_混淆分析.md       ← 作业 A 的现成数据与写作草稿（已算好，直接用）
├── 换模型对照实验.py           ← 作业 B 主交付：两个预训练模型等步数同口径对照
├── 切换模型实验.py             ← 换模型冒烟/单次实验（备份 → 训练 → 恢复原 ckpt）
├── 回填对比结果.py             ← 把对照实验结果写进报告的占位符
├── 汇总分组指标.py             ← 实验 C：15 类 / 4 组 / 相关性 三口径重算 + 误差结构
├── auto_fill_watcher.py        ← 守望进程：训练跑完自动回填（辅助工具）
├── plot_confusion.py           ← 作业 A 配套：混淆矩阵热力图（15 类 / 业务 4 组）
├── code/                      ← 完整代码（可运行）
│   ├── common/                # 标签口径 / 数据去重 / 统一指标 / 模型定位
│   ├── rule/ tfidf/ bert/ llm_zero_shot/ llm_api_zero_shot/ llm_lora/
│   ├── service/               # classify_cli.py 最终服务、classify_news.py 教学版、demo_early_report.py
│   ├── eval/                  # eval_all.py 七路汇总
│   ├── data/                  # train.json 53,360 条 + val.json 10,000 条
│   └── outputs/               # 各路预测结果 + summary.md + bert_ckpt.pt（已训好的权重）
├── pretrain_models/           # 目录联接（不占空间）
│   ├── bert-base-chinese          → 主课程/pretrain_models/bert-base-chinese（真实目录，safetensors）
│   ├── chinese-roberta-wwm-ext    → 主课程/pretrain_models/chinese-roberta-wwm-ext
│   └── Qwen3-0.6B                 → 主课程/pretrain_models/Qwen3-0.6B
├── outputs/                   # 作业 A 的图（混淆矩阵.png / 混淆矩阵_4组.png）+ 分组指标 JSON
├── 课件/                      # week6 课件 PPT（3.4 MB）
└── 产品设计稿/                # 投研 / 法务 / 医疗 / 电商 / 招聘 五个静态产品稿
```

---

## 二、PPT 里的三项作业

| 项 | 出处 | 原文要求 | 交付物 | 本次完成情况 |
|---|---|---|---|---|
| A 混淆分析 | P27 | 挑一组最容易混的，写清楚打算怎么解决 | 1 页分析 | ✅ `作业素材_混淆分析.md` + 两张混淆矩阵图 |
| B 自主实验 | P49 | 自己做实验，可以换数据集 / 预训练模型 / api 接口 | 对比表 + 结论 | ✅ 换了预训练模型（roberta-wwm-ext）+ 换了任务口径，见 `对比实验报告.md` |
| C 简历数字 | P47 | 把这周的数字填进那段话，发到群里 | 一段项目描述 | ⏸ 暂缓（按你的要求先不管） |

---

## 三、开跑前先自检（3 条）

```bash
cd /d "f:\冯天运\AI\AI大模型&Agent\八斗\主课程\week6-分类任务\作业\code"
python common/model_paths.py            # 应打印两个模型都解析到路径
python eval/eval_all.py --run-light      # 约 30 秒，重跑规则 + TF-IDF 并刷新 summary.md
python service/classify_cli.py --text "贵州茅台：2023 年年度业绩预告" --json
```

第三条应看到 `"model": "bert-cls-v1"`（说明 ckpt 与 bert 权重都到位）；若显示 `tfidf-sgd-v1` 说明权重没接上。

---

## 四、作业 A：混淆分析怎么做（主作业）

**第 1 步 取数据**（不用重训，ckpt 已随包）：
```bash
python bert/predict_bert.py --n 0    # 全量干净集 8,901 条，输出 outputs/bert_pred.json
```

**第 2 步 挑一组** —— 推荐「科技 ↔ 财经」：合计 334 次误判，是最大的一对，且**跨业务组**（产业 vs 金融），会真的把消息分错队列；而国际↔军事、娱乐↔故事虽然量也大，但都落在同一业务组内，对「要不要看」没有影响。

**第 3 步 写原因**（三条）：
1. 词面触发：标题里出现融资 / 投资 / 上市 / 合作 / 产值，模型就抓「财经」；出现余额宝 / 区块链 / 手机业务，就抓「科技」。
2. 标签口径重叠：财经类里大量是科技公司，产业类里大量是资本动作，两类边界本身就模糊。
3. 短标题缺实体与语境，只能靠词面猜 —— 这正是下周「信息抽取」要补的能力。

**第 4 步 写方案**（按 ROI 排序，用已有数据佐证）：
| 优先级 | 方案 | 依据 |
|---|---|---|
| ★1 | 标签口径降到业务粒度（15 类 → 4 组 + 相关性二分类） | 同一次预测重算：Macro-F1 0.5312 → 0.6833，相关性 F1 0.8009；合并消掉 39.7% 的错分 |
| ★2 | 合并重叠标签（科技 + 财经 合一） | 同上，需业务确认可接受 |
| ★3 | 输入换成「标题 + 摘要」或引入信息抽取 | 预期最大，依赖下周能力 |
| ★4 | 补长尾标注 + 类别加权 | 长尾 recall 回升，但对总分只动 1~2 个点 |

**不要写**：调 lr / 换池化 / 开 FGM —— P25 的 9 组消融已证明这些只动 1~2 点，瓶颈在标签口径。

完整数据表与错分样例见 **作业素材_混淆分析.md**。

---

## 五、作业 B：自主实验的三条路径

**路径 1 换预训练模型**（最快出对比，权重已备好）：
`chinese-roberta-wwm-ext` 已下好并转成 safetensors，放在 `主课程/pretrain_models/chinese-roberta-wwm-ext/`
（config.json / vocab.txt / tokenizer_config.json / special_tokens_map.json / model.safetensors）。
**与 bert-base-chinese 同架构**（bert / hidden 768 / 12 层 / vocab 21128），所以代码一行都不用改。

切换方式 = 改一个环境变量（`code/common/model_paths.py` 里 `BERT_DIR = os.getenv("WEEK6_BERT_DIR", "bert-base-chinese")`）：

```bash
cd /d "f:\冯天运\AI\AI大模型&Agent\八斗\主课程\week6-分类任务\作业"
set WEEK6_BERT_DIR=chinese-roberta-wwm-ext
python code/bert/train_bert.py --epochs 3
python code/bert/predict_bert.py --n 0
```

或用一键脚本（会**自动备份并在结束后恢复原 bert ckpt**，跑完不用手工改回）：

```bash
python 切换模型实验.py            # 冒烟：5 步训练验证链路，约 105 秒，不落盘
python 切换模型实验.py --full     # 完整 3 epoch + 500 条评测，输出 bert_ckpt_chinese-roberta-wwm-ext.pt
```

**要做成"对比实验"（作业 B 的正式交付）用这个**：
```bash
python 换模型对照实验.py --steps 50 --models bert-base-chinese
python 换模型对照实验.py --steps 50 --models chinese-roberta-wwm-ext
```
它把两个模型的**数据、步数、超参、评测集**全部对齐，只换 backbone，产出
`outputs/model_swap_experiment.json` 与各自独立的 `code/outputs/ckpt_<model>_s50.pt`（不覆盖原权重），结果自动合并。
再推进一步就能把“差多少”说清楚（不用重训，只用同一份 ckpt 重评 + 配对 bootstrap）：

```bash
# 重评全量 8,901 条（分词器必须与权重同源，要切实现变量）
set WEEK6_BERT_DIR=bert-base-chinese
cd code && python bert/predict_bert.py --ckpt outputs/ckpt_bert-base-chinese_s50.pt --n 0 --out outputs/bert_pred_bert-base-chinese_s50_full.json && cd ..
set WEEK6_BERT_DIR=chinese-roberta-wwm-ext
cd code && python bert/predict_bert.py --ckpt outputs/ckpt_chinese-roberta-wwm-ext_s50.pt --n 0 --out outputs/bert_pred_chinese-roberta-wwm-ext_s50_full.json && cd ..

python 配对bootstrap检验.py     # 配对 bootstrap 95% CI + 逐类 F1 对比
```

**本次实测结论**：

| 口径 | bert-base-chinese | roberta-wwm-ext | Δ Macro-F1 | 95% CI | 显著？ |
|---|---|---|---|---|---|
| 500 条抽样 | 0.3964 | 0.4277 | +0.0313 | [−0.0177, +0.0798] | **否**（跨 0） |
| 8,901 条全量 | 0.3831 | 0.4399 | **+0.0567** | **[+0.0456, +0.0675]** | **是**（净变化 +503 条） |

提升集中在长尾类（财经 +0.18、证券 +0.13），头部类基本持平。
一个必须记住的口径事实：这是**欠训练（50 步）**下的差距；参考课程标准答案（3 epoch 收敛后）
的差距只有 **+0.0100**（CI [+0.0019, +0.0178]）—— 两者不矛盾，反而说明 **roberta 收敛更快**，
所以在小预算场景下优势被放大，到收敛后就只剩下一点。
完整设计、口径与结论见 **`对比实验报告.md`**。

> 为什么是 50 步：本机只有 CPU，bert-base 完整 1 epoch 实测 15,978 s（≈4.44 h）。50 步是"前台跑得完、且两模型严格同口径"的规模。结论应读作"**同等训练预算下 roberta 更优**"。

**路径 2 换数据集**（按难度递增）：
1. 零下载：把 `common/labels.py` 的 `GROUP_MAP` 换成你自己的分组，即作业 A 的方案一（本次已做，见实验 C）；
2. 真实数据：巨潮资讯实测可达（`www.cninfo.com.cn` 返回 200），可抓公告做四字段统一；
3. 公开语料：CLUE 的 CLUENER / AFQMC，经 hf-mirror 拉取。

**路径 3 换 API**（代码零改动，课程代码已支持）：
```bash
set LLM_API_KEY=xxx
set LLM_BASE_URL=https://api.deepseek.com
set LLM_MODEL=deepseek-v4-flash
python llm_api_zero_shot/classify_api_zero_shot.py --n 500 --workers 8
```

换通义 / 智谱 / Kimi / 本地 vLLM 只改环境变量；网关要额外 header 用 `LLM_EXTRA_HEADERS`。顺带把 P40 的坑复现一遍：推理模型不关思考会吃光 `max_tokens` → 指标虚高。

**对比表模板**（交付物）：
| 方法 | 数据 | 模型 | Macro-F1 | 覆盖类别 | 延迟 | 成本 | 一句话结论 |
|---|---|---|---|---|---|---|---|

---

## 六、作业 A 的配图（已画好）

```bash
cd /d "f:\冯天运\AI\AI大模型&Agent\八斗\主课程\week6-分类任务\作业"
python plot_confusion.py                 # → outputs/混淆矩阵.png（15 类热力图 + 每行 recall）
python plot_confusion.py --group         # → outputs/混淆矩阵_4组.png（业务 4 组）
```

读图结论可直接写进作业 A：**科技↔财经 334 次误判**（跨业务组，必须改），合并成 4 组后 Macro-F1 0.5312 → 0.6833、acc +15.2 个点、相关性 F1 0.8009。

---

## 七、五个别踩的坑

1. **口径统一**：对外只报去重后（clean_val）的数字；别把带 11% 重复的 val 结果写进简历。
2. **两个口径别混**：`outputs/summary.md` 是 500 条抽样（BERT 0.5436）；`bert_pred.json` 是 8,901 条全量（0.5312）。写的时候标明是哪个。
3. **生成式方法**：解析失败必须计入分母（`acc_incl_unparsed`），否则靠丢样本虚高。
4. **别在本机跑微调**：CPU 上 LoRA / 全量微调是十小时量级；本机只做推理与流程验证，训练结论引用课件口径并标明来源。
5. **ECE 0.1373 大于 0.1**：置信度不能直接当阈值用，要先做温度缩放（这正是 P46 的「过度自信」）。小预算模型（本报告实验 B 的 0.21~0.26）更是完全不能用。

---

## 八、简历那栏（暂缓）

模板在 `code/RESUME_GUIDE.md`（好句 vs 差句、12 个面试追问）。等你作业 A / B 的数字定稿后再填 —— 原则：没跑过的数字宁可不写。

---

## 九、Git 提交清单（照这个交）

仓库建在**本目录**（`作业/` 本身自包含，拷走即可运行）：

```bash
cd /d "f:\冯天运\AI\AI大模型&Agent\八斗\主课程\week6-分类任务\作业"
git init
git add .
git status                     # 确认清单见下表
git commit -m "week6 分类任务：七种方法对比 + 换预训练模型实验 + 换任务口径实验"
```

**要交**（已带 `.gitignore`，自动排除大文件）：

| 路径 | 内容 | 体积 |
|---|---|---|
| `code/` | 全部源码（16 个 .py）+ 文档（ARCHITECTURE / USAGE_GUIDE / README / RESUME_GUIDE） | ~60 KB |
| `code/data/` | train.json 53,360 条 + val.json 10,000 条 | 8.3 MB |
| `code/outputs/` | 各路 `*_pred.json`、`summary.md`、`bert_train_log.json`、对照实验的 `bert_pred_*_s50.json`（500 条）、`bert_pred_*_s50_full.json`（8,901 条）与 `swap_*.log` | ~5 MB |
| `outputs/` | `混淆矩阵.png`、`混淆矩阵_4组.png`、`group_metrics.json`、`model_swap_experiment.json`、`paired_bootstrap.json` | ~320 KB |
| `换模型对照实验.py`、`切换模型实验.py`、`回填对比结果.py`、`汇总分组指标.py`、`plot_confusion.py` | 本周脚本 | ~15 KB |
| `对比实验报告.md`、`README_作业_说明.md`、`作业素材_混淆分析.md` | 作业说明与现成数据/草稿 | ~34 KB |
| `课件/`、`产品设计稿/` | 课件 PPT + 5 个静态产品稿 | 3.8 MB |
| `.gitignore` | 排除权重/联接/缓存 | 302 B |

**不要交**（已在 `.gitignore` 里）：

| 路径 | 原因 |
|---|---|
| `code/outputs/bert_ckpt.pt`、`code/outputs/ckpt_*_s50.pt` | 409 MB × 3，超 GitHub 单文件 100 MB 限制 |
| `pretrain_models/` | 是指向 `主课程/pretrain_models` 的目录联接，提交后在别人机器上是死链；权重改用 README 里的下载说明 |
| `__pycache__/` | 编译缓存 |

**提交后 README 里要写清的三件事**（老师复现靠这个）：
1. 权重怎么拿：`bert-base-chinese` / `chinese-roberta-wwm-ext` 均为 safetensors 格式，从 hf-mirror 拉取后转 safetensors；
2. 数据怎么来：`code/data/` 已随仓库附上；
3. 复现命令：第五节的三条路径 + 本节的命令。
