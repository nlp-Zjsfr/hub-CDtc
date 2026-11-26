# week7 课后作业 · 抽取任务：换模型 / API / 数据集，对比效果差异

> 对应课堂内容：week7「信息抽取」—— 事件抽取 / 实体抽取 / 大模型零样本抽取。
> 作业要求：**换不同模型 / API / 数据集做抽取类任务，对比各种方法的效果差异**。
>
> 风格参考 week06 参考答案：**控制变量只换命令行参数，跑两遍，再对比出结论**。
> 整个作业就 3 个脚本 + 这份说明，不搞模块化工程。

---

## 1. 参考答案做了什么

| 项 | 内容 |
|---|---|
| 任务 | ① 金融事件抽取（DuEE-Fin：事件类型 + 触发词 + 角色论元）<br>② 中文医疗实体识别（CMeEE：实体类别 + 实体文本） |
| 对照维度 | **本地小模型**（不需要 key）：`bert-base-chinese` / `chinese-roberta-wwm-ext` 微调；**Qwen3-0.6B** 本地生成；**API 大模型**（需 key）：DeepSeek / 通义千问 / 智谱 / Moonshot / OpenAI；**提示词**：zero-shot vs few-shot；**数据集**：两个 |
| 只改了哪里 | `extract.py` 的命令行参数（`--provider` / `--model` / `--dataset` / `--mode`），其余一行不动 |
| 统一口径 | 所有方案都算 **entity F1**（句中 `(类别/角色, 文本)` 完全匹配），跨数据集可比；事件型数据集另算 trigger F1 / event F1 |
| 评测集 | DuEE-Fin 用课件已构建好的 dev；CMeEE 用本包自带的 16 条小样本（离线可跑） |

**为什么要统一到 entity F1**：两个数据集标注形态不同（一个是"事件→角色"两层，一个是"实体类别"一层），
只有"带类别的文本片段"这个概念在两者上都成立。如果各算各的指标，
"谁更好"就没法比 —— 和 week06 里坚持用同一个 clean val 是同一个道理。

---

## 2. 怎么跑（三步）

### 第 0 步：设 API key

```powershell
# 只用一个厂商时，设通用变量最省事
$env:LLM_API_KEY = "sk-xxxxxxxx"

# 或者按厂商分别设（可同时跑多家做对比）
$env:DEEPSEEK_API_KEY = "sk-xxxxxxxx"
$env:DASHSCOPE_API_KEY = "sk-xxxxxxxx"
```

**没有 API key 也能完成作业**，有两条路（都不联网、不花钱，结果可写进报告）：

1. **本地 BERT 小模型**（序列标注微调，`extract_bert.py`）—— 判别式基线，最稳
2. **本地 Qwen3-0.6B**（生成式抽取，`extract.py --provider local`）

只有 `mock` 是假模型，不能写进报告。

```powershell
# 确认本地模型在课程根目录 pretrain_models/Qwen3-0.6B/（只要有 config.json 和权重）
```

### 第 1 步：跑各个方案

**优先跑 CMeEE**（标签少、句子短、CPU 上几十秒一组，能出有效对比数字）：

```powershell
# 控制变量 1：只换基座模型
python extract_bert.py --dataset cmeee --model bert-base-chinese --epochs 10 --n 16 --tag bert_cmeee10
python extract_bert.py --dataset cmeee --model chinese-roberta-wwm-ext --epochs 10 --n 16 --tag bert_cmeee10_roberta

# 控制变量 2：只换训练轮数（同一个 bert-base-chinese）
python extract_bert.py --dataset cmeee --model bert-base-chinese --epochs 5 --n 16 --tag bert_cmeee5
```

**可选：DuEE-Fin 事件抽取**（标签 149 个，纯 CPU 小模型基本不收敛，
只用来说明"任务复杂度"的影响，不建议在这上面调参刷分）：

```powershell
python extract_bert.py --dataset duee_fin --model bert-base-chinese --epochs 2 --n 300 --tag bert_cpu300
python extract.py --dataset duee_fin --provider local --n 5 --tag qwen_local_5
```

**有 API key 的话**（效果最好，几秒一条），把上面换成：

```powershell
python extract.py --dataset duee_fin --provider deepseek --mode zero
python extract.py --dataset duee_fin --provider deepseek --mode few
python extract.py --dataset duee_fin --provider qwen --model qwen-plus --mode zero
python extract.py --dataset cmeee --provider deepseek --mode zero
```

离线冒烟（验证流水线，数字无意义、不能写进报告）：

```powershell
python extract.py --dataset cmeee --provider mock --n 16
```

每跑一次生成两份文件：

```
out/<tag>_log.json    指标 + 耗时 + token + JSON 合法率
out/<tag>_pred.json   逐条预测（复核 bad case 用）
```

### 第 2 步：对比 + 画图

```powershell
python compare.py     # → out/compare.md + out/compare.csv
python plot.py        # → out/compare.png
```

把 `out/compare.md` 的表格贴进本报告第 3 节，图贴进作业文档即可。

---

## 3. 结果对比（本机实测，`python compare.py` 生成）

> 本机无 API key、无 GPU，全部用本地模型在 CPU 上跑。下面是真实数字。

### CMeEE（医疗实体，实体型；标签只有 9 类，小样本能学会）

| 方案 | n | entity F1 | P | R | 耗时(s) |
|---|---|---|---|---|---|
| local-bert / bert-base-chinese · 10ep | 16 | **0.9375** | 0.9375 | 0.9375 | 31.7 |
| local-bert / chinese-roberta-wwm-ext · 10ep | 16 | 0.8197 | 0.8621 | 0.7812 | 45.7 |
| local-bert / bert-base-chinese · 5ep | 16 | 0.3333 | 0.3235 | 0.3438 | 22.5 |

**控制变量结论**：
- 换基座：同 10ep 下 bert-base-chinese 比 roberta 高 **0.1178**（小样本上不一定越大越好）。
- 换训练轮数：同基座 5ep→10ep，F1 从 0.3333 涨到 0.9375，**+0.6042**，小样本必须训够轮数。

### DuEE-Fin（金融事件，事件型；标签 149 个，小模型全 0）

| 方案 | n | entity F1 | trigger F1 | event F1 | JSON 合法 | 耗时(s) |
|---|---|---|---|---|---|---|
| local-bert / bert-base-chinese · 2ep | 300 | 0.0000 | 0.0000 | 0.0000 | 300/300 | 603.9 |
| local / Qwen3-0.6B · zero | 5 | 0.0000 | 0.0000 | 0.0000 | 4/5 | 815.4 |

**为什么全 0**（不是代码错，诊断已验证 BIO 链路 gold→decode 的 F1≈1.0）：
- 149 个细粒度标签 + CPU 上几百步，远不够收敛，模型全部预测 O；
- Qwen3-0.6B 虽能输出合法 JSON，但角色名/触发词和金标几乎不对齐（如把"公司上市"写成"企业融资"），精确匹配得 0 分；
- CPU 逐条生成约 160s/条，5 条就花了 13.6 分钟。
- 这一组用于说明：**小模型 + 细粒度事件抽取在纯 CPU 上不可行**，和 CMeEE 的对比正好说明"任务复杂度"对效果的影响。

**要回答的问题**：

1. 换模型 / 换训练轮数，entity F1 差多少？（见上，CMeEE 两组对照）
2. 换个数据集（9 类实体 vs 149 类事件标签），掉点多少？—— 直接掉到 0。
3. 时间成本：小样本实体抽取 22~46s 可出结果；事件型即使 300 条训 2ep 也要 10 分钟且不收敛。

---

## 4. 怎么解读（别只看 F1）

- **entity F1 是主指标**：跨数据集可比，也是"抽得准不准"最直接的答案。
- **trigger / event F1 只在 DuEE-Fin 上有**：CMeEE 没有触发词和事件语义，
  表里写 `—` 而不是 0（写 0 会被误读成"这项得 0 分"）。
- **幻觉率**：预测出的文本在原文里找不到的比例。
  它比 F1 更能说明"这个模型会不会编"—— 会编的模型要额外加校验层。
- **JSON 合法率 / 修复数**：生成式模型的输出经常带围栏、少括号、夹废话。
  这段容错写在 `extract.py:parse_output()` 里，合法率低说明工程成本高。
- **解析失败的行不删**：照样写 `events=[]`。删掉会让分母变小、F1 虚高
  （这是生成式方案最常见的作弊方式，哪怕是无意的）。

---

## 5. 常见坑（实测踩过的）

| 现象 | 原因 | 修法 |
|---|---|---|
| 换了 key 还是 401 | 环境变量名不对 | 按 §0 设置；厂商专用名见 `extract.py` 的 `PROVIDERS` |
| 推理模型（如 deepseek-reasoner）回复是空的 | 思考 token 吃光 `max_tokens` | 调大 `--max_tokens`，或换回 chat 模型 |
| 角色"质押股票/股份数量"永远对不上 | prompt 里用 `/` 当角色分隔符，把角色名拆成两个 | 本脚本用「、」分隔（**分隔符不能出现在字段值里**） |
| 模型输出的下标全错 | 生成式模型算中文字符下标极不可靠 | 只让它输出文本，程序用 `text.find()` 回原文对齐 |
| 回复带 ```json 围栏，json.loads 崩 | 模型"多余礼貌" | `parse_output()` 会剥围栏、抠 JSON、补括号 |
| gold vs gold 的 F1 不是 1.0 | 用 `id` 当键，但 `id` 是**文档 id**，同文档多句会互相覆盖 | 本脚本用 `id + sent_id` 对齐 |
| CMeEE 的图里 trigger 列是空的 | 数据集没有触发词 | 正常，写 `—` 不要写 0 |
| `--provider local` 报找不到模型 | 课程根目录缺 `pretrain_models/Qwen3-0.6B/` | 把模型目录放过去（`config.json` 必须在）；或先用 mock 验证流程 |
| 本地模型跑得很慢 | CPU 逐条解码被 kernel 启动开销拖住 | 先加 `--n 50` 试跑；有 GPU 会自动用 CUDA；正式结果建议 API 方案 |
| 本地模型输出一堆思考过程，JSON 合法率低 | Qwen3 默认可能开思考 | 解析函数会从废话里抠 JSON；也可调大 `--max_tokens` |

---

## 6. 目录说明

```
30-week07-作业答案/week7/
├── README.md              本说明
├── extract.py             LLM / API 抽取：换参数跑一次 → out/<tag>_log.json + _pred.json
├── extract_bert.py        本地 BERT 微调抽取（不需要 key）→ 同格式 log/pred
├── compare.py             对比脚本：读所有 log.json → out/compare.md + compare.csv
├── plot.py                画图脚本：→ out/compare.png
├── data/cmeee_mini.jsonl  内置医疗实体小样本（16 条，离线可跑）
└── out/                   运行结果（跑起来后自动生成）
```

数据读取方式：`extract.py` 的 `find_processed()` 会向上逐级找课件里已构建好的
`week7-信息抽取/code/data/processed/`（和 week06 从公共目录读数据一样），
所以**从项目根或本目录启动都能跑**。CMeEE 数据自带小样本，完全离线。

> DuEE-Fin 原始协议限非商业科研/教学；本作业只读课件已构建的派生数据，不分发原始文件。

---

## 7. 可选进阶（答案里没有实现）

- 多 seed 重复跑，看差距是不是噪声（week06 用配对 bootstrap 做了这件事）
- 把 BERT 的 Linear 解码换成手写 CRF，看非法 BIO 率下降多少
- 把错例分成"漏抽 / 多抽 / 类型错 / 边界错"四类统计
- 接一位本地模型（Ollama）和 API 大模型做成本-效果权衡
