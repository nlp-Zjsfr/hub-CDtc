# Qwen3-0.6B-Base zero-shot / few-shot 事件抽取

在 `week7作业` 目录运行。复用相邻原课程的 `code/data/processed` 数据和作业公共评测器，不进行微调，不默认调用 API。

```bash
# 本次 16GB Mac 的实测内存水位配置；不影响抽取规则。
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.65
export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5
python llm_zero_shot/llm_extract.py --mode zero --n 60 --device mps --dtype fp16
python llm_zero_shot/llm_extract.py --mode few --n 60 --device mps --dtype fp16
```

两种模式使用 seed=42 的同一批样本。默认 `--n 200`，`--n 0` 评测全量 930 条。
本地模型为 `qwen3-0.6b-base`；Base 版本的指令遵循能力应由实测判断，不能套用其他模型的课件数字。

## 文件

- `llm_extract.py`：本地/API/mock 抽取与统一评测入口。
- `event_prompt.py`：课程提示词和输出解析。六条 few-shot 示例均来自当前 train，带原始 id/sent_id；其中两条课程示例不在当前 train 中，已替换为同类型训练样本，答案统一由训练金标序列化。
- `event_data.py`：生成式链路的数据格式、固定采样及金标子集。
- `local_generate.py`：CUDA/MPS/CPU 生成、左 padding、贪心解码和 token/延迟统计，供后续 LoRA 评测复用。
- `model_paths.py`：本地 Qwen 模型和共享数据路径，不依赖尚未创建的 llm_lora 目录。

默认 MPS batch=1、CPU batch=1、CUDA batch=8。`--gen_batch` 可显式覆盖；MPS 每生成 32 个 token 以及每批结束时会清理未使用的缓存。MPS/CPU 的 `--dtype auto` 使用 fp32，CUDA 自动选择支持的半精度。
`--max_length 4096` 是完整提示词上限，另有 `--max_new_tokens 384` 的输出预算。超长会报错，不会静默丢弃 schema 或示例。

## 产物

以 zero-shot 60 条为例：

- `outputs/llm/raw_qwen_base_zero_n60.jsonl`：原始回复、解析事件、token/延迟与截断统计。
- `outputs/llm/pred_qwen_base_zero_n60.jsonl`：统一格式预测。
- `outputs/llm/gold_qwen_base_zero_n60.jsonl` 与 `eval_ids_qwen_base_zero_n60.json`：本次实际评测子集。
- `outputs/llm/report_qwen_base_zero_n60.json`：JSON 合法率、解析失败、幻觉、生成长度等。
- `outputs/metrics/llm/qwen_base_zero_n60.json`：公共评测器的触发词、论元、完整事件 F1。
- `outputs/logs/qwen_base_zero_n60_eval.json` 与 `_console.log`：评测日志。

只评测 60 条时，必须和同一批 60 条上的其他模型结果比较，不能直接与全量 930 条的数字比较。

## 离线测试和复用回复

```bash
python llm_zero_shot/llm_extract.py --dry_run --n 20
python llm_zero_shot/llm_extract.py --mode few --print_prompt
python llm_zero_shot/llm_extract.py --from_raw outputs/llm/raw_qwen_base_zero_n60.jsonl --n 60
```

模拟结果明确标记 `is_mock=true`，只用于检查流程。
`--from_raw` 默认使用新的 `_reparse` 前缀，保留原回复；样本不齐会报错，不会把缺回复的样本悄悄补空。
API 后端保留课程接口，需显式指定 `--backend api` 并配置 `LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL`，本地复现不调用 API。

`report` 的 `json_valid_rate` 表示经课程解析器容错后可解析的比例，包含结构性修复及围栏/噪声剥离，不代表原始回复严格合法。原回复和修复次数均保留。


## DeepSeek 官方 API 抽取

在已激活 `pytorch` 的终端中，进入 `week7作业`：

```bash
python -m pip install -r llm_zero_shot/requirements-api.txt
export DEEPSEEK_API_KEY="在本机填入实际密钥"
export LLM_BASE_URL="https://api.deepseek.com"

# 先验证 5 条；默认模型 deepseek-flash、关闭思考、JSON 模式、输出上限 1024。
python llm_zero_shot/llm_extract.py --backend api --model deepseek-flash --mode zero --n 5 --resume

# 验证成功后，两组使用与本地 Qwen 相同的 60 条 dev。
python llm_zero_shot/llm_extract.py --backend api --model deepseek-flash --mode zero --n 60 --resume
python llm_zero_shot/llm_extract.py --backend api --model deepseek-flash --mode few --n 60 --resume
```

环境变量只对当前终端及其子进程生效，应在配置密钥的同一个终端运行命令。
代码、日志和请求配置文件不保存密钥。

默认输出前缀为 `deepseek_v41_zero_n5`、`deepseek_v41_zero_n60`、`deepseek_v41_few_n60`，不覆盖已有 Qwen 结果。
报告、预测、金标子集、原始回复的路径规则与上文一致，另有 `request_<tag>.json` 保存请求参数和样本/提示词摘要。

`--resume` 复用已落盘的成功回复，仅补调未完成或失败的样本；每条回复原子写入断点文件。已完成后再次运行会复评，不调用 API。
中断时正在进行的请求可能需要重发。模型、样本、提示词或参数变化时应使用新的 `--tag`，程序会拒绝混用不同实验的断点。

报告另含请求次数、本次新增请求次数、缓存命中/未命中 tokens、输出 tokens、模型返回标识、结束原因和按峰时价格计算的美元估算。
重试中已返回的 usage 也计入用量；没有返回 usage 的网络中断无法精确估算费用，实际扣费以平台账单为准。
价格快照日期为 2026-11-26，来源：https://api-docs.deepseek.com/quick_start/pricing/ 。
模型名 `deepseek-flash` 对应官方当前 V4.1 Flash 服务；保存的响应模型标识与调用日期用于追溯。

本地代码的 API 逻辑已通过离线替身测试，正式效果与接口连通性需由真实 API 试跑确认。
