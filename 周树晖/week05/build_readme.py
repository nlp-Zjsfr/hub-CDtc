"""把实际训练、生成评估与 SFT 学习要点集中到 README。"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import PercentFormatter, MaxNLocator

from common import ROOT, read_json, read_jsonl


def build_readme():
    cfg = read_json(ROOT / "config.json")
    data = read_json(ROOT / "results" / "data_audit.json")
    training = read_json(ROOT / "results" / "training.json")
    history = read_json(ROOT / "results" / "history.json")
    evaluation = read_json(ROOT / "results" / "metrics.json")
    base, sft = evaluation["results"]["base"], evaluation["results"]["sft"]
    examples = read_json(ROOT / "results" / "examples.json")
    mask = read_json(ROOT / "results" / "mask_example.json")
    first = read_jsonl(ROOT / "data" / "train.jsonl")[0]
    font = Path("C:/Windows/Fonts/msyh.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    else:
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams.update({"axes.unicode_minus":False,"axes.spines.top":False,"axes.spines.right":False,"figure.facecolor":"#FAFAF7","font.size":10})
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout="constrained")
    epochs=[h["epoch"] for h in history]
    axes[0].plot(epochs,[h["train_answer_nll"] for h in history],marker="o",color="#167D8D",label="训练回答 NLL")
    axes[0].plot(epochs,[h["val"]["answer_nll"] for h in history],marker="s",color="#D98336",label="验证回答 NLL")
    axes[0].axhline(training["initial_val"]["answer_nll"],color="#74808A",linestyle="--",label="初始基座验证 NLL")
    axes[0].set(xlabel="Epoch",ylabel="每个有效回答 token 的 NLL（对数轴）",title="SFT 学习曲线",yscale="log")
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True));axes[0].grid(alpha=.25);axes[0].legend(frameon=False)
    import numpy as np
    x=np.arange(3)
    for i,(name,res,color) in enumerate((("Base",base,"#D98336"),("SFT",sft,"#167D8D"))):
        vals=[res["generation"][k] for k in ("schema_valid","exact_match","eos_stop")]
        bars=axes[1].bar(x+(i-.5)*.32,vals,width=.3,color=color,label=name)
        axes[1].bar_label(bars,labels=[f"{v:.1%}" for v in vals],padding=4,fontsize=9)
    axes[1].set(xticks=x,xticklabels=["结构合法","五字段全对","EOS 停止"],ylim=(0,1.19),title="未见模板测试集：生成评估")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1));axes[1].legend(frameon=False,ncols=2);axes[1].set_axisbelow(True);axes[1].grid(axis="y",alpha=.2)
    fig.savefig(ROOT/"results"/"training_curve.png",dpi=170,bbox_inches="tight");plt.close(fig)
    table_data="\n".join(f"| {name} | {r['samples']} | {r['missing_order']} | {r['missing_product']} | {r['unseen_products']} |" for name,r in data["splits"].items())
    table_metrics="\n".join(f"| {name} | {res['losses']['test']['answer_nll']:.6g} | {res['losses']['test']['answer_ppl']:.6f} | {res['generation']['schema_valid']:.2%} | {res['generation']['exact_match']:.2%} | {res['generation']['eos_stop']:.2%} |" for name,res in (("Base",base),("SFT",sft)))
    table_fields="\n".join(f"| {k} | {base['generation']['field_accuracy'][k]:.2%} | {sft['generation']['field_accuracy'][k]:.2%} |" for k in ("order_id","product","intent","problem","urgent"))
    table_history="\n".join(f"| {h['epoch']} | {h['train_answer_nll']:.6g} | {h['val']['answer_nll']:.6g} | {h['val']['answer_ppl']:.6f} | {h['seconds']:.1f} |" for h in history)
    table_mask="\n".join(f"| {r['position']} | {r['token_id']} | `{r['decoded_piece'].replace(chr(10),'<换行>').replace('|','｜')}` | {r['attention_mask']} | {r['label']} |" for r in mask["positions"])
    def fenced(content, language="text"):
        # 模型原始输出本身可能包含 Markdown 围栏，外层必须更长，才能原样展示。
        width = max([3] + [len(run) + 1 for run in re.findall(r"`+", content)])
        fence = "`" * width
        return f"{fence}{language}\n{content}\n{fence}"
    examples_md="\n\n".join(f"**{e['id']}**\n\n用户：{e['input']}\n\n目标：\n{fenced(json.dumps(e['target'],ensure_ascii=False), 'json')}\n\nBase 原始输出：\n{fenced(e['base'])}\n\nSFT 原始输出（五字段全对：{e['sft_correct']}）：\n{fenced(e['sft'])}" for e in examples)
    env=training["environment"]
    text=fr'''# 第五周作业：基于语言模型的 SFT 监督微调

姓名：周树晖　｜　周次：week05

案例：**将中文售后留言抽取成标准 JSON 工单**。使用预训练的 Qwen3-0.6B-Base，通过回答段监督和 LoRA 微调跑通数据、训练、保存、重新加载、基座对比及生成评估。重点是可理解、可验证的 SFT 流程。

选中第 {training['best_epoch']} 轮适配器。在 {cfg['test_size']} 条未见句式的合成测试样本上，SFT 的五字段完全正确率为 **{sft['generation']['exact_match']:.2%}**，基座为 **{base['generation']['exact_match']:.2%}**。这只是本合成任务的结果，不代表真实客服或通用问答能力。

## 一、与本周课程的对应关系

已阅读课件《从 BERT 到大模型》全部 46 页。作业主要对应第 25–33 页：指令—回答数据、Chat Template、回答段 Loss Mask、结束行为，以及预训练/SFT/偏好优化的区别。第 7–13 页的因果注意力和自回归生成是基础。本例不实施 RLHF、DPO 或 API 服务。

课件的简化说法需要补充：Base 模型也可能已有一定指令能力；SFT 也可能记住新事实、改变原有能力，并非只能改变说话格式。“预训练给知识、SFT 给形式”适合帮助入门，但不是绝对边界。

## 二、先看完整流程

```text
明确任务与输出规则
  → 准备高质量的 user/assistant 对
  → 隔离训练、验证、测试数据
  → 加载固定版本的预训练基座和原有 tokenizer
  → 用相同对话协议编码；只给回答和结束符设置 labels
  → 加入 LoRA；执行前向、回答损失、反传和参数更新
  → 用验证集选择 checkpoint
  → 保存适配器；重新加载基座 + 适配器
  → 给出不含答案的新提示，实际生成并评估
```

SFT 不是向聊天 API 发送一些例子。训练时确实发生反向传播和参数更新，最终产物是 [models/best_adapter](models/best_adapter) 中的权重。推理仍然逐 token 生成。

## 三、案例、数据质量和划分

任务只抽取用户表达，不判断用户是否符合退换货政策。固定字段：

| 字段 | 含义与类型 |
| --- | --- |
| `order_id` | 订单号字符串；未提供时为 null |
| `product` | 商品名字符串；未提供时为 null |
| `intent` | return / replace / refund / query_shipping |
| `problem` | damaged / wrong_size / wrong_color / missing_part / late_delivery / not_needed |
| `urgent` | true 或 false，必须是布尔值 |

课程附带 3,000 条训练和 200 条验证指令。抽查发现错误标签，例如将 12、25 错判为不互质，以及把“打篮球”替换成“踢球”却当作同义改写。还有缺失原文和未注明时间的事实类问答。因此本例没有把这些标签直接当可靠金标准，审计证据保存在 [results/course_data_review.json](results/course_data_review.json)。

本案例使用 **[prepare_data.py](prepare_data.py) 确定性生成的合成售后留言**，不是实际客户数据，也不是外部大模型生成的标签。诉求、问题、是否加急、订单和商品来自程序中的事实对象，先确定事实再生成表述，JSON 标签由事实对象直接构造。随机种子为 42。

| 划分 | 样本数 | 缺订单号 | 缺商品名 | 未见商品名 |
| --- | ---: | ---: | ---: | ---: |
{table_data}

三个集合使用互不重合的句式模板；订单号互不重合，同时保持各集合数字分布相近，避免只因编号前缀不同造成偏差。测试集约一半记录使用训练外商品名，明确缺失字段的样本也参与训练。各划分的完整用户输入交集为 0，统计及 SHA-256 见 [results/data_audit.json](results/data_audit.json)。模板隔离比只随机拆近似句更有意义，但仍然是受控合成数据；真实业务还需要去标识化、人工复核和来自真实流量的独立测试。

一条真实使用的样本如下：

```json
{json.dumps(first['messages'],ensure_ascii=False,indent=2)}
```

## 四、基座、Chat Template 和标签

基座：[Qwen/Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base)，固定 revision `{cfg['revision']}`，Apache-2.0。原始基座参数约 {training['base_parameters']:,}，本次没有重新训练 tokenizer，也没有添加新词。

基座权重下载到 Hugging Face 缓存，约 1.2 GB，不重复放入作业目录；来源记录见 [models/base_model.json](models/base_model.json)。适配器依赖这个精确基座版本，不能单独加载成完整模型。

本例明确指定单轮 ChatML 协议，模板保存在 [chat_template.jinja](chat_template.jinja)，不套用带思考段的 Instruct 模板：

```text
<|im_start|>system
任务规则<|im_end|>
<|im_start|>user
售后留言<|im_end|>
<|im_start|>assistant
标准JSON<|endoftext|>
```

`<|im_start|>` 和 `<|im_end|>` 已存在于 Qwen 字表，用于提示里的角色分隔；回答结束使用本 Base 模型原有的 **`<|endoftext|>`（151643）**。推理时输入到 `assistant` 标头结束，不放参考 JSON。训练和推理调用同一个 `prompt_ids()`，保证前缀 token 完全一致。回答单独编码后再拼接结束 token，避免分段 BPE 边界变化导致 prompt 长度与 mask 错位。

**一次实际发现的 EOS 问题：** 初始小实验照用了 Instruct 风格的 im_end，但验证误差一直集中在结束 token。检查原始权重发现该 Base 的 im_start、im_end 向量完全相同；本模型输入/输出权重共享，冻结输出头时，两者的输出分数必然相同，仅改变 LoRA 隐藏表示无法让 im_end 成为唯一最大值。于是改用基座原有 endoftext，从原始基座重新完成下述训练。证据见 [results/eos_diagnostic.json](results/eos_diagnostic.json)。初始实验权重没有混入最终结果，也没有查看测试生成来决定修正。

| 区域 | attention_mask | labels |
| --- | ---: | --- |
| system/user/assistant 标头 | 1 | -100 |
| assistant 回答正文 | 1 | 原始 token ID |
| assistant 结束 token | 1 | 结束 token ID |
| 右侧补齐位置 | 0 | -100 |

**两种 mask 不同：** attention_mask 控制补齐位置是否可见；labels=-100 控制哪些目标不计入损失。因果三角掩码由模型内部处理。不能把问题前缀的 attention_mask 也设成 0，否则模型看不到用户问题。问题虽然没有直接预测损失，仍通过回答的注意力路径影响梯度。

本项目手工计算一次 shift：`logits[:, :-1]` 预测 `labels[:, 1:]`。因此调用 Hugging Face 模型时不传 `labels`，以免模型内部再 shift 一次。若直接用模型自带的 `model(..., labels=labels).loss`，则应交给模型 shift，不要再预先移动标签。

实际首条训练样本的前缀长度为 {mask['prompt_length']}，总长度为 {mask['sequence_length']}。边界附近的真实 token 如下：

| 位置 | token ID | 解码片段 | attention_mask | label |
| --- | ---: | --- | ---: | ---: |
{table_mask}

EOS 必须参与监督；否则模型可能学会正文却不会停。本例 PAD 和 EOS **都是 151643**，因此必须按真实长度遮掉补齐位置，不能把所有相同 ID 的标签都抹掉，也不能把这个 ID 从输出概率分布中删除。真实回答末尾的 EOS 保留监督，补齐出的同 ID 才标成 -100。

## 五、SFT 与普通训练、LoRA 的关系

| 对比项 | 第四周从零语言模型训练 | 本次 SFT |
| --- | --- | --- |
| 初始化 | 随机权重 | 已预训练的基座 |
| 数据 | 普通连续文本 | 输入指令与期望回答 |
| 监督目标 | 下一字符/词 | 仍然是下一 token |
| 损失位置 | 通常所有有效目标位置 | 本例仅回答与结束符 |
| 学习重点 | 语言分布 | 条件回答、字段、格式与停止行为 |
| 参数更新 | 通常全参数 | 本例只更新 LoRA |

**SFT 是训练目标与数据组织方式，LoRA 是参数更新方式，两者不是同义词。** SFT 可以全参数微调，也可以用 LoRA；LoRA 也可以用于别的训练任务。回答段损失是本例选择的常见方案，也不是所有 SFT 研究唯一允许的损失策略。

LoRA 将有效权重写成 `W' = W + (alpha/r)·B·A`。基座 W 冻结，更新 A、B。本例 r={cfg['lora_r']}、alpha={cfg['lora_alpha']}，缩放系数为 {cfg['lora_alpha']/cfg['lora_r']:g}。它在注意力 q/k/v/o 和 MLP gate/up/down 线性层上增加低秩分支，主干仍是原有因果语言模型，不增加业务分类头；需要时可合并回原线性层。

实际更新 **{training['trainable_parameters']:,} 个参数，占含适配器总参数的 {training['trainable_fraction']:.3%}**。代码检查全部可训练参数名字都属于 LoRA，优化器也只接收这些参数，并在每轮抽查冻结权重指纹。冻结基座不意味着开启适配器时绝对不会遗忘；禁用适配器才能恢复原基座行为。

## 六、训练参数与标准习惯

| 项目 | 本次设置 |
| --- | --- |
| 序列上限 | {cfg['max_length']} tokens，超长报错，不静默截掉回答/EOS |
| 样本拼接 packing | 关闭，每个 batch 内样本独立 |
| Micro batch / 累积步数 | {cfg['micro_batch_size']} / {cfg['gradient_accumulation_steps']} |
| 单卡有效 batch | {cfg['micro_batch_size']*cfg['gradient_accumulation_steps']} 条样本 |
| 最大轮数 / 早停耐心 | {cfg['epochs']} / {cfg['patience']} |
| AdamW | lr={cfg['learning_rate']}，betas=(0.9,0.999)，eps=1e-8，weight_decay={cfg['weight_decay']} |
| 学习率计划 | {cfg['warmup_ratio']:.0%} 更新步预热，再余弦衰减到 {cfg['min_learning_rate']} |
| 梯度裁剪 | 累积完成后裁剪到范数 {cfg['gradient_clip']} |
| 精度 | 基座 BF16；可训练 LoRA 和优化器状态 FP32 |
| 显存控制 | 非重入梯度检查点；训练关闭 KV Cache |
| 模型选择 | 验证集回答 token NLL 最低；不使用测试结果选型 |
| 随机性 | 数据与训练固定种子；不同设备/内核仍可能有数值差异 |

**梯度累积的细节：** 每个累积窗口先计算所有 micro batch 的有效回答 token 总数 N，再对每个 micro batch 的交叉熵总和除以 N 后反传。这样得到“这个有效 batch 内每个回答 token 等权”的梯度。不能随意把各个变长 micro batch 的平均 loss 再简单平均，否则短回答和长回答权重会变。尾部不足一个累积窗口时也按实际 N 归一化。学习率每次 optimizer.step 更新，不按每个 micro batch 更新。

训练 teacher forcing：一次输入完整的参考序列，所有回答位置并行预测。因果掩码挡住未来，因此没有偷看下一 token。推理时没有参考答案，必须把生成结果逐个接回去。训练低损失与自由生成正确率不同，不能只看曲线。

实际训练 {training['epochs_run']} 轮、{training['optimizer_updates']} 次参数更新，选择第 **{training['best_epoch']}** 轮。GPU 为 {env['gpu']}，峰值已分配显存约 {env['peak_allocated_GB']:.2f} GiB，本次训练阶段耗时 {training['training_seconds_this_run']:.1f} 秒（不含下载、模型加载和最终生成评估）。

| Epoch | 在线训练回答 NLL | 验证回答 NLL | 验证回答 PPL | 秒数 |
| --- | ---: | ---: | ---: | ---: |
{table_history}

![SFT 学习曲线与生成评估](results/training_curve.png)

训练 NLL 在开启 LoRA Dropout、参数持续更新时统计；验证 NLL 在关闭 Dropout 的固定权重下统计，所以验证值可以低于训练值。

## 七、验证、生成和基座对比

先由验证 NLL 锁定最佳适配器，再重新加载“固定基座 + 最佳适配器”进行测试。Base 对照通过 `disable_adapter()` 关闭同一模型上的适配器，输入协议完全相同。两者都用贪心解码，最多 {cfg['max_new_tokens']} 个新 token；推理使用左补齐、attention_mask、KV Cache，遇到 `<|endoftext|>` 停止。训练则采用右补齐。基座从未更新。

| 模型 | 测试回答 NLL ↓ | 回答 PPL ↓ | 结构合法率 ↑ | 五字段全对率 ↑ | EOS 停止率 ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
{table_metrics}

| 字段 | Base 正确率 | SFT 正确率 |
| --- | ---: | ---: |
{table_fields}

“结构合法”要求输出能直接解析成 JSON 对象，五个键完整且无额外键，枚举与类型正确。没有去代码围栏、补缺失字段或把字符串布尔强制转换为布尔值。“五字段全对”还要求所有值等于参考事实；键顺序和空白不影响 JSON 对象的相等性。非法 JSON 的字段也计错。

NLL 是有效回答 token 的负对数似然总和除以总 token 数，PPL=exp(NLL)，包括结束 token。固定 JSON 键占不少位置，因此很低的回答 PPL 也不保证订单号或缺失字段正确。这不是普通语料上的全序列 PPL，也不能与第四周的字符 PPL 直接比较。

未见商品子集的 SFT 五字段全对率为 {sft['generation']['unseen_products']['exact_match']:.2%}（{sft['generation']['unseen_products']['samples']} 条）。本实验采用合成短文本、有限枚举、单轮对话，测试规模也有限；没有测试真实噪声、长多轮、复杂否定、售后政策判断或通用能力遗忘。这里的结果只能支持“已跑通这一受控任务的 SFT 流程”。

完整结果在 [results/metrics.json](results/metrics.json)，全部原始生成在 [base_test_predictions.jsonl](results/base_test_predictions.jsonl) 和 [sft_test_predictions.jsonl](results/sft_test_predictions.jsonl)。

### 固定示例

下面展示测试集前 4 条；若有错误，另外加入第一个错误案例，不按好坏筛选：

{examples_md}

## 八、自己运行与复现

本机已安装的环境在 `D:/AI课程/.venvs/sft-week05`，项目目录下执行：

```powershell
# 本机可直接激活已准备好的环境
& 'D:\AI课程\.venvs\sft-week05\Scripts\Activate.ps1'

# 一键：固定基座 → 数据 → 测试 → 训练 → 评估 → README
python -B -X utf8 run_all.py

# 只检查数据、标签和损失，不训练
python -B -X utf8 test_sft.py

# 使用已训练适配器预测，或 --base 查看原基座
python -B -X utf8 predict.py --text "订单号是SO587243，商品是保温杯，收到时已经破损。我想换货，请优先处理。"

# 从最近完成的 epoch 续训
python -B -X utf8 run_all.py --resume

# 只重新评估并生成报告
python -B -X utf8 run_all.py --evaluate-only
```

新机器需先安装匹配 GPU 的 PyTorch，再安装 [requirements.txt](requirements.txt)；[requirements-lock.txt](requirements-lock.txt) 记录本次实际版本。首次运行会下载公开基座，之后使用缓存。全参数训练的显存预算不同，不能照搬本例 LoRA 的参数预算。CPU 代码路径用于小规模检查，完整案例建议使用支持 BF16 的 CUDA GPU。

检查点 `checkpoints/last.pt` 保存最近完成 epoch 的 LoRA、优化器、步数、数据/模板指纹和随机状态；恢复时要求配置、训练/验证数据及模板一致，重跑未保存的当前 epoch。训练已经完成时 `train.py --resume` 不会再次更新或覆盖训练记录。它是本机续训状态，已在 .gitignore 中排除；实际提交或部署使用最佳适配器和固定基座，不需上传优化器状态。

本次环境：Python {env['python']}，PyTorch {env['torch']}，Transformers {env['transformers']}，PEFT {env['peft']}。8 项检查覆盖前缀一致性、回答/EOS 标签、补齐掩码、单次 shift、有效 token 归一化、集合隔离、超长拒绝和严格评估。

## 九、复用到自己的 SFT 项目

1. 先定义任务边界和输出协议，收集真实、有授权且经复核的正确示范；不要先盲目加训练轮数。
2. 按真实独立单位拆分，例如用户、订单、原始文档或模板，先拆分再做增强。近重复、冲突标签、缺失输入都要检查。
3. 选能覆盖任务语言、上下文和能力需求的预训练模型，固定模型与 tokenizer 版本；不要把 Instruct 的继续微调误称为从 Base 首次对齐。
4. 打印一条真正编码后的 input_ids、labels 和 EOS，确认“问题可见但不计损失”，确认没有双重 shift、全 -100 或截断后的残缺答案。
5. 先做小批次前向/反传和显存检查，随后按预算训练；记录学习率、有效 token 数、梯度与验证指标。低学习率、短训练及 LoRA 能降低破坏风险，但都不能保证不遗忘。
6. 用验证集选择轮次，再在未参与调整的测试集实际生成。格式指标、业务字段和失败样本比仅看训练 loss 更接近使用效果。
7. 保存可重载产物，并在独立进程里重新加载验证。部署时保持相同模板、基座 revision、结束符和字段规则；不要只保存 LoRA 文件却遗漏基座信息。

参考：[基座模型卡](https://huggingface.co/Qwen/Qwen3-0.6B-Base)、[PEFT LoRA](https://huggingface.co/docs/peft/v0.18.0/package_reference/lora)、[SFT 数据与损失策略](https://huggingface.co/docs/trl/sft_trainer)。本例用可阅读的 PyTorch 循环显式展示掩码、shift 和累积，不依赖 TRL 训练器。
'''
    (ROOT/"README.md").write_text(text,encoding="utf-8")
    print("README and one referenced training chart updated.",flush=True)


if __name__ == "__main__":
    build_readme()
