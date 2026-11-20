"""
SemEval-2020 Task 11 宣传手法标签（全项目共用）

与 Hugging Face `SemEvalWorkshop/sem_eval_2020_task_11` 的 ClassLabel 顺序、字符串一致。
标签是语料里的金标，不改名、不合并。
"""

import re

LABELS = [
    "Appeal_to_Authority",
    "Appeal_to_fear-prejudice",
    "Bandwagon,Reductio_ad_hitlerum",
    "Black-and-White_Fallacy",
    "Causal_Oversimplification",
    "Doubt",
    "Exaggeration,Minimisation",
    "Flag-Waving",
    "Loaded_Language",
    "Name_Calling,Labeling",
    "Repetition",
    "Slogans",
    "Thought-terminating_Cliches",
    "Whataboutism,Straw_Men,Red_Herring",
]
LABEL2ID = {name: i for i, name in enumerate(LABELS)}
ID2LABEL = {i: name for name, i in LABEL2ID.items()}

# 任务说明里的简短定义，供零样本 / LoRA 的 system prompt 使用
BOUNDARY_NOTES = {
    "Appeal_to_Authority": "引用权威人物或机构来支撑主张，而不给出可核对的证据。",
    "Appeal_to_fear-prejudice": "唤起恐惧或群体成见，让读者接受某个结论。",
    "Bandwagon,Reductio_ad_hitlerum": "声称人人都如此，或把对象比作希特勒/纳粹来否定它。",
    "Black-and-White_Fallacy": "把问题说成只有两种选择，否认中间可能。",
    "Causal_Oversimplification": "把复杂事件归结为单一原因。",
    "Doubt": "质疑对方的信誉或事实，但不给出具体反证。",
    "Exaggeration,Minimisation": "把事实的规模、程度夸大或缩小。",
    "Flag-Waving": "诉诸国家、民族或群体认同来争取认同。",
    "Loaded_Language": "用带强烈感情色彩的词影响判断。",
    "Name_Calling,Labeling": "用标签给对象定性，代替论证。",
    "Repetition": "反复传达同一信息，使其显得更可信。",
    "Slogans": "用简短口号代替论证。",
    "Thought-terminating_Cliches": "用套话打断进一步讨论。",
    "Whataboutism,Straw_Men,Red_Herring": "用「那别人呢」、歪曲对方观点或岔开话题来回避。",
}

# 分流队列：TC 金标全是宣传手法，没有「无宣传」类
#   flag_bias —— 高置信，进复核
#   review    —— 低置信或无法解析，转人工
CONF_THRESHOLD = 0.75


def queue_name(label_id: int, conf: float, conf_threshold: float = CONF_THRESHOLD) -> str:
    if label_id < 0 or label_id >= len(LABELS) or conf < conf_threshold:
        return "review"
    return "flag_bias"


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def parse_label(text: str) -> int:
    """把模型输出解析成标签 id。对不上返回 -1。"""
    if not text or not text.strip():
        return -1
    line = text.strip().splitlines()[0].strip().strip("\"'` ")
    if line in LABEL2ID:
        return LABEL2ID[line]
    norm = _norm(line)
    for name in LABELS:
        if _norm(name) == norm:
            return LABEL2ID[name]
    hits = [name for name in LABELS if _norm(name) and _norm(name) in norm]
    if not hits:
        return -1
    return LABEL2ID[max(hits, key=lambda name: len(_norm(name)))]
