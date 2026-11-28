"""
方法 B 的 prompt 模板、目标输出构造、模型输出解析与 span 回填。

一条样本长这样（chat 格式）：
  system   : 你是一个中文命名实体识别助手。
  user     : 从句子中抽取命名实体……句子：浙商银行企业信贷部叶老桂博士……
  assistant: {"公司":["浙商银行"],"人名":["叶老桂"]}

输出约定：JSON 对象，键 = 中文类型名，值 = 实体文本列表（按首次出现顺序、去重）；没有的类型不写。
只让模型输出**文本**，位置由 to_entities() 回原句查找 —— 生成式模型数下标不可靠。
"""
import json
import re

from shared.data import LABEL_ZH, LABELS, ZH2LABEL

SYSTEM = "你是一个中文命名实体识别助手。"
TYPES_ZH = "、".join(LABEL_ZH[k] for k in LABELS)
INSTRUCTION = (
    f"从句子中抽取命名实体，类型只能是：{TYPES_ZH}。\n"
    "只输出一个 JSON 对象：键为实体类型，值为该类型的实体原文列表（按出现顺序，去重）；没有实体的类型不要输出。\n"
    "句子：{text}"
)


def build_messages(text):
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": INSTRUCTION.replace("{text}", text)}]


def build_target(entities):
    """金标 entities → 紧凑 JSON 字符串（类型按首个实体出现顺序排列）"""
    out = {}
    for e in sorted(entities, key=lambda e: e["start"]):
        lst = out.setdefault(LABEL_ZH[e["label"]], [])
        if e["text"] not in lst:
            lst.append(e["text"])
    return json.dumps(out, ensure_ascii=False, separators=(",", ":"))


# ─────────────────────────── 解析 ───────────────────────────

_PAIR_RE = re.compile(r'"([^"]+)"\s*:\s*\[([^\]]*)\]?')
_STR_RE = re.compile(r'"([^"]*)"')


def parse_output(s):
    """
    返回 (dict[类型 -> list[str]], status)
      status: "ok"       直接 json.loads 成功
              "repaired" 截掉代码块/补括号/正则兜底后才解析出来
              "fail"     什么都没解析出来（按空预测处理，不删行）
    """
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
    i = s.find("{")
    if i < 0:
        return {}, ("ok" if s in ("", "{}") else "fail")
    body = s[i:]
    j = body.rfind("}")
    if j >= 0:
        try:
            d = json.loads(body[:j + 1])
            if isinstance(d, dict):
                return _clean(d), "ok"
        except json.JSONDecodeError:
            pass
    # 兜底 1：正则逐对抓取 "类型":[ "a","b" ]（容忍截断/尾逗号）
    d = {}
    for k, v in _PAIR_RE.findall(body):
        d.setdefault(k, []).extend(_STR_RE.findall(v))
    if d:
        return _clean(d), "repaired"
    # 兜底 2：Python 风格 / 值不带引号，如 {'人名': 陈尔真, '职位': ['总经理']}（zero-shot 常见）
    for k, v in _LOOSE_RE.findall(body):
        vals = _LOOSE_STR_RE.findall(v) if v.lstrip().startswith("[") else [v]
        vals = [x.strip().strip("'\"") for x in vals]
        d.setdefault(k.strip(), []).extend(x for x in vals if x)
    return (_clean(d), "repaired") if d else ({}, "fail")


_LOOSE_RE = re.compile(r"['\"]?([^'\",{}:\[\]]+)['\"]?\s*:\s*(\[[^\]]*\]?|[^,}]+)")
_LOOSE_STR_RE = re.compile(r"['\"]?([^'\",\[\]]+)['\"]?")


def _clean(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, str):
            v = [v]
        if not isinstance(v, list):
            continue
        out[str(k).strip()] = [str(x).strip() for x in v if isinstance(x, (str, int, float)) and str(x).strip()]
    return out


def to_entities(text, d):
    """
    {中文类型: [实体文本]} → entities（带 start/end），并统计工程指标。
    回填规则：
      - 类型名不在 10 类里 → 丢弃，计 unknown_type
      - 实体文本不在原句里 → 丢弃，计 hallucinated（幻觉）
      - 文本在原句出现多次 → 每次出现都标注（与金标「同一文本处处标注」的多数情况一致）
      - 先长后短分配，已被占用的位置不重复标注（避免「中国」和「中国银行」重叠）
    """
    cands, n_unknown, n_hallu = [], 0, 0
    for k, vals in d.items():
        lab = ZH2LABEL.get(k) or (k if k in LABELS else None)
        if lab is None:
            n_unknown += len(vals)
            continue
        for v in dict.fromkeys(vals):
            if v in text:
                cands.append((lab, v))
            else:
                n_hallu += 1
    used = [False] * len(text)
    ents = []
    for lab, v in sorted(cands, key=lambda x: -len(x[1])):
        for m in re.finditer(re.escape(v), text):
            s, e = m.start(), m.end()
            if not any(used[s:e]):
                ents.append({"label": lab, "text": v, "start": s, "end": e})
                for p in range(s, e):
                    used[p] = True
    ents.sort(key=lambda e: e["start"])
    return ents, {"unknown_type": n_unknown, "hallucinated": n_hallu, "n_returned": len(cands) + n_hallu + n_unknown}
