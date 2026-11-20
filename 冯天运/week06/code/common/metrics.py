"""
统一评测指标（全项目共用）

【这个文件在干什么】
四路方法（规则 / TF-IDF / BERT / LLM）都用**同一个函数**算指标，
保证「0.53」在四个脚本里是同一个意思 —— 横向对比的前提。

教学重点：
  1. 分类任务只看 Accuracy 会被长尾类骗：头部类预测对就够高，尾部类全错也看不出来。
     所以主指标用 Macro-F1（每类等权），辅以 per-class recall 和混淆矩阵。
  2. Macro-F1 和 Accuracy 的差距本身就是诊断信号：
     两者差得多 → 尾部类有问题；差不多 → 各类表现均匀。
  3. 置信度校准（ECE）用来回答「模型说 0.9 时，实际命中率是不是 0.9」——
     这个数字决定了阈值能不能用来做「低置信转人工」。

为什么单独成文件：
  四路方法必须用**同一个函数**算指标，避免「这脚本用 weighted F1、那脚本用 binary F1」
  导致数字不可比（这类错在真实项目里非常常见）。
"""

import numpy as np      # 数组运算（混淆矩阵、ECE 分桶都要向量化）


def classification_report(y_true, y_pred, num_classes=15):
    """
    返回统一口径的指标字典。

    注意 f1_score 必须显式写 average：sklearn 默认是 binary（二分类），
    直接 f1_score(y, p) 在多分类上会报错或给出误导结果。
    """
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

    y_true = np.asarray(y_true)             # 转 numpy（list 转数组，后续才能用布尔掩码）
    y_pred = np.asarray(y_pred)
    valid = y_pred >= 0                      # LLM 可能解析失败，用 -1 标记
    unparsed = int((~valid).sum())           # 解析失败的条数

    report = {
        "n": int(len(y_true)),               # 总条数
        "unparsed": unparsed,                # 解析失败条数
        "unparsed_rate": round(unparsed / len(y_true), 4) if len(y_true) else 0.0,  # 失败率
    }
    if valid.sum() == 0:
        return report                        # 全解析失败：没有别的指标可算，直接返回

    yt, yp = y_true[valid], y_pred[valid]    # 只在解析成功的样本上算「条件指标」
    report["acc"] = round(float(accuracy_score(yt, yp)), 4)   # 条件准确率（剔除失败样本）
    # 解析失败的样本按「答错」计入总准确率：生成式方法常靠丢弃失败样本来虚高准确率，
    # 对外的口径必须把所有样本都算进分母（张榜/简历用这个数）
    n_correct = int((yt == yp).sum())
    report["acc_incl_unparsed"] = round(n_correct / len(y_true), 4) if len(y_true) else 0.0
    # ↑ acc_incl_unparsed = (成功且答对的条数) / 全部条数 ← 对外口径，LLM 必须用这个
    report["macro_f1"] = round(float(f1_score(yt, yp, average="macro")), 4)    # 主指标：每类等权
    report["weighted_f1"] = round(float(f1_score(yt, yp, average="weighted")), 4)  # 辅助：按样本量加权

    # 每类 recall：定位「哪一类被吃掉了」，比总分布更有用
    per_class = {}
    labels = list(range(num_classes))        # 固定 15 类（防止某类没出现导致类数变化）
    recalls = []
    for c in labels:
        mask = yt == c                       # 该类的所有样本
        r = float((yp[mask] == c).mean()) if mask.sum() else None   # None = 该类在真值里没有样本
        recalls.append(r)
        per_class[c] = None if r is None else round(r, 4)
    report["per_class_recall"] = per_class   # {类id: recall}
    report["confusion_matrix"] = confusion_matrix(yt, yp, labels=labels).tolist()  # 15×15，行=真值列=预测
    return report


def expected_calibration_error(y_true, probs, n_bins=10):
    """
    ECE：把预测概率分桶，比较「平均置信度」和「实际准确率」的差。
    值越小越可信；>0.1 基本不能拿置信度当阈值用。

    probs: shape [N, C] 的 softmax 概率
    """
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    conf = probs.max(axis=1)                 # 每条预测的置信度 = 最大类别概率
    pred = probs.argmax(axis=1)              # 预测类别
    correct = (pred == y_true).astype(float)  # 是否正确（0/1）

    bins = np.linspace(0.0, 1.0, n_bins + 1) # 把 [0,1] 均分成 10 个区间
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi)    # 落在这个置信度桶里的样本
        if mask.sum() == 0:
            continue                         # 空桶跳过
        gap = abs(correct[mask].mean() - conf[mask].mean())   # 这个桶里「实际准确率 vs 自称置信度」
        ece += gap * mask.sum() / len(conf)  # 按桶内样本占比加权求和
    return round(float(ece), 4)


def pred_distribution(preds, num_classes=15):
    """预测分布：LLM zero-shot 塌缩到少数几类时，一眼就能看出来"""
    counts = {}
    for p in preds:
        counts[int(p)] = counts.get(int(p), 0) + 1   # 统计每个类别被预测了多少次
    used = sum(1 for k in counts if k >= 0)           # 用到的类别数（-1 = 解析失败，不算）
    return {"used_classes": used, "counts": counts}


def to_markdown_table(results: dict):
    """
    把一个 {方法名: 指标字典} 的字典渲染成 Markdown 表，方便直接贴进文档。
    results 的每一项至少要有 acc / macro_f1。
    """
    lines = ["| 方法 | 准确率 | Macro-F1 | 解析失败 |", "|---|---|---|---|"]
    for name, r in results.items():
        acc = r.get("acc", "-")              # 缺字段用 "-" 占位（不报错）
        f1 = r.get("macro_f1", "-")
        up = r.get("unparsed", 0)
        lines.append(f"| {name} | {acc} | {f1} | {up} |")
    return "\n".join(lines)                   # 返回一个 Markdown 字符串
