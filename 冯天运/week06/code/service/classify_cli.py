"""
分流服务（CLI 版）：输入一条标题，输出对接下游的 JSON —— 就是课件 P43 那张「接口契约」的样子。

【这个脚本在干什么】
把「后端选型 → 分类 → 置信度校准 → 队列路由」封装成一个 CLI / 可 import 的函数。
它和 classify_news.py 的区别是：这里有**多后端自动降级 + 温度缩放校准**。

教学重点：
  1. **接口契约比准确率更重要**：下游要的是 {topic, relevant, conf, queue}。
     有 conf 才能做「低置信转人工」，只有 label 下游没法决策。
  2. **服务要能自己选后端**：有 BERT 权重就走 BERT，没有就退回 TF-IDF 线性模型，
     两者都没有再退规则 —— 而且**降级必须显式**（结果里的 model 字段会写清楚是哪条路径）。
  3. **置信度要真**：所以 TF-IDF 这档故意用「带 predict_proba 的逻辑回归」，
     而不是课件对比里用的 LinearSVC（SVC 没有概率，拿 decision_function 硬凑会误导下游）。

返回字段（与课件 P43 完全一致）：
  title     原始标题
  topic     4 组主题（金融 / 产业 / 宏观 / 噪音），给业务看的中文名
  sub_topic 15 类细标签（证券 / 科技 / 教育 …），给统计用
  relevant  是否与投研相关（由主题组派生）
  conf      置信度 0~1；低于阈值时 queue=review
  queue     research / review / drop（枚举，业务方按这个路由，不用自由文本）
  model     哪条路径出的结果：bert-cls-v1 / tfidf-lr-v1 / rule-v1 / fallback
  version   接口版本（出问题能追溯到哪天）

用法：
  python service/classify_cli.py                          # 交互模式（输入标题回车，q 退出）
  python service/classify_cli.py --text "贵州茅台：2023 年年度业绩预告"
  python service/classify_cli.py --file inputs.txt        # 每行一条
  type inputs.txt | python service/classify_cli.py -      # 从管道读
  python service/classify_cli.py --text "..." --json      # 只输出一行 JSON（给脚本/Agent 用）
  python service/classify_cli.py --model tfidf            # 强制指定后端
  python service/classify_cli.py --use-rules              # 打开规则快筛，演示 L5 级联

也被谁调用：第 9 周会把这里的 classify() 包成 Agent 的第一个工具（Function Calling / MCP / CLI 三接法）。
"""

import argparse                        # 命令行参数
import json                           # 输出 JSON
import sys                            # 模块搜索路径 / stderr
import time                           # 测延迟

import numpy as np                    # 数值计算（温度缩放要 numpy）
from pathlib import Path               # 路径处理

CODE_ROOT = Path(__file__).resolve().parent.parent   # code/ 根目录
sys.path.insert(0, str(CODE_ROOT))    # 让 common.* 可导入

from common.labels import LABELS, is_relevant, queue_name, to_group  # noqa: E402

VERSION = "2026-11"                   # 接口版本
BACKEND_NAME = {"bert": "bert-cls-v1", "tfidf": "tfidf-sgd-v1", "rule": "rule-v1", "fallback": "fallback-v0"}
# ↑ 对外暴露的模型名（下游看到的是这个，不关心内部实现）

# 规则快筛：公告标题套路很强，一条正则就能挡掉一批调用（生产里最常见的省钱手段）。
# 默认关闭，保证演示看到的是「模型」的结果；加 --use-rules 才是完整的 L5 级联。
FAST_RULES = [
    ("业绩预告", "财经"), ("年度报告", "财经"), ("季度报告", "财经"),
    ("回购", "证券"), ("减持", "证券"), ("增持", "证券"),
]


class ShuntService:
    """加载一次后端，之后每次调用复用（服务化写法：别在请求里现加载模型）"""

    def __init__(self, model="auto", ckpt=None, conf_threshold=0.8, use_rules=False, verbose=False):
        self.conf_threshold = conf_threshold      # 置信度闸门阈值
        self.use_rules = use_rules                # 是否开规则快筛
        self.verbose = verbose                    # 是否输出 _debug 字段
        self.backend = None            # bert / tfidf / rule / fallback
        self.tok = self.model = None   # BERT 的分词器与模型
        self.vec = self.clf = None     # TF-IDF 的向量器与分类器
        self.temperature = 1.0          # 置信度校准用的温度（1.0 = 不校准）

        self.ckpt_path = Path(ckpt) if ckpt else CODE_ROOT / "outputs" / "bert_ckpt.pt"
        # ---- 后端选择：优先级 BERT → TF-IDF → 规则兜底 ----
        if model in ("auto", "bert"):
            self._try_init_bert()
            if model == "bert" and self.backend is None:
                # 显式点名要 BERT 却加载失败 → 直接报错（不静默降级）
                raise SystemExit(f"指定了 --model bert，但没加载成功：{self.ckpt_path} 不存在或读取失败")
        if self.backend is None and model in ("auto", "tfidf"):
            if model == "auto":
                # 降级要显式：让别人知道这次没用 BERT，以及怎么才能用上
                print("[info] 没有 BERT 权重，改用 TF-IDF 后端（想用 BERT 先跑 bert/train_bert.py）",
                      file=sys.stderr)
            self._init_tfidf()
        if self.backend is None:       # 兜底：只剩规则
            self.backend = "fallback"
            print("[warn] 没有任何可用的模型，进入 fallback（全部判 review）", file=sys.stderr)

    # ---------- 置信度校准：温度缩放 ----------
    @staticmethod
    def _softmax(z):
        """数值稳定的 softmax：先减最大值再 exp，避免溢出"""
        z = z - z.max(axis=1, keepdims=True)   # 减去每行最大值（数学等价，不会溢出）
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def _fit_temperature(self, proba, y):
        """
        在一批「没参与训练」的样本上，找一个 T 让 NLL 最小。
        proba 是模型原始概率；返回 T（<1 表示原概率偏保守，需要放大）。
        """
        P = np.clip(proba, 1e-12, 1.0)          # 防止 log(0)
        logits = np.log(P)                     # 概率 → logit
        rows = np.arange(len(y))               # 每行的下标
        best_T, best_nll = 1.0, float("inf")
        for T in np.arange(0.2, 3.01, 0.05):   # 在 [0.2, 3.0] 上网格搜索温度
            q = self._softmax(logits / T)      # 温度缩放后的概率（T 越大越平滑/保守）
            nll = -float(np.log(q[rows, y] + 1e-12).mean())   # 负对数似然，越小越好
            if nll < best_nll:
                best_nll, best_T = nll, float(T)            # 记住最优温度
        return round(best_T, 2)

    def _calibrate(self, proba):
        """把原始概率按温度缩放后再取最大类（T 由 _fit_temperature 得到）"""
        if self.temperature == 1.0:
            proba = proba.reshape(1, -1) if proba.ndim == 1 else proba
            return proba                        # 不校准：原样返回（注意要统一成二维）
        proba = proba.reshape(1, -1) if proba.ndim == 1 else proba
        return self._softmax(np.log(np.clip(proba, 1e-12, 1.0)) / self.temperature)

    # ---------- 后端 1：BERT 微调（效果最好，需要 ckpt） ----------
    def _try_init_bert(self):
        """尝试加载 BERT；任何一步失败都只是「不启用 BERT」，不影响服务整体"""
        if not self.ckpt_path.exists():
            if self.verbose:
                print(f"[info] 没有 BERT 权重（{self.ckpt_path.name}），跳过", file=sys.stderr)
            return
        try:
            t0 = time.time()
            import torch
            from transformers import AutoTokenizer

            sys.path.insert(0, str(CODE_ROOT / "bert"))    # 让 from train_bert import ... 找得到
            from train_bert import BertClassifier            # 复用训练时的模型定义
            from common.model_paths import bert_path          # 本地预训练模型目录

            ckpt = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)  # 读权重
            self.tok = AutoTokenizer.from_pretrained(str(bert_path()))                 # 分词器
            self.model = BertClassifier(bert_path(), num_labels=len(LABELS), pool=ckpt.get("pool", "cls"))
            self.model.load_state_dict(ckpt["state_dict"])   # 灌参数
            self.model.eval()                                # 推理模式
            self.backend = "bert"
            self._calibrate_bert()                           # 装完模型先校准置信度
            print(f"[init] BERT 后端就绪（{time.time() - t0:.1f}s，温度 T={self.temperature}）",
                  file=sys.stderr)
        except Exception as e:          # 权重坏了也不能让服务整体挂掉
            print(f"[warn] BERT 加载失败（{e}），尝试降级", file=sys.stderr)
            self.backend, self.model = None, None

    def _calibrate_bert(self, n=200):
        """用干净验证集里的一小批样本拟合温度（这批样本没参与训练；200 条足够，再多只是拖慢启动）"""
        import torch

        from common.data import clean_val, load_splits

        train, val = load_splits()          # 原始切分
        clean, _ = clean_val(train, val)    # 去重（这批没参与训练）
        clean = clean[:n]                   # 只取前 n 条（控制启动耗时）
        probs, ys = [], []
        with torch.no_grad():               # 不建计算图
            for i in range(0, len(clean), 64):    # 每 64 条一批
                batch = clean[i:i + 64]
                enc = self.tok([d["sentence"] for d in batch], truncation=True, max_length=64,
                               padding="longest", return_tensors="pt")
                logits = self.model(enc["input_ids"], enc["attention_mask"])
                probs.append(torch.softmax(logits.float(), -1).numpy())   # 概率矩阵
                ys.extend(d["label"] for d in batch)
        self.temperature = self._fit_temperature(np.vstack(probs), np.array(ys))

    # ---------- 后端 2：TF-IDF + 逻辑回归（20 秒内可训练，永远可用） ----------
    def _init_tfidf(self):
        """训练 TF-IDF + SGD 逻辑回归，并在留出的校准集上拟合温度"""
        t0 = time.time()
        import warnings

        from sklearn.exceptions import ConvergenceWarning
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import SGDClassifier

        from common.data import load_splits

        train, _ = load_splits()
        # 切出 10% 专门做置信度校准（这些样本不参与训练，否则温度会被拉向过拟合的自信）
        rng = np.random.RandomState(42)               # 固定种子，保证每次启动切分一致
        idx = rng.permutation(len(train))             # 打乱下标
        n_cal = max(1000, len(train) // 10)           # 校准集大小（至少 1000 条）
        cal_idx, fit_idx = idx[:n_cal], idx[n_cal:]   # 前段=校准，后段=训练
        fit_set = [train[i] for i in fit_idx]
        cal_set = [train[i] for i in cal_idx]

        self.vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 2),
                                   max_features=300000, min_df=2, sublinear_tf=True)
        # analyzer="char"：中文按字切 n-gram；min_df=2：只出现 1 次的特征丢掉；sublinear_tf=True：TF 取 log
        X = self.vec.fit_transform([d["sentence"] for d in fit_set])
        # 为什么用 SGD(log_loss) 而不是 LogisticRegression：
        #   111k 维 × 5.3 万条，lbfgs 求解要 80 秒才收敛；SGD 十秒内到同一水平
        #   （实测 macro-F1：SGD 0.518 / LinearSVC 0.510），服务启动才等得起。
        # class_weight=balanced 与课件里的「类别加权」同一个思路，否则长尾类几乎不会出现。
        self.clf = SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=30,
                                 class_weight="balanced", random_state=42)
        with warnings.catch_warnings():      # 迭代上限的 warning 不影响结果，别刷屏
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.clf.fit(X, [d["label"] for d in fit_set])

        # 温度缩放：原始 SGD 概率明显偏保守，不校准的话阈值 0.8 几乎筛不出东西
        Xc = self.vec.transform([d["sentence"] for d in cal_set])
        self.temperature = self._fit_temperature(self.clf.predict_proba(Xc),
                                                np.array([d["label"] for d in cal_set]))
        self.backend = "tfidf"
        print(f"[init] TF-IDF 后端就绪（{X.shape[1]} 维特征，{X.shape[0]} 条样本，{time.time() - t0:.1f}s，温度 T={self.temperature}）",
              file=sys.stderr)

    # ---------- 单条预测 ----------
    def _rule_hit(self, text):
        """规则快筛：命中关键词直接返回类别 id"""
        for kw, label in FAST_RULES:
            if kw in text:
                return LABELS.index(label)
        return None

    def _predict(self, title):
        """返回 (label_id, conf, backend)"""
        if self.use_rules:
            idx = self._rule_hit(title)
            if idx is not None:
                # 规则命中的「置信度」是人为设定的，不是模型输出 —— 这点必须在代码里写清楚
                return idx, 0.99, "rule"

        if self.backend == "bert":
            import torch
            enc = self.tok(title, truncation=True, max_length=64, padding="max_length", return_tensors="pt")
            with torch.no_grad():
                logits = self.model(enc["input_ids"], enc["attention_mask"])
            prob = self._calibrate(torch.softmax(logits.float(), -1)[0].numpy())[0]  # 校准后的概率
            idx = int(prob.argmax())
            return idx, float(prob[idx]), "bert"

        if self.backend == "tfidf":
            raw = self.clf.predict_proba(self.vec.transform([title]))[0]   # 原始概率
            prob = self._calibrate(raw)[0]          # 温度缩放后的置信度
            idx = int(prob.argmax())
            return idx, float(prob[idx]), "tfidf"

        return 0, 0.0, "fallback"                   # 都没有 → 全判为 0 类且置信度 0（→ review）

    def classify(self, title):
        """
        对外接口：一条标题 → 一个 dict（字段与课件 P43 的接口契约一致）。
        下游拿到 conf 和 queue 就能直接做路由，不需要理解模型内部。
        """
        title = title.strip()
        t0 = time.time()
        idx, conf, used = self._predict(title)      # 走完整个后端链路
        out = {
            "title": title,
            "topic": to_group(idx),                  # 15 类 → 4 组主题
            "sub_topic": LABELS[idx],                # 15 类细标签
            "relevant": is_relevant(idx),            # 是否投研相关
            "conf": round(conf, 4),                 # 置信度（已校准）
            "queue": queue_name(idx, conf, self.conf_threshold),   # 队列路由
            "model": BACKEND_NAME[used],             # 哪条路径出的结果
            "version": VERSION,
        }
        if self.verbose:
            out["_debug"] = {"backend": used, "label_id": idx,
                             "latency_ms": round((time.time() - t0) * 1000, 1),
                             "threshold": self.conf_threshold}
        return out


# ---------- 便捷函数：给别的脚本/Agent 直接 import ----------
_DEFAULT = None     # 全局单例（第一次调用时创建，之后复用，避免每次都加载模型）


def classify(title, **kwargs):
    """from service.classify_cli import classify; classify("贵州茅台：2023 年年度业绩预告")"""
    global _DEFAULT
    if _DEFAULT is None or kwargs:      # 首次调用，或传了参数 → 新建实例
        _DEFAULT = ShuntService(**kwargs)
    return _DEFAULT.classify(title)     # 复用实例（模型只加载一次）


def run_oneshot(svc, texts, as_json=False):
    """批量跑一批标题并输出结果"""
    for t in texts:
        if not t.strip():
            continue                    # 跳过空行
        res = svc.classify(t)
        if as_json:
            print(json.dumps(res, ensure_ascii=False))            # 紧凑：一条一行
        else:
            print(json.dumps(res, ensure_ascii=False, indent=2))  # 缩进：便于人读


def run_repl(svc):
    """交互模式：输入一条看一条"""
    print(f"分流服务已就绪：后端={BACKEND_NAME[svc.backend]}，置信度阈值={svc.conf_threshold}，接口版本={VERSION}")
    print("输入一条标题回车看结果；q / exit 退出。")
    while True:
        try:
            title = input("\n标题> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见")
            break
        if title.lower() in ("q", "quit", "exit"):
            print("再见")
            break
        if not title:
            continue
        print(json.dumps(svc.classify(title), ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser(description="分流服务 CLI（输入标题 → 返回 P43 契约的 JSON）")
    ap.add_argument("--text", help="单条标题")
    ap.add_argument("--file", help="每行一条标题的文件；传 - 表示从标准输入读")
    ap.add_argument("--model", default="auto", choices=["auto", "bert", "tfidf"], help="指定后端")
    ap.add_argument("--ckpt", default=None, help="BERT 权重路径（默认 outputs/bert_ckpt.pt）")
    ap.add_argument("--conf", type=float, default=0.8, help="置信度阈值，低于它进 review")
    ap.add_argument("--use-rules", action="store_true", help="打开规则快筛（演示 L5 级联）")
    ap.add_argument("--json", action="store_true", help="紧凑输出：一条一行 JSON")
    ap.add_argument("--verbose", action="store_true", help="额外输出 _debug 字段（后端/延迟）")
    args = ap.parse_args()

    svc = ShuntService(args.model, args.ckpt, args.conf, args.use_rules, args.verbose)

    if args.text:
        run_oneshot(svc, [args.text], args.json)
    elif args.file:
        if args.file == "-":
            run_oneshot(svc, sys.stdin.read().splitlines(), args.json)
        else:
            run_oneshot(svc, Path(args.file).read_text(encoding="utf-8").splitlines(), args.json)
    elif not sys.stdin.isatty():       # 管道输入但没给 -：也当作 stdin
        run_oneshot(svc, sys.stdin.read().splitlines(), args.json)
    else:
        run_repl(svc)


if __name__ == "__main__":
    main()
