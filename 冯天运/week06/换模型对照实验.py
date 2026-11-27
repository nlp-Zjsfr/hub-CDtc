"""
换预训练模型 · 对照实验（作业 B 的主交付）
================================================================
【这个脚本在干什么】
  只改一个变量——预训练模型（backbone），其余全部固定，然后比较两个模型的效果。
  这是"对照实验"的标准做法：控制变量法。

【为什么必须控制这 4 项】
  1) 同数据：同一份 train.json / val.json
  2) 同步数：--max_steps N（受本机 CPU 限制只能做等步数对照，做不了等 epoch 全量）
  3) 同超参：batch / lr / 池化 / 类别加权 / 损失函数全部用默认值
  4) 同评测集：去重干净集 500 条（与 code/outputs/summary.md 同口径，seed=42）
  只要有一项没对齐，"roberta 比 bert 好 3 个点"就可能是步数多、评测集不同造成的假象。

【它是怎么换模型的（关键）】
  不改任何模型代码，只设一个环境变量 WEEK6_BERT_DIR=<模型名>；
  子进程里的 common/model_paths.py 读到这个变量，就会去加载对应的权重目录。

【产物】
  code/outputs/ckpt_<model>_s<N>.pt          每个模型自己的权重（不覆盖原 bert_ckpt.pt）
  code/outputs/bert_pred_<model>_s<N>.json    逐条预测 + 指标
  code/outputs/swap_<model>.log               训练/评测完整日志
  outputs/model_swap_experiment.json          两个模型的对比结果汇总

【用法（在作业目录下执行）】
  python 换模型对照实验.py --steps 50 --models bert-base-chinese
  python 换模型对照实验.py --steps 50 --models chinese-roberta-wwm-ext
  # 两次跑完，结果会自动合并进同一份 model_swap_experiment.json
  # 也可以一次跑两个（默认并行）：--steps 50
"""
# argparse：解析命令行参数（--steps / --models / --serial ...）
import argparse
# json：读写汇总结果 json
import json
# os：读写环境变量（这里用来给子进程传 WEEK6_BERT_DIR）
import os
# subprocess：调用另外两个脚本（train_bert.py / predict_bert.py）
import subprocess
# sys：拿到当前 python 解释器路径（sys.executable），保证用同一个环境
import sys
# threading：让两个模型并行跑（各自开一个线程）
import threading
# time：计时
import time
# pathlib：拼路径
from pathlib import Path

HERE = Path(__file__).resolve().parent      # 本脚本所在目录 = 作业/
CODE = HERE / "code"                        # 课程代码目录（训练/评测脚本都在这）
OUT = CODE / "outputs"                     # 权重/预测/日志的输出目录

DEFAULT_MODELS = ["bert-base-chinese", "chinese-roberta-wwm-ext"]  # 要对比的两个模型
EVAL_N = 500                                # 评测集条数（去重干净集，seed=42，与 summary.md 同口径）


def run(cmd, env, log_path, timeout=20000):
    """
    执行一条子进程命令，把 stdout/stderr 全部写进日志文件。
    为什么不用 subprocess 的 capture_output 直接打印？因为训练日志很长，
    写文件既能看进度，又不会把终端刷爆（失败时日志还在，方便定位）。
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)   # 确保日志目录存在
    t0 = time.time()                                     # 记录开始时间，用于算耗时
    with open(log_path, "w", encoding="utf-8") as lf:     # "w" = 覆盖写（每次重跑日志都是干净的）
        lf.write("+ " + " ".join(str(c) for c in cmd) + "\n")   # 先把命令本身记进日志，便于复现
        lf.flush()                                              # 立刻落盘，防止进程被杀后丢命令
        r = subprocess.run([str(c) for c in cmd], cwd=str(CODE), env=env,
                           stdout=lf, stderr=subprocess.STDOUT, timeout=timeout)
        # cwd=CODE：子命令（bert/train_bert.py）用相对路径 import code 里的模块
        # env=env：把 WEEK6_BERT_DIR 传进去，子进程才知道该加载哪个模型
        # stderr=STDOUT：报错信息也进同一个日志，不丢栈
        # timeout：防止子进程挂死把父进程拖住（20000 秒 ≈ 5.5 小时）
    return {"cmd": [str(c) for c in cmd], "rc": r.returncode,   # rc=0 才算成功
            "seconds": round(time.time() - t0, 1), "log": str(log_path)}


def one_model(model, steps, env, results, lock, log_every=50):
    """
    跑完"一个模型"的完整链路：训练 → 落盘权重 → 在干净集上评测 → 收集指标。
    两个模型各调一次这个函数（默认在不同线程里并行）。
    """
    tag = model.replace(".", "_")          # 模型名里的 "." 换成 "_"，用来拼文件名（safe-ish）
    e = dict(env)                          # 复制一份环境变量，别污染父进程的 os.environ
    e["WEEK6_BERT_DIR"] = model            # ★核心：告诉子进程"这次用哪个预训练模型"
    e["PYTHONUTF8"] = "1"                   # 让子进程用 UTF-8 处理中文（否则 Windows GBK 下日志乱码）
    e["PYTHONIOENCODING"] = "utf-8"
    ckpt = "ckpt_%s_s%d.pt" % (tag, steps)  # 权重文件名，如 ckpt_bert-base-chinese_s50.pt
    pred = "outputs/bert_pred_%s_s%d.json" % (tag, steps)   # 预测结果文件名
    rec = {"model": model, "tag": tag}      # rec = 这个模型的"记录"，最后汇总进 json

    # ---- 第 1 步：训练（等价于 python bert/train_bert.py --epochs 1 --max_steps 50 ...）
    rec["train"] = run([sys.executable, "bert/train_bert.py", "--epochs", "1",
                        "--max_steps", str(steps), "--max_eval", str(EVAL_N),
                        "--log_every", str(log_every), "--ckpt_out", ckpt],
                       e, OUT / ("swap_%s.log" % tag))
    # --epochs 1 + --max_steps N：一个 epoch 但只跑 N 步（受 CPU 限制的等步数对照）
    # --max_eval：训练过程中在验证集前 500 条上看一下（只作过程参考，不作最终结论）
    # --ckpt_out：权重存成自己的名字，避免覆盖掉原来的 bert_ckpt.pt
    if rec["train"]["rc"] == 0:
        # ---- 第 2 步：评测（训练成功才评测）
        rec["predict"] = run([sys.executable, "bert/predict_bert.py", "--ckpt",
                              "outputs/" + ckpt, "--n", str(EVAL_N), "--out", pred],
                             e, OUT / ("swap_%s_pred.log" % tag))
        # --ckpt 用刚训出来的权重；--n 500 表示在去重干净集里抽 500 条评（与 summary.md 同口径）
        # --out 指定结果 json 的落盘位置
    else:
        rec["predict"] = {"rc": -1, "skipped": "训练失败"}   # 训练失败就跳过评测，别用脏数据下结论

    # ---- 第 3 步：把评测指标从 json 里读出来，整理成结构化记录
    pj = CODE / pred                        # 预测结果文件（注意在 code/outputs 下）
    if pj.exists():
        d = json.loads(pj.read_text(encoding="utf-8"))   # 读回 json
        r = d.get("report", {})              # report 里是统一口径的指标
        rec["eval"] = {
            "n_eval": d.get("meta", {}).get("n_eval"),   # 实际评了多少条
            "acc": r.get("acc"),                          # 准确率
            "macro_f1": r.get("macro_f1"),                # Macro-F1（主指标，每类等权）
            "acc_incl_unparsed": r.get("acc_incl_unparsed"),  # 含解析失败的口径
            "ece": r.get("ece"),                          # 置信度校准误差
            "ms_per_sample": r.get("ms_per_sample"),      # 单条推理延迟
            "unparsed": r.get("unparsed"),                # 解析失败条数
        }
    with lock:                              # 线程安全：两个线程同时往 results 里塞结果要加锁
        results.append(rec)


def main():
    """命令行入口：解析参数 → 跑模型 → 合并结果 → 写汇总 json → 打印对比表"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300, help="每个模型训练步数（两模型必须相同）")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)  # nargs="+" 接受空格分隔的多个模型名
    ap.add_argument("--serial", action="store_true", help="串行跑（默认并行）")
    ap.add_argument("--log_every", type=int, default=50, help="每多少步打印一次 loss")
    ap.add_argument("--out", default="outputs/model_swap_experiment.json")  # 汇总结果落盘位置
    args = ap.parse_args()                 # 把命令行解析成 args 对象

    env = dict(os.environ)                 # 以当前环境为基础（后面会加上 WEEK6_BERT_DIR）
    results, lock = [], threading.Lock()   # results 收集每个模型的记录；lock 保护并发写入
    print("对照实验：%d 步 / 模型，评测集=去重干净集 %d 条，%s"
          % (args.steps, EVAL_N, "串行" if args.serial else "并行"), flush=True)
    # flush=True：立刻输出，不等缓冲区（后台跑时能实时看到进度）

    if args.serial:
        # 串行：一个模型跑完再跑下一个（省内存、更稳，但总时长翻倍）
        for m in args.models:
            one_model(m, args.steps, env, results, lock, args.log_every)
    else:
        # 并行：每个模型开一个线程（线程内部是 subprocess，真正的算力还是各进程占 CPU）
        ts = [threading.Thread(target=one_model,
                               args=(m, args.steps, env, results, lock, args.log_every))
              for m in args.models]
        for t in ts:
            t.start()                      # 启动所有线程
        for t in ts:
            t.join()                       # 等所有线程结束（join 前主线程在这里阻塞）

    order = {m: i for i, m in enumerate(args.models)}   # 模型名 → 显示顺序
    results.sort(key=lambda r: order.get(r["model"], 99))  # 让表格顺序稳定（不随线程完成先后变）

    # 分次跑（例如先跑一个模型、再跑另一个）时，把结果**合并**进已有汇总；
    # 口径不一致（步数不同）则不混用旧结果。
    out = HERE / args.out
    merged = {r["model"]: r for r in results}          # 本次结果按模型名建索引
    if out.exists():                                   # 已有汇总（上次跑过别的模型）
        try:
            old = json.loads(out.read_text(encoding="utf-8"))
        except Exception:
            old = {}                                   # 读坏了就当没有
        if (old.get("design") or {}).get("steps") == args.steps:
            # 步数一致 → 口径一致 → 可以合并（setdefault：本次的优先，旧的不覆盖）
            for r in old.get("results", []):
                merged.setdefault(r["model"], r)
        else:
            print("[warn] 已有汇总的步数与本次不同，不合并旧结果（避免口径混用）", flush=True)

    final_results = sorted(merged.values(), key=lambda r: list(merged).index(r["model"]))
    payload = {
        "design": {
            # design 段记录"实验条件"，审稿/复现时先看它，确认口径一致
            "steps": args.steps, "batch_size": 64, "lr": 2e-5, "pool": "cls",
            "class_weight": True, "loss": "CE",
            "eval_set": "去重干净集 %d 条（seed=42，与 summary.md 同口径）" % EVAL_N,
            "note": "受本机 CPU 限制（bert-base 1 epoch 实测 15978 s），采用等步数对照，"
                    "绝对值低于完整 3 epoch 训练，但两模型之间可比。",
        },
        "results": final_results,         # 每个模型的训练/评测记录
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    # ensure_ascii=False：中文以原文写入 json（人可读）；indent=2：缩进，便于人工检查

    # 最后在终端打印一张对比表（和报告里的表一致）
    print("\n| 模型 | 训练步数 | 干净集 acc | Macro-F1 | ECE | ms/条 |")
    print("|---|---|---|---|---|---|")
    for r in results:                    # 注意这里打的是"本次"的 results（不是 merged）
        e = r.get("eval", {})
        print("| %s | %d | %s | %s | %s | %s |" % (
            r["model"], args.steps, e.get("acc"), e.get("macro_f1"),
            e.get("ece"), e.get("ms_per_sample")))
    print("\n→ 汇总写入", out)


if __name__ == "__main__":     # 只有"直接运行本文件"时才执行 main()，被 import 时不执行
    main()
