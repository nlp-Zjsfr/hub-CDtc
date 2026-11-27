"""
守望进程：等换模型对照实验跑完后，自动把结果回填进《对比实验报告.md》。

【这个脚本在干什么】
轮询检查「实验是否已跑完」，一旦跑完就自动执行 回填对比结果.py，
省得人盯着终端等十几分钟再手动回填。

判定完成的依据（全部满足才算完成）：
  1) 两个模型的预测文件都在：code/outputs/bert_pred_<model>_s300.json
  2) 汇总文件存在：outputs/model_swap_experiment.json
  3) 汇总文件里的两个模型都有 eval 指标

然后执行：python 回填对比结果.py
日志：outputs/auto_fill_watcher.log
"""
import json                       # 读汇总 JSON
import subprocess                 # 起子进程跑回填脚本
import sys                        # 用 sys.executable 复用当前解释器
import time                       # 轮询与超时
from pathlib import Path          # 路径处理

HERE = Path(__file__).resolve().parent        # 作业目录
CODE_OUT = HERE / "code" / "outputs"           # 预测结果目录
SUMMARY = HERE / "outputs" / "model_swap_experiment.json"   # 汇总结果
FILL = HERE / "回填对比结果.py"               # 要自动执行的回填脚本
LOG = HERE / "outputs" / "auto_fill_watcher.log"            # 本进程日志
MODELS = ["bert-base-chinese", "chinese-roberta-wwm-ext"]   # 必须跑齐的两个模型
DEADLINE = time.time() + 6 * 3600             # 最长等 6 小时（CPU 全量训练可能很慢）


def log(msg):
    """同时打到终端和日志文件（追加模式）"""
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def ready():
    """按汇总文件里记录的实际步数判定，避免写死步数"""
    if not SUMMARY.exists():
        return False                            # 汇总文件还没生成
    try:
        d = json.loads(SUMMARY.read_text(encoding="utf-8"))
    except Exception:
        return False                            # 正在写入中，解析失败就再等等
    steps = (d.get("design") or {}).get("steps")   # 实际跑的步数（从汇总里读，不写死）
    if not steps:
        return False
    got = {r.get("model") for r in d.get("results", [])}
    if not set(MODELS) <= got:                   # 两个模型都还没跑完
        return False
    for r in d.get("results", []):
        tag = r["model"].replace(".", "_")       # 模型名里的点换成下划线（文件名规范）
        if not (CODE_OUT / ("bert_pred_%s_s%d.json" % (tag, steps))).exists():
            return False                        # 这个模型的预测文件还没落盘
        if not (r.get("eval") or {}).get("macro_f1"):
            return False                        # 没有 F1 → 评测还没写回
    return True                                 # 全部满足 → 可以回填


def main():
    log("watcher start, waiting for swap experiment to finish ...")
    while time.time() < DEADLINE:               # 没到截止时间就一直等
        if ready():
            log("result ready -> run filler")
            r = subprocess.run([sys.executable, str(FILL)], cwd=str(HERE),
                               capture_output=True, text=True, encoding="utf-8", errors="replace")
            # cwd=HERE：回填脚本用相对路径找同目录的 outputs/ 与报告
            log("filler rc=%s" % r.returncode)
            log((r.stdout or "").strip())
            if r.stderr:
                log("stderr: " + r.stderr.strip()[:800])
            log("done")
            return
        time.sleep(60)                          # 每分钟看一次
    log("timeout: experiment not finished within 6h")


if __name__ == "__main__":
    main()
