"""
一键「换预训练模型」实验：备份 ckpt → 切模型 → 训练 → 评测 → 自动恢复原 ckpt

【这个脚本在干什么】
改一个环境变量 → 换掉预训练模型 → 跑训练 → 跑评测 → 把原权重恢复回去。
这是作业 B 的「手动版」；作业 B 的正式交付是同目录下的 换模型对照实验.py
（那个会串行跑多个模型并自动合并结果）。

用法（在作业目录下执行）：
  python 切换模型实验.py                  # 冒烟 5 步，验证整条链路
  python 切换模型实验.py --full           # 完整 3 epoch（CPU 上很慢，建议有 GPU 再跑）
  python 切换模型实验.py --model chinese-roberta-wwm-ext --pool mean
"""
import argparse, os, shutil, subprocess, sys   # 参数 / 环境变量 / 文件复制 / 起子进程 / 退出码
from pathlib import Path                        # 路径处理

HERE = Path(__file__).resolve().parent         # 作业目录
CODE = HERE / "code"                            # 代码目录（子进程的工作目录）
OUT = CODE / "outputs"                          # 权重/结果目录
ORIG_CKPT = OUT / "bert_ckpt.pt"                # 默认权重（bert-base-chinese 训练出来的）


def run(cmd, env, timeout=None):
    """起一个子进程跑命令：打印命令 → 执行 → 打印输出 → 返回退出码"""
    print("+", " ".join(str(c) for c in cmd), flush=True)     # 先把命令打出来，便于复现
    r = subprocess.run([str(c) for c in cmd], cwd=str(CODE), env=env, capture_output=True,
                       text=True, encoding="utf-8", errors="replace", timeout=timeout)
    # cwd=CODE：让脚本里的相对路径（outputs/...）都相对于 code/，与直接运行一致
    # env=env：把 WEEK6_BERT_DIR 传给子进程 —— ★这就是「换模型」真正生效的地方
    print(r.stdout[-2500:])                      # 只看最后 2500 字符（训练日志很长）
    if r.returncode != 0:
        print("!! 失败：", r.stderr[-2500:], file=sys.stderr)
    return r.returncode                         # 0 = 成功


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="chinese-roberta-wwm-ext")   # 要换成的预训练模型目录名
    ap.add_argument("--pool", default="cls")                        # 池化方式（对照组要保持一致）
    ap.add_argument("--epochs", type=int, default=3)                # 完整模式下的轮数
    ap.add_argument("--n_eval", type=int, default=500)              # 评测抽样条数
    ap.add_argument("--full", action="store_true")                 # 不给就是冒烟模式
    args = ap.parse_args()

    env = dict(os.environ)                       # 复制当前环境（保留 PATH、PYTHONPATH 等）
    env["WEEK6_BERT_DIR"] = args.model           # ★★核心：告诉子进程用哪个预训练模型
    os.environ["WEEK6_BERT_DIR"] = args.model    # 同步到本进程，路径预览才与子进程一致
    # 注意：model_paths.py 在「模块导入时」读这个变量，所以必须在启动子进程前设好

    # 打印本次实际加载的预训练模型路径（换模型是否生效，一眼可见）
    sys.path.insert(0, str(CODE))
    try:
        from common.model_paths import bert_path
        p = bert_path()                          # 解析出本地模型目录
        print("本次预训练模型:", args.model, "→", p)
        print("  权重文件:", sorted(x.name for x in p.iterdir()))   # 确认目录里真有权重
    except Exception as e:  # noqa: BLE001
        print("模型路径解析失败:", e)

    backup = None
    if ORIG_CKPT.exists():
        backup = OUT / ("bert_ckpt_backup_" + args.model + ".pt")
        shutil.copy2(ORIG_CKPT, backup)          # 备份原权重（否则新模型训练会覆盖它）
        print("已备份原 ckpt →", backup.name)

    try:
        cmd = [sys.executable, "bert/train_bert.py", "--pool", args.pool]
        if args.full:
            cmd += ["--epochs", str(args.epochs)]       # 完整训练
        else:
            cmd += ["--epochs", "1", "--max_steps", "5", "--max_eval", "100", "--no_save"]
            # ↑ 冒烟：只跑 5 步、评 100 条、不落盘（几分钟内确认整条链路没断）
        if run(cmd, env) != 0:
            return                                  # 训练失败就不往下走了
        if args.full:
            new_ckpt = OUT / ("bert_ckpt_" + args.model + ".pt")
            shutil.move(str(ORIG_CKPT), str(new_ckpt))    # 新权重改名保存，别覆盖 bert_ckpt.pt
            run([sys.executable, "bert/predict_bert.py", "--ckpt", str(new_ckpt),
                 "--n", str(args.n_eval), "--out", "outputs/bert_pred_" + args.model + ".json"], env)
        else:
            print("冒烟模式：不落盘、不评测，只验证链路")
    finally:
        # ★无论成功失败都要恢复原权重（finally：异常路径也会执行）
        if backup and ORIG_CKPT.exists():
            shutil.copy2(backup, ORIG_CKPT)
            print("已恢复原 ckpt（bert-base-chinese）")
            backup.unlink()  # 备份已用完即回收，避免多占 390 MB
            print("已回收备份文件")
        print("提示：WEEK6_BERT_DIR 只在本次进程内生效，默认模型仍是 bert-base-chinese")


if __name__ == "__main__":
    main()
