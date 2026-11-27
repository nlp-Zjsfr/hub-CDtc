"""数据准备、正确性检查、SFT、基座对比评估和报告的一键入口。"""
import argparse
import unittest


def main():
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--evaluate-only", action="store_true")
    modes.add_argument("--report-only", action="store_true")
    modes.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.evaluate_only and not args.report_only:
        from download_model import download
        download()
        if not args.resume:
            from prepare_data import prepare
            prepare()
        checks = unittest.defaultTestLoader.loadTestsFromName("test_sft")
        if not unittest.TextTestRunner(verbosity=1).run(checks).wasSuccessful():
            raise RuntimeError("SFT 数据或损失检查未通过")
        from train import train
        train(resume=args.resume)
    if not args.report_only:
        from evaluate import evaluate
        evaluate()
    from build_readme import build_readme
    build_readme()


if __name__ == "__main__":
    main()
