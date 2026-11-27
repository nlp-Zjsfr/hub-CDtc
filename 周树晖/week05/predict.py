import argparse

import torch

from common import config, load_tokenizer
from evaluate import generate_rows
from modeling import inference_model, base_model
from prepare_data import SYSTEM


def main():
    parser = argparse.ArgumentParser(description="售后留言结构化：输出原始模型结果")
    parser.add_argument("--text", required=True)
    parser.add_argument("--base", action="store_true", help="关闭适配器，查看同一基座的结果")
    args = parser.parse_args()
    cfg = config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    tok = load_tokenizer(cfg)
    model = base_model(cfg, device).eval() if args.base else inference_model(cfg, device)
    row = {"id": "interactive", "messages": [{"role":"system", "content":SYSTEM}, {"role":"user", "content":args.text}]}
    result = generate_rows(model, tok, [row], cfg)
    # 交互模式没有参考答案，仅返回模型原始正文，不生成虚假的评分。
    print(result[0]["prediction"])


if __name__ == "__main__":
    main()
