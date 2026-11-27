import json
from pathlib import Path

# 自动定位到 homework/outputs/bert_model
root = Path(__file__).resolve().parent.parent / "outputs" / "bert_model"

for folder in ["best", "last"]:
    p = root / folder / "config.json"
    if p.exists():
        try:
            # 尝试用 GBK 读取
            data = json.loads(p.read_text(encoding="gbk"))
            # 重新用 UTF-8 写入
            p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"✅ 修复成功: {p}")
        except Exception as e:
            print(f"❌ 修复失败 {p}: {e}")