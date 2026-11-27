"""合成售后留言抽取案例；隔离模板和订单号，测试集加入未见商品。"""
import json
import random
from collections import Counter

from common import ROOT, config, write_json, write_jsonl, sha256

SYSTEM = ("你是售后工单信息抽取助手。仅输出一个JSON对象，不要解释或Markdown。字段固定为order_id、product、intent、problem、urgent。"
          "订单号和商品缺失时填null。intent只能为return、replace、refund、query_shipping；problem只能为damaged、wrong_size、wrong_color、missing_part、late_delivery、not_needed。"
          "urgent必须为布尔值。只记录用户表达，不判断售后资格，不编造信息。")
PRODUCTS = ["保温杯", "运动鞋", "蓝牙耳机", "手机壳", "台灯", "背包", "咖啡杯", "充电器", "围巾", "雨伞", "床单", "收纳盒"]
NEW_PRODUCTS = ["机械键盘", "电动牙刷", "瑜伽垫", "行李箱", "显示器支架", "无线鼠标"]
SCENARIOS = [
    ("damaged", ["收到时已经破损", "到货后发现坏了", "拆开就发现有损坏"], ["return", "replace", "refund"]),
    ("wrong_size", ["尺码不合适", "尺寸不适合我", "买到的尺码不对"], ["return", "replace"]),
    ("wrong_color", ["发来的颜色不对", "颜色和我选的不一致", "收到的颜色有误"], ["return", "replace"]),
    ("missing_part", ["包装里少了配件", "配件没有发齐", "有配件漏发了"], ["replace", "refund"]),
    ("late_delivery", ["超过约定时间还没有送到", "配送已经延误", "过了预计日期还没到货"], ["query_shipping", "refund"]),
    ("not_needed", ["现在已经不需要了", "临时改变主意不想要了", "这件东西用不上了"], ["return", "refund"]),
]
REQUESTS = {"return": ["我想退货", "请帮我申请退货", "我要把它退回去"],
            "replace": ["我想换货", "请给我换一件", "我要申请更换商品"],
            "refund": ["我要求退款", "请把款项退还给我", "我想申请退款"],
            "query_shipping": ["请帮我查一下物流", "我想查询配送进度", "麻烦查询包裹到哪里了"]}
TEMPLATES = {
    "train": ["{order}，{product}，{issue}。{request}。{urgent}", "你好，{product}出了问题：{issue}；{order}。{request}，{urgent}",
              "售后留言：{order}。商品情况：{product}，{issue}。我的诉求是：{request}。{urgent}",
              "{request}。因为{issue}，{product}。供核对：{order}。{urgent}",
              "麻烦处理一下，{order}；{product}；{issue}；{request}。{urgent}",
              "这边是{order}，{product}。反馈{issue}，因此{request}。{urgent}"],
    "val": ["请登记我的情况。{order}，{product}。碰到的问题是{issue}，所以{request}。备注：{urgent}",
            "用户诉求记录：{request}；核对信息：{order}，{product}；原因：{issue}；处理速度：{urgent}"],
    "test": ["帮我整理这条售后申请：{product}，{order}。主要是{issue}。我的要求：{request}。另外，{urgent}",
             "先说诉求，{request}。再补充背景，{issue}；{product}；{order}。最后说明，{urgent}",
             "客服您好！{order} / {product}。实际情况：{issue}。希望这样解决：{request}。时效要求：{urgent}"]}


def prepare():
    cfg = config()
    all_prompts, all_orders, stats = set(), set(), {"source": "deterministic synthetic teaching data, not real customer traffic", "seed": cfg["seed"], "splits": {}}
    for split, n, low in (("train", cfg["train_size"], 100000), ("val", cfg["val_size"], 200000), ("test", cfg["test_size"], 300000)):
        rng = random.Random(cfg["seed"] + low)
        rows = []
        while len(rows) < n:
            i = len(rows)
            problem, issues, intents = rng.choice(SCENARIOS)
            intent = rng.choice(intents)
            new_product = split == "test" and i % 2 == 0
            product = rng.choice(NEW_PRODUCTS if new_product else PRODUCTS)
            order = f"SO{rng.randint(100000, 999999):06d}"
            while order in all_orders:
                order = f"SO{rng.randint(100000, 999999):06d}"
            missing_order, missing_product = rng.random() < .12, rng.random() < .12
            urgent = rng.choice([True, False])
            phrase = rng.choice(["请优先处理", "麻烦加急", "这单需要尽快处理"] if urgent else ["按正常流程处理就好", "不需要加急", "这单不用优先处理"])
            template_id = rng.randrange(len(TEMPLATES[split]))
            text = TEMPLATES[split][template_id].format(
                order="我没有提供订单号" if missing_order else "订单号是" + order,
                product="我没有说明商品名称" if missing_product else "商品是" + product,
                issue=rng.choice(issues), request=rng.choice(REQUESTS[intent]), urgent=phrase)
            if text in all_prompts:
                continue
            all_prompts.add(text)
            all_orders.add(order)
            target = {"order_id": None if missing_order else order, "product": None if missing_product else product,
                      "intent": intent, "problem": problem, "urgent": urgent}
            answer = json.dumps(target, ensure_ascii=False, separators=(",", ":"))
            rows.append({"id": f"{split}-{i:04d}", "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": text}, {"role": "assistant", "content": answer}],
                         "target": target, "meta": {"template": f"{split}-{template_id}", "unseen_product": new_product and not missing_product}})
        path = ROOT / "data" / f"{split}.jsonl"
        write_jsonl(path, rows)
        stats["splits"][split] = {"samples": n, "sha256": sha256(path), "intents": dict(Counter(r["target"]["intent"] for r in rows)),
                                   "missing_order": sum(r["target"]["order_id"] is None for r in rows),
                                   "missing_product": sum(r["target"]["product"] is None for r in rows),
                                   "unseen_products": sum(r["meta"]["unseen_product"] for r in rows)}
    stats["cross_split_exact_overlap"] = 0
    stats["split_policy"] = "disjoint sentence templates and order IDs; ID digit distributions shared; test includes unseen product names"
    write_json(ROOT / "results" / "data_audit.json", stats)
    print("Prepared synthetic tickets:", {k: s["samples"] for k, s in stats["splits"].items()}, flush=True)


if __name__ == "__main__":
    prepare()
