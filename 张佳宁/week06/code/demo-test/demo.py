import sys
sys.path.insert(0, r"D:\1aSTUDY\AI_Agent\hub-CDtc\张佳宁\week06\code")
from service.classify_news import ShuntService

svc = ShuntService(conf_threshold=0.8)          # 模型只加载一次，之后复用
for title in ["宁德时代拟回购不超过 50 亿元公司股份", "国足世预赛客场 1:3 失利","马斯克参加APEC峰会","天津市采购大量煤炭","北京常驻人口创历年新高"]:
    print(svc.classify(title))