# -*- coding: utf-8 -*-
"""
爬取正北方网首页，提取新闻标题和链接，并打印前 10 条。
仅依赖 requests；Python 3.12（conda 环境 py312）。

用法：
    python northnews_spider.py            # 爬取首页，打印前 10 条新闻
    python northnews_spider.py --selftest # 只跑内置离线测试样例（不联网）
"""
import re
import sys

import requests

HOME_URL = "https://www.northnews.cn/"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}  # 部分站点会拒绝无 UA 的请求

# 新闻正文页 URL 形态：/news/2026/1015/2528083.html 或短链 /p/2528042.html
ARTICLE_RE = re.compile(
    r"^https?://[\w.-]*northnews\.cn/(?:[a-z]+/)*\d{4}/\d{4}/\d+\.html$"
    r"|^https?://[\w.-]*northnews\.cn/p/\d+\.html$",
    re.I,
)
# 匹配 <a href="...">标题</a>
TAG_RE = re.compile(r"<a\b[^>]*href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>", re.S | re.I)


def fetch(url):
    """下载页面并按 UTF-8 解码（该站响应头未声明 charset，requests 会误判为 ISO-8859-1）。"""
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()  # 4xx/5xx 直接抛 HTTPError
    try:
        return resp.content.decode("utf-8")
    except UnicodeDecodeError:  # 万一页面改编码，退回到自动探测
        return resp.content.decode(resp.apparent_encoding or "gbk", errors="ignore")


def extract_news(html, base_url, strict=True):
    """从 HTML 中提取 [(标题, 链接)]，按出现顺序去重。strict=False 时放宽为站内任意 .html 链接。"""
    news, seen = [], set()
    for href, inner in TAG_RE.findall(html):
        title = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", inner)).strip()  # 去标签、压空白
        href = href.strip()
        url = requests.compat.urljoin(base_url, href)  # 把相对链接补成绝对链接
        # 过滤：标题太短、脚本/锚点链接、重复链接
        if len(title) < 6 or href.startswith(("javascript:", "#", "mailto:")) or url in seen:
            continue
        if strict:
            if not ARTICLE_RE.match(url):  # 只保留新闻正文页，排除栏目/专题/外链
                continue
        elif "northnews.cn" not in url or not url.endswith((".html", ".shtml")):
            continue
        seen.add(url)
        news.append((title, url))
    return news


def selftest():
    """内置测试样例：用一段本地 HTML 校验解析逻辑（不联网）。"""
    sample = """
    <html><body>
      <a href="/news/2026/1015/2528083.html"><span>【好评中国】西安网评：从于都河畔再出发</span></a>
      <a href="https://www.northnews.cn/p/2528042.html">【央视快评】为世界发展繁荣持续注入新动力</a>
      <a href="/news/2026/1015/2528083.html">重复链接应被去掉</a>
      <a href="/news/guonei/index.html">内蒙古</a>
      <a href="https://weibo.com/northnews">正北方网官方微博</a>
      <a href="javascript:scroll(0,0)">回到顶部</a>
    </body></html>
    """
    got = extract_news(sample, HOME_URL)
    assert [t for t, _ in got] == [
        "【好评中国】西安网评：从于都河畔再出发",
        "【央视快评】为世界发展繁荣持续注入新动力",
    ], got
    assert got[0][1] == "https://www.northnews.cn/news/2026/1015/2528083.html", got[0][1]
    print("自测通过：相对链接补全、去重、栏目/外链过滤均正确。")


def main():
    try:
        html = fetch(HOME_URL)
    except requests.exceptions.Timeout:
        print("错误：请求超时，请检查网络后重试。")
        return
    except requests.exceptions.RequestException as e:  # 连接失败、DNS、HTTP 4xx/5xx 等
        print(f"错误：请求失败 -> {e}")
        return

    news = extract_news(html, HOME_URL)
    if not news:  # 页面结构可能变了，放宽规则再试一次
        news = extract_news(html, HOME_URL, strict=False)
    if not news:
        print("未提取到新闻，页面结构可能已变化。")
        return

    print(f"共提取到 {len(news)} 条新闻，前 10 条如下：\n")
    for i, (title, url) in enumerate(news[:10], 1):
        print(f"{i:>2}. {title}\n    {url}")


if __name__ == "__main__":
    # Windows 控制台默认 GBK，防止打印中文标题时报 UnicodeEncodeError
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if "--selftest" in sys.argv:
        selftest()
    else:
        selftest()  # 每次运行先跑一遍离线自测，确认解析逻辑正常
        print("-" * 60)
        main()
