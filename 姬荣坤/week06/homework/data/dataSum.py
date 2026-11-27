import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# ==========================================
# 1. 设置中文字体 (适配 Windows 环境)
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 2. 加载数据
# ==========================================
print("正在加载数据集...")
dataset = load_dataset("sookiemonster/asrs-narratives-rebalance")
train_data = dataset["train"]

# ==========================================
# 3. 统计文本长度分布
# ==========================================
texts = train_data["text"]
lengths = [len(t) for t in texts]

avg_len = np.mean(lengths)
p50 = np.percentile(lengths, 50)
p90 = np.percentile(lengths, 90)
p95 = np.percentile(lengths, 95)
p99 = np.percentile(lengths, 99)
max_len = np.max(lengths)

print("\n=== 📝 文本长度统计 ===")
print(f"平均长度: {avg_len:.2f} 字符")
print(f"P50 (中位数): {p50:.0f} 字符")
print(f"P90: {p90:.0f} 字符")
print(f"P95: {p95:.0f} 字符")
print(f"P99: {p99:.0f} 字符")
print(f"最大长度: {max_len} 字符")

# ==========================================
# 4. 统计类别分布
# ==========================================
labels = train_data["label"]
label_names = train_data.features["label"].names
label_counts = Counter(labels)

# 按数量降序排序
sorted_counts = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
sorted_labels = [label_names[k] for k, v in sorted_counts]
sorted_values = [v for k, v in sorted_counts]

max_count = max(sorted_values)
min_count = min(sorted_values)
ratio = max_count / min_count if min_count > 0 else 0

print("\n=== 📊 类别分布统计 ===")
print(f"最多类别 ({sorted_labels[0]}): {max_count} 条")
print(f"最少类别 ({sorted_labels[-1]}): {min_count} 条")
print(f"最大/最小比例: {ratio:.1f}x")
if ratio > 5:
    print("结论: 类别分布存在明显的【长尾现象】")
else:
    print("结论: 类别分布【相对均匀】")

# ==========================================
# 5. 绘制图表
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# --- 图 1：文本长度分布直方图 ---
axes[0].hist(lengths, bins=80, color='#7eb0d5', edgecolor='white', alpha=0.9)
axes[0].set_title(f'长度分布：中位数 {p50:.0f} 字，P99 在 {p99:.0f} 字以内', fontsize=16, fontweight='bold', pad=15)
axes[0].set_xlabel('文本字符长度', fontsize=14)
axes[0].set_ylabel('样本数', fontsize=14)

y_max = axes[0].get_ylim()[1]

# 绘制参考线 (P50, P90, P99)
axes[0].axvline(p50, color='#1f4e79', linestyle='--', linewidth=2, label=f'P50={p50:.0f}')
axes[0].axvline(p90, color='#d3721d', linestyle=':', linewidth=2, label=f'P90={p90:.0f}')
axes[0].axvline(p99, color='#d3721d', linestyle='--', linewidth=2, label=f'P99={p99:.0f}')

# 添加文本标注
axes[0].text(p50 + max_len*0.01, y_max*0.90, f'P50={p50:.0f}', color='#1f4e79', fontsize=12, fontweight='bold')
axes[0].text(p90 + max_len*0.01, y_max*0.80, f'P90={p90:.0f}', color='#d3721d', fontsize=12)
axes[0].text(p99 + max_len*0.01, y_max*0.70, f'P99={p99:.0f}', color='#d3721d', fontsize=12)

# 标注常见的截断红线 (例如 BERT 的 512)
if max_len > 512:
    axes[0].axvline(512, color='red', linestyle='-', linewidth=2)
    axes[0].text(512 - max_len*0.02, y_max*0.90, 'max_length=512\n(覆盖大多数)', color='red', fontsize=12, ha='right')
else:
    axes[0].axvline(max_len, color='red', linestyle='-', linewidth=2)
    axes[0].text(max_len - max_len*0.02, y_max*0.90, f'max={max_len}\n(覆盖100%)', color='red', fontsize=12, ha='right')

axes[0].legend(loc='center right', fontsize=12)
axes[0].grid(axis='y', linestyle='--', alpha=0.6)

# --- 图 2：类别分布水平条形图 ---
y_pos = np.arange(len(sorted_labels))
axes[1].barh(y_pos, sorted_values, color='#2b7bba', height=0.7)

# 设置Y轴标签并反转，让样本最多的类别在最上方
axes[1].set_yticks(y_pos)
axes[1].set_yticklabels(sorted_labels, fontsize=11)
axes[1].invert_yaxis()

axes[1].set_title(f'类别分布：最大 / 最小 = {ratio:.0f}x (长尾明显)', fontsize=16, fontweight='bold', pad=15)
axes[1].set_xlabel('训练集样本数', fontsize=14)

# 在柱子末端标注具体数值和占比
total_samples = sum(sorted_values)
for i, v in enumerate(sorted_values):
    axes[1].text(v + total_samples*0.01, i, f'{v} ({v/total_samples*100:.1f}%)', va='center', fontsize=10)

axes[1].grid(axis='x', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()