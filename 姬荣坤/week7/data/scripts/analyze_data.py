import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import platform

# ================= 配置画图字体 =================
if platform.system() == "Windows":
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # Windows
elif platform.system() == "Darwin":
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']          # Mac
else:
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']       # Linux
plt.rcParams['axes.unicode_minus'] = False


def analyze_extract_data(script_dir):
    # 定位到 homework/data 目录
    data_dir = os.path.dirname(script_dir)
    processed_dir = os.path.join(data_dir, "processed")
    train_path = os.path.join(processed_dir, "train_extract.jsonl")
    
    if not os.path.exists(train_path):
        print(f"❌ 错误: 找不到数据文件 {train_path}")
        return

    # 数据结构初始化
    total_lines = 0
    global_text_lengths = []
    
    # 统计字典
    type_counts = Counter()
    type_triggers = defaultdict(set)
    type_roles = defaultdict(set)
    type_text_lengths = defaultdict(list)
    sample_roles_record = []
    
    print(f"正在读取并分析: {train_path} ...\n")
    
    with open(train_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            total_lines += 1
            try:
                item = json.loads(line)
                text = item.get("text", "")
                text_len = len(text)
                global_text_lengths.append(text_len)
                
                event = item.get("event")
                if event and isinstance(event, dict):
                    evt_type = event.get("type", "Unknown")
                    type_counts[evt_type] += 1
                    type_text_lengths[evt_type].append(text_len)
                    
                    trigger = event.get("trigger")
                    if trigger and isinstance(trigger, dict) and "text" in trigger:
                        type_triggers[evt_type].add(trigger["text"])
                        
                    args = event.get("args", [])
                    sample_roles = set()
                    if isinstance(args, list):
                        for arg in args:
                            role = arg.get("role")
                            if role:
                                type_roles[evt_type].add(role)
                                sample_roles.add(role)
                    
                    sample_roles_record.append((evt_type, len(sample_roles)))
                                
            except json.JSONDecodeError:
                continue

    # ================= 1. 全局文本长度统计 =================
    print("="*60)
    print("📊 全局输入文本 (text) 特征统计")
    print("="*60)
    if global_text_lengths:
        avg_len = np.mean(global_text_lengths)
        median_len = np.median(global_text_lengths)
        print(f"总样本数     : {total_lines}")
        print(f"平均长度     : {avg_len:.2f} 字符")
        print(f"中位数长度   : {median_len:.0f} 字符")
        print(f"最短 / 最长  : {np.min(global_text_lengths)} / {np.max(global_text_lengths)} 字符")
        print(f"P90 / P99    : {np.percentile(global_text_lengths, 90):.0f} / {np.percentile(global_text_lengths, 99):.0f} 字符")
        
        # 图1：全局长度分布图
        plt.figure(figsize=(10, 5))
        plt.hist(global_text_lengths, bins=60, color='skyblue', edgecolor='black', alpha=0.7)
        plt.axvline(avg_len, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {avg_len:.1f}')
        plt.axvline(median_len, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_len:.1f}')
        plt.title('全局输入文本长度 (Text Length) 分布直方图', fontsize=14)
        plt.xlabel('字符数', fontsize=12)
        plt.ylabel('样本数量', fontsize=12)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(processed_dir, "text_length_distribution.png"), dpi=300)
        plt.close()

    # ================= 2. 事件类型统计与完整度 =================
    print("\n" + "="*60)
    print(f"📊 事件类型 (Event Type) 统计 (共 {len(type_counts)} 种)")
    print("="*60)
    
    full_role_counts = {k: len(v) for k, v in type_roles.items()}
    complete_samples = Counter()
    for evt_type, role_cnt in sample_roles_record:
        if role_cnt == full_role_counts[evt_type]:
            complete_samples[evt_type] += 1

    export_data = []
    sorted_types = type_counts.most_common()
    
    # 准备画图数据
    plot_categories = []
    plot_counts = []
    plot_avg_lengths = []

    for evt_type, count in sorted_types:
        percentage = (count / total_lines) * 100
        triggers = type_triggers[evt_type]
        roles = type_roles[evt_type]
        lengths = type_text_lengths[evt_type]
        avg_len = np.mean(lengths)
        
        full_cnt = full_role_counts.get(evt_type, 0)
        comp_cnt = complete_samples.get(evt_type, 0)
        comp_rate = (comp_cnt / count * 100) if count > 0 else 0
        
        # ============ 严格对齐用户要求的输出格式 ============
        print(f"🔹 类别: 【{evt_type}】 | 数量: {count} ({percentage:.2f}%) | 平均长度: {avg_len:.1f}")
        print(f"   - 触发词 (Triggers): {', '.join(sorted(triggers)) if triggers else '无'}")
        print(f"   - 论元角色 (Roles) : {', '.join(sorted(roles)) if roles else '无'}")
        print(f"   - 全量角色数 : {full_cnt} 个")
        print(f"   - 包含所有角色的样本 : {comp_cnt} 个 (占比: {comp_rate:.2f}%)")
        # ====================================================
        
        export_data.append({
            "事件类型 (Type)": evt_type,
            "样本数量": count,
            "占比 (%)": round(percentage, 2),
            "平均文本长度": round(avg_len, 2),
            "全量角色数": full_cnt,
            "全角色样本数": comp_cnt,
            "全角色占比 (%)": round(comp_rate, 2),
            "包含触发词 (Triggers)": ", ".join(sorted(triggers)),
            "包含论元角色 (Roles)": ", ".join(sorted(roles))
        })

        # 收集画图数据
        plot_categories.append(evt_type)
        plot_counts.append(count)
        plot_avg_lengths.append(avg_len)

    # ================= 3. 生成新增的 2 个柱状图 =================
    
    # 图2：各类别样本数量分布
    plt.figure(figsize=(12, 6))
    bars1 = plt.bar(plot_categories, plot_counts, color='#2ca02c', alpha=0.8)
    plt.title('各事件类别样本数量分布', fontsize=15)
    plt.xlabel('事件类型', fontsize=12)
    plt.ylabel('样本数', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    # 在柱子上标数字
    for bar in bars1:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 10, int(yval), ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(processed_dir, "type_counts_distribution.png"), dpi=300)
    plt.close()

    # 图3：各类别平均文本长度
    plt.figure(figsize=(12, 6))
    bars2 = plt.bar(plot_categories, plot_avg_lengths, color='#ff7f0e', alpha=0.8)
    plt.title('各事件类别的平均文本长度', fontsize=15)
    plt.xlabel('事件类型', fontsize=12)
    plt.ylabel('平均长度 (字符)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    # 在柱子上标数字
    for bar in bars2:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f"{yval:.1f}", ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(processed_dir, "type_avg_length_distribution.png"), dpi=300)
    plt.close()

    print("\n✅ 3 张图片已生成并保存至 processed 目录：")
    print("   1. text_length_distribution.png (全局长度直方图)")
    print("   2. type_counts_distribution.png (类别样本数柱状图)")
    print("   3. type_avg_length_distribution.png (类别平均长度柱状图)")

    # ================= 4. 导出为 Excel =================
    if export_data:
        df = pd.DataFrame(export_data)
        output_excel = os.path.join(processed_dir, "event_schema_analysis.xlsx")
        df.to_excel(output_excel, index=False, engine='openpyxl')
        print(f"\n✅ 详细的事件 Schema 与特征统计已导出至: {output_excel}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    analyze_extract_data(script_dir)