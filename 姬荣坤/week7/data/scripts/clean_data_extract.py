import os
import json

def clean_dev_extract_data(script_dir):
    # 定位到 homework/data 目录 (因为脚本在 homework/data/scripts 里)
    data_dir = os.path.dirname(script_dir) 
    processed_dir = os.path.join(data_dir, "processed")
    
    train_path = os.path.join(processed_dir, "train_extract.jsonl")
    dev_path = os.path.join(processed_dir, "dev_extract.jsonl")
    clean_dev_path = os.path.join(processed_dir, "dev_extract_clean.jsonl")
    
    print(f"数据目录: {processed_dir}")
    
    # ================= 1. 读取训练集文本 =================
    train_texts = set()
    if not os.path.exists(train_path):
         print(f"❌ 错误: 找不到训练集文件 {train_path}")
         return
         
    print(f"正在读取训练集: {train_path} ...")
    with open(train_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line)
                    # 将完整的 text 字段加入集合，利用 set 去重及高速查找
                    train_texts.add(item.get("text", ""))
                except json.JSONDecodeError:
                    continue
                    
    print(f"✅ 训练集读取完毕，共提取到 {len(train_texts)} 条不重复的文本。")
    
    # ================= 2. 读取验证集并过滤 =================
    if not os.path.exists(dev_path):
         print(f"❌ 错误: 找不到验证集文件 {dev_path}")
         return
         
    total_dev = 0
    kept_dev = 0
    removed_dev = 0
    
    print(f"正在处理验证集并进行去重: {dev_path} ...")
    # 同时打开输入和输出文件
    with open(dev_path, 'r', encoding='utf-8') as f_in, \
         open(clean_dev_path, 'w', encoding='utf-8') as f_out:
             
        for line in f_in:
            if not line.strip():
                continue
            
            total_dev += 1
            try:
                item = json.loads(line)
                text = item.get("text", "")
                
                # 如果该文本在训练集中出现过，直接丢弃
                if text in train_texts:
                    removed_dev += 1
                    continue
                
                # 保留不重复的数据，原样写入新文件
                f_out.write(line if line.endswith('\n') else line + '\n')
                kept_dev += 1
            except json.JSONDecodeError:
                print(f"⚠️ 警告: 发现无法解析的 JSON 行，已跳过。")
                continue
                
    # ================= 3. 打印最终统计结果 =================
    print("\n" + "="*40)
    print("📊 清洗完成统计报告 (Extract 数据)")
    print("="*40)
    print(f"📌 原始 dev_extract.jsonl 行数 : {total_dev}")
    print(f"🗑️ 因与 train_extract 重复剔除 : {removed_dev} 行")
    print(f"💾 最终保留写入 clean 文件 : {kept_dev} 行")
    print(f"✅ 文件已生成: {clean_dev_path}")
    print("="*40)

if __name__ == "__main__":
    # 获取当前脚本的绝对路径，以便在任何地方运行都能找对目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    clean_dev_extract_data(script_dir)