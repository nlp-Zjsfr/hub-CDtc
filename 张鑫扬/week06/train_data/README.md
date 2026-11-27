---
configs:
- config_name: default
  data_files:
  - path: zh.test.csv
    split: test
license: Apache License 2.0
tags:
- "\u9886\u57DF"
- "\u6587\u672C\u9886\u57DF"
- "\u6587\u672C\u5206\u7C7B"
text:
  text-classification:
    language:
    - zh
    type:
    - multi-class
---

<!---
 以上YAML section提供属性/tags描述---
>

<!---
 以下为markdown格式的dataset描述---
>

## 数据集描述

### 数据集简介
该数据集由阿里达摩院创建，用于测试文本领域分类模型能力，分类依据为国民经济行业分类（GB/T 4754—2017），基本覆盖了国民经济全行业，共包含18个大类行业的测试数据：交通运输仓储邮政、住宿餐饮、信息软件、农业、制造业、卫生医疗、国际组织、建筑、房地产、政府组织、教育、文体娱乐、水利环境、电力燃气水生产、科学技术、租赁法律、采矿、金融。 每个领域的具体内容，详见《2017年国民经济行业分类（GB/T 4754—2017）》。


### 数据集支持的任务
可用于中文文本风格分类模型评测，例如[FastText文本领域分类-中文-国民经济行业18大类模型](https://modelscope.cn/models/damo/nlp_domain_classification_chinese/summary)使用该数据集进行了评测。

## 数据集的格式和结构

### 数据格式
该数据集为一个csv文件，每行为一个样本，包括输入文本和对应领域


### 数据集加载方式

```python
import csv
import pandas as pd
df=pd.read_csv('zh.test.csv')
print(df.sample(10))
```

### 数据分片
该数据仅为测试集，不涉及分片