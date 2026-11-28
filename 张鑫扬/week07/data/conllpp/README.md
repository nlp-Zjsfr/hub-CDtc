---
license: Apache License 2.0
tags:
- NER
text:
  token-classification:    
    type:
      - ner
    language:
      - en

---

# conllpp命名实体识别数据集

## 数据集概述
conllpp数据集是conll2003命名实体识别数据集的修正版本，其中测试集中5.38%的句子标签通过人工校验进行修正。
为了保持数据集完整，该数据集同时包括了conll2003的训练集、验证集。

### 数据集简介
本数据集包括训练集（14041）、验证集（3250）、测试集（3453），实体类型包括地点(LOC)、混合(MISC)、组织(ORG)、人名(PER)。

### 数据集的格式和结构
数据格式采用conll标准，NER数据包括两列，第一列输入句中的词划分以及最后一列中每个词对应的命名实体类型标签。一个具体case的例子如下：

```
SOCCER NN I-NP O
- : O O
JAPAN NNP I-NP B-LOC
GET VB I-VP O
LUCKY NNP I-NP O
WIN NNP I-NP O
, , O O
CHINA NNP I-NP B-LOC
IN IN I-PP O
SURPRISE DT I-NP O
DEFEAT NN I-NP O
. . O O
```

## 数据集版权信息

Creative Commons Attribution 4.0 International。

## 引用方式
  ```bib
    @article{wang2019cross,
      title={CrossWeigh: Training Named Entity Tagger from Imperfect Annotations},
      author={Wang, Zihan and Shang, Jingbo and Liu, Liyuan and Lu, Lihao and Liu, Jiacheng and Han, Jiawei},
      journal={arXiv preprint arXiv:1909.01441},
      year={2019}
    }
  ```