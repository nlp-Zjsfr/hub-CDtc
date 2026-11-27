# 预训练模型下载说明

本目录下的权重文件**没有提交到 Git**（体积大，GitHub 单文件上限 100 MB，
且公开 fork 不允许上传新的 LFS 对象）。换机器后按下面命令重新下载即可。

## 1. ModelScope 中文 RoBERTa-base（`bert/` 使用）

```bash
python -c "from modelscope.hub.snapshot_download import snapshot_download; \
snapshot_download('iic/nlp_roberta_backbone_base_std', \
local_dir='common/pretrain_models/roberta-base-chinese')"
```

## 2. bert-base-chinese（项目早期 TNEWS 管线使用，可选）

```bash
python -c "from modelscope.hub.snapshot_download import snapshot_download; \
snapshot_download('google-bert/bert-base-chinese', \
local_dir='common/pretrain_models/bert-base-chinese')"
```

如果 ModelScope 上没有该镜像，也可以从 Hugging Face 下载后放到同一目录：

```bash
huggingface-cli download google-bert/bert-base-chinese \
  --local-dir common/pretrain_models/bert-base-chinese
```

下载完成后可自检路径：

```bash
python common/model_paths.py
```

## 3. Qwen3-0.6B（`llm_*` / `service` 使用，可选）

```bash
python -c "from modelscope.hub.snapshot_download import snapshot_download; \
snapshot_download('Qwen/Qwen3-0.6B', local_dir='common/pretrain_models/Qwen3-0.6B')"
```
