# Prophetnet_paddle
## 模型简介
ProphetNet（先知网络）是一种新型的 seq2seq 预训练模型。在训练时，Prophetnet 每一时刻将会学习同时预测未来的 N 个字符，这种自监督学习目标可以使得模型考虑未来更远的字符，防止模型对强局部相关（strong local correlation）过拟合。

本项目是 Prophetnet 在 PaddlePaddle 2.2 上开源实现的文本摘要的例子，包含了在 CNN/DailyMail 数据集，Gigaword 数据集上微调和生成的代码。

参考项目：

- https://github.com/microsoft/ProphetNet/tree/master/GLGE_baselines

项目aistudio地址：

- https://aistudio.baidu.com/aistudio/projectdetail/3488800

## 项目依赖
```
pip install -r requirements.txt
python -m pip install paddlepaddle-gpu==2.2.2.post112 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
pip install paddlenlp==2.2.3
```

## 复现精度
>#### 在CNN/DM数据集的测试效果如下表。
>paddlepaddle复现结果

|网络 |opt|batch_size|数据集|ROUGE_1|ROUGE_2|ROUGE_L|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|prophetnet-large-uncased|Adam|4|CNN/DM|44.17|21.24|41.36|

>训练日志：[链接](log/cnndm_train.log)
> 
>原论文结果

|网络 |数据集|ROUGE_1|ROUGE_2|ROUGE_L|
| :---: | :---: | :---: | :---: | :---: |
|prophetnet-large-uncased|CNN/DM|44.20|21.17|41.30|

>#### 在gigaword数据集的测试效果如下表。
>paddlepaddle复现结果

|网络 |opt|batch_size|数据集|ROUGE_1|ROUGE_2|ROUGE_L|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|prophetnet-large-uncased|Adam|16|gigaword|38.92|19.81|36.06|

>训练日志：[链接](log/gigaword_train.log)
> 
>原论文结果

|网络 |数据集|ROUGE_1|ROUGE_2|ROUGE_L|
| :---: | :---: | :---: | :---: | :---: |
|prophetnet-large-uncased|gigaword|39.51|20.42|36.69|

## 获取数据
GLGE 数据集下载: [链接](https://drive.google.com/file/d/1F4zppa9Gqrh6iNyVsZJkxfbm5waalqEA/view)

GLGE 测试集下载: [链接](https://drive.google.com/file/d/11lDXIG87dChIfukq3x2Wx4r5_duCRm_J/view)

将glge_public.tar与glge_hidden_v1.1.tar.gz放入到项目根目录下。
```
bash uncompress_data.sh
```

## 下载预训练权重与词表
```
wget https://huggingface.co/microsoft/prophetnet-large-uncased/resolve/main/pytorch_model.bin
wget https://huggingface.co/microsoft/prophetnet-large-uncased/resolve/main/prophetnet.tokenizer
```
paddle权重转化(需要安装pytorch)
```
python torch2paddle.py
```
已转化好的paddle权重[下载链接](https://pan.baidu.com/s/1FOnd01rNvDJoONYegacq1Q), 提取码：o28q，下载后放入项目根目录。

## 数据预处理
```
python uncase_tokenize_data.py --dataset <DATASET>
```

说明：

- `<DATASET>`可选`cnndm`, `gigaword`.

## 训练
```
bash run_train.sh <DATASET>
```

已经finetune好的模型权重：

- cnndm : [链接](https://pan.baidu.com/s/1cemrUDxkqEW9raoasJ_VKw), 提取码：1egi

- gigaword : [链接](https://pan.baidu.com/s/1qRH2FStT3vNQtDjZLkYJBQ), 提取码：on5v

## 评估
使用prophetNet源码的[评估脚本](https://github.com/microsoft/ProphetNet/tree/master/GLGE_baselines/script/script/evaluate), 此脚本依赖于pyrouge，需要提前安装rouge。
```
pip install git+https://github.com/pltrdy/pyrouge
```
```
bash run_eval.sh <DATASET>
```

## 实验环境
- GPU RTX3090 * 1, CPU Intel i7-11700k
- Ubuntu 18.04
