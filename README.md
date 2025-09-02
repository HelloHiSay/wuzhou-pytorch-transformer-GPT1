# wuzhou-pytorch-transformer-GPT1 基于Pytorch实现GPT1

# 1 环境配置
## 1.1 基础环境
python == 3.8
ftfy == 6.3.1
numpy == 1.24.1
pandas == 2.0.3
scikit_learn == 1.3.2
spacy == 3.4.4
torch == 2.4.1+cu121
tqdm == 4.67.1
CUDA Version == 12.0
![[Picture/基于Pytorch实现GPT1/8a3183172fe5a2c2511c2e6e05d5ca5c_MD5.png|375]]

## 1.2 模型权重文件
下载OpenAI预训练权重并吧model文件夹放入和tran.py同一级文件夹下[finetune-transformer-lm](https://github.com/openai/finetune-transformer-lm)

## 1.3 ROCStories 完形填空任务数据集
[ROCStories 和故事完形填空测试](https://cs.rochester.edu/nlp/rocstories/)
ROCStories Cloze Test 是一个阅读理解数据集，每篇“故事”由 4 句话组成（上下文），后面有两个候选结尾（ending1 和 ending2），目标是判断哪个结尾更合理

# 2 模型结构
n_layer = 12, n_head = 12, n_embd = 768（12层，12头，768维）与原论文一致
![[Picture/基于Pytorch实现GPT1/741c0830cfe77cf7e0e97141bf7091a1_MD5.png]]

# 3 数据处理
datasets.py
![[Picture/基于Pytorch实现GPT1/9b4ba989e9aa09d85b82496a388162d7_MD5.png]]
datasets.py文件下_rocstories返回四个参数分别对应故事上下文，第一个候选结尾，第二个候选结尾，存储标签

rocstories进行验证集训练集测试集的划分，并返回四个元组
(trX1, trX2, trX3, trY)训练集：故事上下文、候选结尾1、候选结尾2、标签
(vaX1, vaX2, vaX3, vaY)验证集：故事上下文、候选结尾1、候选结尾2、标签
(teX1, teX2, teX3)测试集：故事上下文、候选结尾1、候选结尾2（无标签）

# 4 训练策略和方法
任务目标：ROCStories 任务是给一个故事开头x<sub>1</sub>，两个候选结尾x<sub>2</sub>、x<sub>3</sub>，选择合理的结尾

AI总结：该训练方法基于GPT模型迁移学习，采用语言建模+分类联合损失，配合AdmW优化、warmup调度、梯度裁剪和dropout正则化，通过验证集选择最优模型，最终实现ROCStories 多选任务的准确预测


# 5 结果
通过一下命令来复现
```
python -m spacy download en

python train.py --dataset rocstories --desc rocstories --submit --analysis --data_dir ./data/ROCStories/ --n_gpu 8
```

![[Picture/基于Pytorch实现GPT1/4cfd26c1f05c427cd65398208c276635_MD5.png]]

epoch 0  : 74.87% (train)  74.06% (valid)
epoch 1  : 86.90%          83.42%
epoch 2  : 92.51%          87.43%
Best Valid Acc : 87.43%
Test Acc       : 84.18%
原论文故事完形填空数据集有86.5%的准确率与这次复现的准确率比较贴近
![[Picture/基于Pytorch实现GPT1/467347c4916fc43cddb06861e7d2aeb2_MD5.png]]

huggingface的Github开源链接：[huggingface/pytorch-openai-transformer-lm: 🐥A PyTorch implementation of OpenAI's finetuned transformer language model with a script to import the weights pre-trained by OpenAI](https://github.com/huggingface/pytorch-openai-transformer-lm)

五舟配置的开源链接：
