---
title: "BART: Denoising Sequence to Sequence Pre training for Natural Language Generation Translation and Comprehension"
authors: [Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levym Ves Stoyanov, Luke Zettlemoyer]
aliases: [BART]
tags: [PLM, BART]
date: 2021-09-07
reported: false
---
<cite>[[Mike Lewis]], [[Yinhan Liu]], [[Naman Goyal]], [[Marjan Ghazvininejad]], [[Abdelrahman Mohamed]], [[Omer Levym Ves Stoyanov]], [[Luke Zettlemoyer]]</cite>

# Introduction

BART: Bidirectional and Auto-Regressive Transformers

BART 是一种用于预训练 seq2seq 模型的降噪自动编码器。BART使用标准的 [[Attention-is-all-you-need|Transformer]] 架构，训练时使用任意噪声函数对文本加噪，然后试图重建受损的文本。

BART 的一个关键优势在于它的预训练过程使用任意的噪声函数，包括打乱原始句子的顺序，对任意长度的文本片段进行掩码替换等。这迫使模型对句子的整体长度进行更多推理，并对输入进行更长范围的转换。

BART 格外适合微调到生成任务，在阅读理解，抽象对话，问答和总结任务中也有很好的效果。本文还基于 BART 介绍了一种新的机器翻译的模式，即在 BART 之上添加一些额外的 Transformer 层来将外语转换为带有噪声的目标语言，从而使用 BART 作为目标端语言模型进行去噪。

# Background

早期的语言模型使用 L2R(GPT) 或者 L2R 和 R2L 叠加 (ELMo) 的方法，这样只能基于一侧的信息或者无法提取两侧信息的交互，在一些任务上有问题。（自回归模型，在生成任务上表现较好，但在分类任务上较差）

[[BERT-Pre-training-of-Deep-Bidirectional-Transformers-for-Language-Understanding|BERT]] 使用 MLM ，允许在预训练过程中学习两侧信息的交互，但是由于预测过程不是自回归的，在生成任务上不是最优的。

# Model

BART 是一个标准的 seq2seq 模型，其编码器是双向的，解码器是 L2R 自回归的。在预训练时我们优化重建文本和原始文本的负对数似然 (negative log likehood)

## Architecture

BART 使用标准的 Transformer 架构，但是将 [[ReLU|ReLU]] 激活函数替换为GeLus，并且从分布 $N(0, 0.02)$ 中初始化参数。

BART 的 encoder 模型结构与 BERT 相似，有以下不同：

1. decoder 的每一层额外的在 encoder 的最终隐藏层上做交叉注意
2. BERT 在执行单词预测之前还有一层前馈网络，但 BART 没有

## Pre-training BART

BART 基于受损文本重建任务进行预训练，重建损失即 decoder 的输出和原始文本之间的交叉熵。

BART 允许使用任意类型的噪声。实验中我们使用了以下几种：

- Token Masking:  与 BERT 使用的相同，即随机采样一些 tokens 并使用 [MASK] 替换
- Token Deletion:  随机删去一些 tokens ，与 Token Masking 相比，这种方法迫使模型预测被删除的位置
- Text Infilling: 随机采样一些长度符合泊松分布( $\lambda = 3$ ) 的文本片段，并用遮罩 [MASK] 替换。对于长度为 0 的文本片段，相当于插入了 [MASK] 。这种噪声迫使模型预测被替换的文本片段的长度。
- Sentence Permutation: 将文档按照句号分割成不同的句子，然后随机排列句子的顺序。这种噪声迫使模型学习同一文档中句子的顺序。
- Document Rotation: 随机选择一个 token ，然后将文本旋转到以这个 token 为开头的状态。这种噪声训练模型识别文本开头的能力。

![](BART/1.png)

# Fine-tuning BART

## 序列分类任务

对于序列分类任务，将输入同时送进 encoder 和 decoder ，与 BERT 的 [CLS] token 类似， BART 在 decoder 输出的末尾位置添加一个 token 专门用于分类任务，对应于这个 token 的最终隐藏层输出经过一个线性分类器得到输入序列的分类

在末尾添加的原因是这样 decoder 中这个 token 的表征可以注意到整个输入在 decoder 中的信息。

## Token分类任务

对于 Token 分类任务，将完整的输入文本送入 encoder 和 decoder ，然后使用 decoder 的顶部隐藏状态 (top hidden state) 作为每个词的表征，这个表征用于 token 的分类(即每个词属于某种 token )。

## 序列生成任务

BART 的 decoder 是自回归的，因此它可以直接微调到如抽取式问答和总结类的生成任务。在这类任务中，信息直接从输入中抽取并修改，这与 BART 的去噪预训练目标吻合。

## 机器翻译任务

对于翻译任务，BART 通过将 encoder 的 embedding 层替换为少量随机初始化的encoder 层，利用这个新的 encoder 将外语编码成有噪声的目标语言的编码。然后将该有噪声的编码作为 BART 的输入，由 BART 将其降噪成高质量的翻译结果。即， BART 作为目标语言端解码器，新添加的 encoder 作为外语端编码器，组成了一个新的 seq2seq 机器翻译模型。
具体地说，微调该翻译任务分为两步：

1.  保持大多数的 BART 参数不变，仅更新新添加的 encoder ， BART 的位置嵌入层和 BART 的 encoder 中第一层自注意力层的投影矩阵( $W_Q, W_K, W_V$ ) 
2. 在少量迭代下更新所有的参数

# 数据/任务集

SQuAD, MNLI, ELI5, XSum。详情参考：

[[Data-Set-and-Benchmark]]

# 实验说明的一些结论

实验结果：

![](BART/2.png)

- 预训练方法的性能因任务而异
- Token Masking 是十分重要的预训练目标
- L2R的预训练能够提高生成任务的性能
- 对于 SQuAD 任务，双向编码器十分重要(在分类任务中，结合 token 之后的信息十分关键)
- 预训练目标不是唯一关键的因素