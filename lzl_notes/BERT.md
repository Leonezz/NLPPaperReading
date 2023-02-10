## 1 背景

​	现有预训练模型的网络结构限制了模型本身的表达能力，其中最主要的限制就是没有采用双向编码的方法来对输入进行编码。使得模型在编码过程中只能够看到当前时刻之前的信息，而不能够同时捕捉到当前时刻之后的信息。

如下，从左到右编码或者从右向左编码都无法判断 it 指代的是 animal 还是 street

- The animal didn't cross the street because **it** was too tired.

- The animal didn't cross the street because **it** was too narrow.



​	作者提出了采用 BERT（Bidirectional Encoder Representations from Transformers）这一网络结构来实现模型的双向编码学习能力。同时，为了使得模型能够有效的学习到双向编码的能力，BERT 在训练过程中使用了基于掩盖的语言模型（Masked Language Model, MLM），即随机对输入序列中的某些位置进行遮蔽，然后通过模型来对其进行预测。

​	由于 MLM 预测任务能够使得模型编码得到的结果同时包含上下文的语境信息，因此有利于训练得到更深的 BERT 网络模型。除此之外，在训练BERT的过程中作者还加入了下句预测任务（Next Sentence Prediction, NSP），即同时输入两句话到模型中，然后预测第 2 句话是不是第 1 句话的下一句话。



## 2 Bert 网络结构

![image-20221206162140268](note_images\image-20221206162140268.png)



## 3 Segment Embedding

​	Token Embedding 和 transformer 一样，但 Positional Embedding 并不是采用公式计算得到，而是类似普通的词嵌入一样为每一个位置初始化了一个向量，然后随着网络一起训练得到。

​	Segment Embedding 是用来区分输入序列中的不同部分，其本质就是通过一个普通的词嵌入来区分每一个序列所处的位置。划分句子，同样是初始化向量后，随着网络一起训练得到。

​	例如在 NSP 任务中，对于任意一个句子（一共两个）来说，其中的每一位置都将用同一个向量来进行表示，以此来区分哪部分字符是第 1 句哪部分字符是第 2 句，即此时 Segment 词表的长度为 2。

​	最后，再将这三部分 Embedding 后的结果相加（并进行标准化）便得到了最终的 Input Embedding 部分的输出

![image-20221209211656417](note_images\image-20221209211656417.png)



## 4 MLM 任务

​	对于 MLM 任务来说，其做法是随机掩盖掉输入序列中 15%的 Token（即用“[MASK]”替换掉原有的 Token），然后在 BERT 的输出结果中取对应掩盖位置上的向量进行真实值预测。

​	作者提到，虽然 MLM 的这种做法能够得到一个很好的预训练模型，但是仍旧存在不足之处。由于在 fine-tuning 时，由于输入序列中并不存在“[MASK]”这样的 Token，因此这将导致 pre-training 和 fine-tuning 之间存在不匹配不一致的问题。

​	为了解决这一问题，作者在原始 MLM 的基础了做了部分改动，即先选定 15% 的 Token，然后将其中的 80% 替换为“[MASK]”、 10% 随机替换为其它 Token、剩下的 10% 不变。最后取这 15% 的 Token 对应的输出做分类来预测其真实值。

<img src="note_images\image-20221206202322117.png" alt="image-20221206202322117" style="zoom: 67%;" />

## 5 NSP任务

​	由于很多下游任务需要依赖于分析两句话之间的关系来进行建模，例如问题回答等。为了使得模型能够具备有这样的能力，作者在论文中又提出了二分类的下句预测任务。

​	对于每个样本来说都是由 A 和 B 两句话构成，其中 50% 的情况 B 确实为 A 的下一句话（标签为 IsNext），另外的 50% 的情况是 B 为语料中其它的随机句子（标签为 NotNext），然后模型来预测 B 是否为 A 的下一句话。

​	如下图所示便是 ML 和 NSP 这两个任务在 BERT 预训练时的输入输出示意图，其中最上层输出的 *C* 在预训练时用于 NSP 中的分类任务；其它位置上的 $T_i、  T^{'}_j$ 则用于预测被掩盖的 Token。

<img src="note_images\43636006724959.png" alt="43636006724959" style="zoom:67%;" />





























