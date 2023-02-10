## 0 Transformer 结构

单层

<img src="note_images\image-20221123191342472.png" alt="image-20221123191342472" style="zoom:50%;" />



多层 Transformer：原论文中采用了 6 个编码器和 6 个解码器

<img src="note_images\88794918439701.png" alt="88794918439701" style="zoom:67%;" />

​    在多层 Transformer 中，多层编码器先对输入序列进行编码，然后得到最后一个 Encoder 的输出 Memory；解码器先通过 Masked Multi-Head Attention 对输入序列进行编码，然后将输出结果同 Memory 通过 Encoder-Decoder Attention 后得到第 1 层解码器的输出；接着再将第 1 层 Decoder 的输出通过 Masked Multi-Head Attention进行编码，最后再将**编码后的结果同Memory**通过Encoder-Decoder Attention 后得到第 2 层解码器的输出，以此类推得到最后一个 Decoder 的输出。

​    在多层 Transformer的解码过程中，每一个 Decoder 在 Encoder Decoder Attention 中所使用的 Memory 均是同一个。

## 1 Attention

### 1.1 Attention and Self-attention

<img src="note_images\image-20221201205037663.png" alt="image-20221201205037663" style="zoom:50%;" />

注意力机制可以描述为将query 和一系列的 key-value 对映射到某个输出的过程，而**这个输出的向量就是根据 query 和 key 计算得到的权重作用于 value 上的权重和**。
$$
Attention(Q,K,V)=softmax(\frac{QK^{T}}{\sqrt{d_k}})V
$$


### 1.2 Multi-Head Attention

目的：解决多头注意力可以用于克服模型在对当前位置的信息进行编码时，会过度的将注意力集中于自身的位置的问题。

![image-20221201205151330](note_images\image-20221201205151330.png)
$$
MultiHead(Q,K,V)=Concat(head_1,···,head_h)W_O\\
where \ head_i=Attention(QW_i^Q,KW_i^K,VW_i^V)\\
W_i^Q\in \mathbb{R}^{d_{model}\times d_k}, \ W_i^K\in \mathbb{R}^{d_{model}\times d_k}, \ W_i^V\in \mathbb{R}^{d_{model}\times d_v}, \ W_i^O\in \mathbb{R}^{hd_v\times d_{model}}
$$
相当于有多个权重矩阵。

作者使用了 h=8 个并行的自注意力模块（8 个头）来构建一个注意力层，并且对于每个自注意力模块都限定了 $d_k = d_v = d_{model} / h = 64$。

### 1.3 Attention Mask

<img src="note_images\image-20221205140033753.png" alt="image-20221205140033753" style="zoom: 67%;" />

在训练过程中的 Decoder 对于每一个样本来说都需要这样一个对称矩阵来掩盖掉当前时刻之后所有位置的信息。目的是为了使得decoder不能看见未来的信息.也就是对于一个序列中的第i个token,解码的时候只能够依靠i时刻之前(包括i)的的输出,而不能依赖于i时刻之后的输出.因此我们要采取一个遮盖的方法(Mask)使得其在计算self-attention的时候只用i个时刻之前的token进行计算,因为Decoder是用来做预测的,而在训练预测能力的时候,我们不能够"提前看答案",因此要将未来的信息给遮盖住。

<img src="note_images\image-20221205142629920.png" alt="image-20221205142629920" style="zoom:50%;" />



### 1.4 Q K V

整个 Transformer 中涉及到自注意力机制的一共有 3 个部分：Encoder 中的 Multi-Head Attention；Decoder 中的 Masked Multi-Head Attention；Encoder 和 Decoder 交互部分的 Multi-Head Attention。

<img src="note_images\image-20221205142451025.png" alt="image-20221205142451025" style="zoom:50%;" />

- 对于 Encoder 中的 Multi-Head Attention 来说，其原始 q、k、v 均是 Encoder的 Token 输入经过 Embedding 后的结果。q、k、v 分别经过一次线性变换（各自乘以一个权重矩阵）后得到了 Q、K、V，然后再进行自注意力运算得到 Encoder 部分的输出结果 Memory。
- 对于 Decoder 中的 Masked Multi-Head Attention 来说，其原始 q、k、v 均是 Decoder 的 Token 输入经过 Embedding 后的结果。q、k、v 分别经过一次线性变换后得到了 Q、K、V，然后再进行自注意力运算得到 Masked Multi-Head Attention 部分的输出结果，即待解码向量。
- 对于 Encoder 和 Decoder 交互部分的 Multi-Head Attention，其原始 q、k、v 分别是上面的待解码向量、Memory 和 Memory。q、k、v 分别经过一次线性变换后得到了 Q、K、V，然后再进行自注意力运算得到 Decoder 部分的输出结果。之所以这样设计也是在模仿传统 Encoder-Decoder 网络模型的解码过程。

### 1.5 Why Self-Attention

1. 降低每层的复杂度
2. 能够并行计算，由所需的最小顺序操作数来衡量。
3. 在网络中计算long-range依赖需要的计算路径长度。即为了学习long-range依赖，信号在网络中必须要经过的路径长度，这个长度越短，模型就越容易学习long-range依赖。

## 2 Add & Norm and Feed Forward

**Add**指 **X**+MultiHeadAttention(**X**)，是一种残差连接，通常用于解决多层网络训练的问题，可以让网络只关注当前差异的部分，在 ResNet 中经常用到：

![image-20221123191221498](C:\Users\lzl\AppData\Roaming\Typora\typora-user-images\image-20221123191221498.png)
$$
LayerNorm(X+MultiHeadAttention(X))\\
LayerNorm(X+FeedForward(X))
$$

**Feed Forward ** 是为了更好的提取特征。通过线性变换，先将数据映射到高纬度的空间再映射到低纬度的空间，提取了更深层次的特征。
$$
FFN(x) = max(0, xW1 + b1)W2 + b2
$$

```python
src2 = self.activation(self.linear1(src))
src2 = self.linear2(self.dropout(src2))
src = src + self.dropout2(src2)
src = self.norm2(src)
```



## 3 Embedding

Embedding = Token_embedding + Positional_embedding

```python
src_embed = self.token_embedding(src)  # [src_len, batch_size, embed_dim]
src_embed = self.pos_embedding(src_embed)  # [src_len, batch_size, embed_dim]
```



### 3.1 Token Embedding

是将各个词（或者字）通过一个 Embedding 层映射到低维稠密的向量空间。

可以使用Vocab，建立词典，一个词对应一个索引即 Token。然后对 Token 进行 Embedding。



### 3.2 Positional Embedding

仅用Token Embedding会缺少时序信息，因此考虑将位置信息加入。PE 就是 Positional Embedding 矩阵
$$
PE_{pos, 2i}=sin(pos/10000^{2i/d_{model}}) \\
PE_{pos, 2i+1}=cos(pos/10000^{2i/d_{model}})
$$
<img src="note_images\image-20221201204203055.png" alt="image-20221201204203055" style="zoom:67%;" />





