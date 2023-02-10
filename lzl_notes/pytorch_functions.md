[pytorch中文文档](https://www.pytorchtutorial.com/docs/)

## 1 torch.nn

### 1.1 nn.embedding

一个保存了固定字典和大小的简单查找表。

这个模块常用来保存词嵌入和用下标检索它们。模块的输入是一个下标的列表，输出是对应的词嵌入。

```python
class torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2, scale_grad_by_freq=False, sparse=False)
参数：

    num_embeddings (int) - 嵌入字典的大小
    embedding_dim (int) - 每个嵌入向量的大小
    padding_idx (int, optional) - 如果提供的话，输出遇到此下标时用零填充
    max_norm (float, optional) - 如果提供的话，会重新归一化词嵌入，使它们的范数小于提供的值
    norm_type (float, optional) - 对于max_norm选项计算p范数时的p
    scale_grad_by_freq (boolean, optional) - 如果提供的话，会根据字典中单词频率缩放梯度

变量：

    weight (Tensor) -形状为(num_embeddings, embedding_dim)的模块中可学习的权值

形状：

    输入： LongTensor (N, W), N = mini-batch, W = 每个mini-batch中提取的下标数
    输出： (N, W, embedding_dim)
```



```python
self.embedding = nn.Embedding(vocab_size, emb_size)
```

self.embedding本质上就是一个查找表，词作为下标去索引。



### 1.2 nn.LayerNorm





### 1.3 nn.functional

 ```python
 import torch.nn.functional as F
 ```

#### 1.3.1 F.softmax

$$
Softmax(x_i)=\frac{exp(x_i)}{\sum_jexp(x_j)}
$$

```python
F.softmax(input, dim=None, _stacklevel=3, dtype=None)
```

- **input** (*Tensor*) – input
- **dim** (*int*) – A dimension along which softmax will be computed.
- **dtype** (*torch.dtype*, optional) – the desired data type of returned tensor. If specified, the input tensor is casted to *dtype* before the operation is performed. This is useful for preventing data type overflows. Default: None.

其输入值是一个向量，向量中元素为任意实数的评分值，softmax 函数对其进行压缩，输出一个向量，其中每个元素值在0到1之间，且所有元素之和为1
