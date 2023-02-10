## 0 词向量

### 1 Word2vec

方法假设：文本中离得越近的词语相似度越高。

主要用 skip-gram 计算

![image-20221125150915702](C:\Users\lzl\AppData\Roaming\Typora\typora-user-images\image-20221125150915702.png)



> **评估 word2vec 效果** 
>
> 1. 输出与特定词语相关度比较大的词语
> 2. 可视化
> 3. 类比实验：国王-王后=男人-女人
>
> **缺点** 
>
> 1. 没有考虑多义词
> 2. skip-gram时窗口长度有限
> 3. 没有考虑全局文本信息
> 4. 不是严格意义的语序
> 5. ······



### 2 glove

[详解](http://www.fanyeong.com/2018/02/19/glove-in-detail/) 



## 1 BP神经网络

Back-Propagation 反向传播

[知乎](https://zhuanlan.zhihu.com/p/486303925) 



## 2 MLP（多层感知机）



[MLP](https://aistudio.csdn.net/62e38aaecd38997446774dcb.html?spm=1001.2101.3001.6661.1&utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-1-93405572-blog-102802517.pc_relevant_recovery_v2&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-1-93405572-blog-102802517.pc_relevant_recovery_v2&utm_relevant_index=1) 



## 3 LSTM(长短期记忆网络)

[B站](https://www.bilibili.com/video/BV1JU4y1H7PC/?spm_id_from=333.337.search-card.all.click&vd_source=2ca6993c5ecbc492902a1449c800fe3d) 

[视频](https://www.bilibili.com/video/BV1Z34y1k7mc/?share_source=copy_web&vd_source=2ca6993c5ecbc492902a1449c800fe3d) 





## 4 





## 5 CRNN





## 7 Self-attention

自注意力机制解决的问题是：神经网络接收的输入是很多大小不一的向量，并且不同向量向量之间有一定的关系，但是实际训练的时候无法充分发挥这些输入之间的关系而导致模型训练结果效果极差。

==QKV机制== 

假设有一个问题：给出一段文本，使用一些关键词对它进行描述！
为了方便统一正确答案，这道题可能预先已经给大家写出来一些关键词作为提示，其中这些给出的提示就可以看作**key**。
而整个文本的信息就相当于**query**，**value**的含义则更加抽象，可比作是你看到这段文本信息后，脑子里复现的答案信息。
假设刚开始大家都不是很聪明，第一次看到这段文本之后脑子里基本上复现的信息**value**就只有提示的这些信息即**key**，因为**key**与**value**基本是相同的。
但是随着对这个问题的深入理解，通过我们的思考脑子里想起来的东西越来越多，并且能够开始对这段文本即**query**，提取关键词信息进行表示，这就是注意力作用的过程，通过这整个过程我们最终脑子里的**value**发生了变化。
根据提示**key**生成了**query**的关键词表示方法，也就是另一种特征表示方法。
刚刚我们说到**key**和**value**一般情况下默认是相同，与**query**是不同的，这种是我们一般的注意力输入形式。
但有一种特殊情况，就是我们**query**与**key**和**value**相同，这种情况我们成为自注意力机制，就如同我们刚刚的例子，使用一般的注意力机制，是使用不同给定文本的关键词表示它。
而自注意力机制，需要用给定文本自身来表述自己，也就是说你需要从源文本中抽取关键词来表述它，相当于对文本自身的一次特征提取。




Q是一组[查询语句](https://www.zhihu.com/search?q=查询语句&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2718310467})，V是数据库，里面有若干数据项。对于每一条查询语句，我们期望从数据库中查询出一个数据项（[加权](https://www.zhihu.com/search?q=加权&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2718310467})过后的）来。如何查询？这既要考虑每个q本身，又要考虑V中每一个项。如果用K表示一组钥匙，这组钥匙每一把对应V中每一项，代表了V中每一项的某种查询特征，（所以K和V的数量一定是相等的，维度则没有严格限制，做[attention](https://www.zhihu.com/search?q=attention&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2718310467})时维度和q一样只是为了在做[点积](https://www.zhihu.com/search?q=点积&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2718310467})时方便，不过也存在不用点积的attention）。然后对于每一个Q中的q，我们去求和每一个k的attention，作为对应value的加权系数，并用它来加权数据库V中的每一项，就得到了q期望的查询结果。
所以query是查询语句，value是数据项，key是对应每个数据项的钥匙。名字起得是很生动的。不过和真正的[数据库查询](https://www.zhihu.com/search?q=数据库查询&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2718310467})不一样的是，我们不仅考虑了查询语句，还把数据库中所有项都加权作为结果。所以说是全局的。

理论指导实践:1.对于一个文本，我们希望找到某张图片中和文本描述相关的局部图像，怎么办？文本作query(查询），图像做value（数据库）2.对于一个图像，想要找一个文本中和图像所含内容有关的局部文本，如何设计？图像作query，文本作value.3.自注意力（我查我自己）:我们想知道句子中某个词在整个句子中的分量（或者相关文本），怎么设计？句子本身乘以三个矩阵得到Q,K,V，每个词去查整个句子。4.交叉注意力（查别人）:transformer模型的decoder中，由decoder的输入经过变换作为query，由encoder的输出作为key和value（数据库）。value和query来自不同的地方，就是交叉注意力。可以看到key和value一定是代表着同一个东西。即:[Q,(K,V)]。如果用encoder的输出做value，用[decoder](https://www.zhihu.com/search?q=decoder&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2718310467})的输入做key和query 那就完完全全不make sense了。所以从宏观上就可以判断谁该作query，谁该作value和key 。而不是乱设计。

==如何知道什么是重点？== 



| <img src="C:\Users\lzl\AppData\Roaming\Typora\typora-user-images\image-20221122142003260.png" alt="image-20221122142003260"  /> | ![image-20221122143006882](C:\Users\lzl\AppData\Roaming\Typora\typora-user-images\image-20221122143006882.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![image-20221122143151270](C:\Users\lzl\AppData\Roaming\Typora\typora-user-images\image-20221122143151270.png) | ![image-20221122143330365](C:\Users\lzl\AppData\Roaming\Typora\typora-user-images\image-20221122143330365.png) |
| 矩阵运算：                                                   |                                                              |
| ![image-20221122145128074](C:\Users\lzl\AppData\Roaming\Typora\typora-user-images\image-20221122145128074.png) | ![image-20221122145353567](C:\Users\lzl\AppData\Roaming\Typora\typora-user-images\image-20221122145353567.png) |
| ![image-20221122145815255](C:\Users\lzl\AppData\Roaming\Typora\typora-user-images\image-20221122145815255.png) | A'是含有注意力权重的矩阵                                     |



==Multi-head Self-attention== 

| <img src="C:\Users\lzl\AppData\Roaming\Typora\typora-user-images\image-20221122150732786.png" alt="image-20221122150732786" style="zoom:67%;" /><img src="C:\Users\lzl\AppData\Roaming\Typora\typora-user-images\image-20221122150841991.png" alt="image-20221122150841991" style="zoom:67%;" /> |
| :----------------------------------------------------------- |



==为A'添加位置信息== 

添加位置信息 $e_i$ 

<img src="C:\Users\lzl\AppData\Roaming\Typora\typora-user-images\image-20221122151707533.png" alt="image-20221122151707533" style="zoom:80%;" />



==与CNN对比== 

| <img src="C:\Users\lzl\AppData\Roaming\Typora\typora-user-images\image-20221122153527654.png" alt="image-20221122153527654" style="zoom:110%;" /> | <img src="C:\Users\lzl\AppData\Roaming\Typora\typora-user-images\image-20221122153552122.png" alt="image-20221122153552122"  /> | ![image-20221122153637812](C:\Users\lzl\AppData\Roaming\Typora\typora-user-images\image-20221122153637812.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |

==与RNN对比== 

<img src="C:\Users\lzl\AppData\Roaming\Typora\typora-user-images\image-20221122153401230.png" alt="image-20221122153401230" style="zoom: 67%;" />





## 8 auto-regressive

![image-20221123153946482](C:\Users\lzl\AppData\Roaming\Typora\typora-user-images\image-20221123153946482.png)



