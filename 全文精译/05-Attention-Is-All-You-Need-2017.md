# 注意力就是你所需要的一切（全文精译）

## 书目信息

- **原题**：Attention Is All You Need
- **作者**：Ashish Vaswani、Noam Shazeer、Niki Parmar、Jakob Uszkoreit、Llion Jones、Aidan N. Gomez、Łukasz Kaiser、Illia Polosukhin
- **单位**：Google Brain、Google Research、多伦多大学（各作者具体单位见正文首页注）
- **发表**：《神经信息处理系统进展》（Advances in Neural Information Processing Systems）第 30 卷，NeurIPS 2017 会议论文
- **arXiv**：arXiv:1706.03762 [cs.CL]；v1 于 2017 年 6 月 12 日提交，现行版为 v7，最后修订于 2023 年 8 月 2 日（arXiv 摘要页标明）
- **原文链接（核实过的真实来源链）**：
  - arXiv 摘要页（核对题名、作者、摘要与版本信息所据）：https://arxiv.org/abs/1706.03762
  - arXiv HTML 全文版 v7（本译文取全文所据版本）：https://arxiv.org/html/1706.03762v7
- **对应导读**：见《AI 顶尖论文谱系》Transformer 篇（★）
- **译本说明**：本译文据上述 arXiv 现行版（v7）全文逐节译出。v5 之后的修订版包含附录（Attention Visualizations，注意力可视化）与作者单位等修正，本译文以现行版为准：正文第 1–7 节、图 1–图 5 图注、表 1–表 4、脚注 1–2、致谢，以及附录的图 3–图 5 说明文字全部译出；公式以 LaTeX 呈现并附中文说明；参考文献 [1]–[40] 保留英文原样。原文图像本身未嵌入本文件，图注已全部译出，附录三幅注意力图的画面内容另以译注概述。第 6.1 节正文有一处英法 BLEU 数字，HTML 版作 41.0，与摘要及表 2 的 41.8 不一致，译文以 41.8 为准并在该处加译注标明。译者注一律写作"（译注：……）"，不与原文混排。本文件为 Ezra 个人学习收藏，仅本地存档。

（译注：本文后世通称 Transformer 论文。题名直译"注意力就是你所需要的一切"，借用了披头士歌曲《All You Need Is Love》的句式。文中 "transduction" 指把一个序列转成另一个序列的任务，译作"序列转换"；"self-attention" 译作"自注意力"；"multi-head attention" 译作"多头注意力"。）

---

## 注意力就是你所需要的一切

**Ashish Vaswani\*、Noam Shazeer\*、Niki Parmar\*、Jakob Uszkoreit\*、Llion Jones\*、Aidan N. Gomez\*†、Łukasz Kaiser\*、Illia Polosukhin\*‡**

\* 贡献相同（equal contribution）。作者排列顺序随机。Jakob 提出用自注意力取代循环神经网络（RNN），并启动了验证这一想法的努力。Ashish 与 Illia 一起设计并实现了第一批 Transformer 模型，且在本文工作的每一个环节都起关键作用。Noam 提出缩放点积注意力（scaled dot-product attention）、多头注意力（multi-head attention）以及无参数的位置表示，并成为另一位几乎参与全部细节的人。Niki 在我们最初的代码库与 tensor2tensor 中设计、实现、调参并评测了不计其数的模型变体。Llion 也试验了新颖的模型变体，负责我们最初的代码库，以及高效推理与可视化。Lukasz 与 Aidan 花了无数个长日设计并实现 tensor2tensor 的各个部分，取代了我们早先的代码库，大幅改进了结果，也极大地加速了我们的研究。

单位与邮箱：Ashish Vaswani，Google Brain，avaswani@google.com；Noam Shazeer，Google Brain，noam@google.com；Niki Parmar，Google Research，nikip@google.com；Jakob Uszkoreit，Google Research，usz@google.com；Llion Jones，Google Research，llion@google.com；Aidan N. Gomez，多伦多大学（† 本工作完成于其在 Google Brain 期间），aidan@cs.toronto.edu；Łukasz Kaiser，Google Brain，lukaszkaiser@google.com；Illia Polosukhin（‡ 本工作完成于其在 Google Research 期间），illia.polosukhin@gmail.com。

（译注：原文首页另有一行授权声明："Provided proper attribution is provided, Google hereby grants permission to reproduce the tables and figures in this paper solely for use in journalistic or scholarly works."（只要注明出处，Google 特此授权仅为新闻或学术用途复制本文的表格与图。））

### 摘要

占主导地位的序列转换（sequence transduction）模型，基于复杂的循环（recurrent）或卷积（convolutional）神经网络，其中包含一个编码器（encoder）和一个解码器（decoder）。表现最好的模型还通过注意力机制（attention mechanism）把编码器与解码器连接起来。我们提出一种新的简单网络架构——Transformer（变换器），它完全基于注意力机制，彻底抛弃了循环与卷积。在两个机器翻译任务上的实验表明，这类模型质量更优，同时并行性更好、训练所需时间显著更少。我们的模型在 WMT 2014 英德翻译任务上取得 28.4 BLEU，比此前最好的结果（包括集成模型）高出 2 BLEU 以上。在 WMT 2014 英法翻译任务上，我们的模型在八块 GPU 上训练 3.5 天后，取得了 41.8 的单模型最先进 BLEU 分数，训练成本只是文献中最好模型的一小部分。我们还把 Transformer 成功用于英语成分句法分析（English constituency parsing），在大数据与有限数据两种设定下都表现良好，说明它能很好地推广到其他任务。

## 1 引言

循环神经网络，尤其是长短期记忆（long short-term memory）[13] 与门控循环（gated recurrent）神经网络 [7]，已经牢固确立了它们在语言建模、机器翻译等序列建模与转换问题上的最先进地位 [35, 2, 5]。此后，大量工作不断推进循环语言模型与编码器—解码器架构的边界 [38, 24, 15]。

循环模型通常沿输入与输出序列的符号位置（symbol positions）分解计算。把位置与计算时间步对齐之后，它们生成一列隐藏状态 $h_t$，由前一个隐藏状态 $h_{t-1}$ 与位置 $t$ 的输入共同决定。这种与生俱来的序列性，使训练样本内部无法并行计算；序列变长时这一点变得关键，因为内存限制会制约样本之间的批处理。

近来的工作通过分解技巧（factorization tricks）[21] 与条件计算（conditional computation）[32] 显著提高了计算效率，后者还改进了模型表现。但序列计算这一根本约束依然存在。

注意力机制已经成为各类任务中出色的序列建模与转换模型不可或缺的一部分，它能对依赖关系建模，不论这些依赖在输入或输出序列中的距离多远 [2, 19]。不过，除极少数情形 [27] 外，这类注意力机制都是与循环网络结合使用的。

本文提出 Transformer——一种摒弃循环、完全依靠注意力机制来刻画输入与输出之间全局依赖的模型架构。Transformer 的并行程度显著更高，在八块 P100 GPU 上只训练十二小时，就能在翻译质量上达到新的最先进水平。

## 2 背景

减少序列计算这一目标，同样是 Extended Neural GPU [16]、ByteNet [18] 与 ConvS2S [9] 的基础。这三者都以卷积神经网络为基本构建块，对所有输入与输出位置并行计算隐藏表示。在这些模型里，把任意两个输入或输出位置的信号关联起来所需的操作数，随位置之间的距离增长——ConvS2S 中是线性增长，ByteNet 中是对数增长。这使学习远距离位置之间的依赖变得更难 [12]。在 Transformer 中，这个操作数被降为常数，代价是注意力加权位置被平均后有效分辨率有所降低；我们用第 3.2 节所述的多头注意力来抵消这一效应。

自注意力（self-attention），有时也叫序列内注意力（intra-attention），是一种把单个序列的不同位置关联起来、从而计算该序列表征的注意力机制。自注意力已成功用于多种任务，包括阅读理解、摘要生成、文本蕴涵，以及学习与任务无关的句子表示 [4, 27, 28, 22]。

端到端记忆网络（end-to-end memory networks）基于循环式的注意力机制，而不是与序列对齐的循环，已被证明在简单的语言问答与语言建模任务上表现良好 [34]。

但据我们所知，Transformer 是第一个完全依靠自注意力来计算输入与输出的表示、不用与序列对齐的 RNN 或卷积的转换模型。

以下各节将描述 Transformer，阐明采用自注意力的理由，并讨论它相对于 [17, 18] 与 [9] 这类模型的优势。

## 3 模型架构

**图 1** Transformer——模型架构。（译注：图的左半是编码器，右半是解码器。编码器一层自下而上为：输入嵌入加位置编码、多头注意力、相加与归一化、前馈网络、相加与归一化，整层堆叠 N 次；解码器多出一个对编码器输出做多头注意力的子层，末端经线性层与 softmax 输出概率。）

多数有竞争力的神经序列转换模型都有编码器—解码器结构 [5, 2, 35]。编码器把由符号表示组成的输入序列 $(x_1, \ldots, x_n)$ 映射为一列连续表示 $\mathbf{z} = (z_1, \ldots, z_n)$。给定 $\mathbf{z}$，解码器再逐个元素地生成由符号组成的输出序列 $(y_1, \ldots, y_m)$。模型在每一步都是自回归（auto-regressive）的 [10]：生成下一个符号时，把先前生成的符号作为额外的输入。

Transformer 沿用这一总体架构，编码器与解码器都使用堆叠的自注意力层与逐点的全连接层，分别如图 1 的左半与右半所示。

### 3.1 编码器与解码器堆栈

**编码器：** 编码器由 $N = 6$ 个相同层堆叠而成。每层有两个子层：第一个是多头自注意力机制，第二个是一个简单的、逐位置的全连接前馈网络。我们在每个子层周围都使用残差连接（residual connection）[11]，其后接层归一化（layer normalization）[1]。也就是说，每个子层的输出是 $\mathrm{LayerNorm}(x + \mathrm{Sublayer}(x))$，其中 $\mathrm{Sublayer}(x)$ 是子层本身实现的函数。为了让这些残差连接成立，模型中所有子层以及嵌入层，输出维度都是 $d_{\text{model}} = 512$。

**解码器：** 解码器同样由 $N = 6$ 个相同层堆叠而成。除了编码器层中的两个子层，解码器还插入第三个子层，它对编码器堆栈的输出做多头注意力。与编码器相同，我们在每个子层周围使用残差连接，其后接层归一化。我们还修改了解码器堆栈中的自注意力子层，防止某个位置注意到它后面的位置。这种掩码（masking）再结合输出嵌入偏移一个位置，保证了对位置 $i$ 的预测只能依赖于位置小于 $i$ 的那些已知输出。

### 3.2 注意力

注意力函数可以描述为：把一个查询（query）和一组键—值对（key-value pairs）映射为一个输出，其中查询、键、值与输出都是向量。输出是各值的加权和，赋予每个值的权重由一个相容性函数（compatibility function）根据查询与对应键计算得出。

#### 3.2.1 缩放点积注意力

我们把自己所用的这种注意力称为"缩放点积注意力"（Scaled Dot-Product Attention）（图 2）。输入由维度为 $d_k$ 的查询与键、维度为 $d_v$ 的值组成。我们计算查询与所有键的点积，每个都除以 $\sqrt{d_k}$，再施加 softmax 函数，得到加在各值上的权重。

实际计算时，我们同时对一组查询做注意力函数计算，把它们打包成矩阵 $Q$。键与值也分别打包成矩阵 $K$ 与 $V$。输出矩阵的计算如下：

$$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V \tag{1}$$

（公式说明：$QK^T$ 给出每个查询与每个键的相似度；除以 $\sqrt{d_k}$ 是缩放，防止点积过大把 softmax 推入梯度极小的区域；softmax 沿键的维度把相似度变成权重，再乘 $V$ 得到各值的加权和。）

两种最常用的注意力函数是加性注意力（additive attention）[2] 与点积（乘性）注意力（dot-product / multiplicative attention）。点积注意力与我们的算法相同，区别只在缩放因子 $\frac{1}{\sqrt{d_k}}$。加性注意力用一个带单个隐藏层的前馈网络计算相容性函数。两者理论复杂度相近，但点积注意力在实践中快得多、也省空间得多，因为它可以用高度优化的矩阵乘法代码实现。

当 $d_k$ 取值较小时，两种机制的表现相近；$d_k$ 较大时，不带缩放的点积注意力不如加性注意力 [3]。我们推测，$d_k$ 较大时点积的量级会长得很大，把 softmax 函数推入梯度极小的区域¹。为了抵消这一效应，我们把点积按 $\frac{1}{\sqrt{d_k}}$ 缩放。

（脚注 1：要理解点积为何变大，设 $q$ 与 $k$ 的各分量是相互独立的随机变量，均值为 0、方差为 1。那么它们的点积 $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ 均值为 0、方差为 $d_k$。）

#### 3.2.2 多头注意力

**图 2** （左）缩放点积注意力。（右）多头注意力由若干个并行运行的注意力层组成。（译注：左图的流程是 Q、K 算 MatMul 后除以 $\sqrt{d_k}$、可选掩码、SoftMax，再与 V 做 MatMul；右图把 Q、K、V 各经 h 份不同的线性投影并行做缩放点积注意力，输出拼接后再经一次线性投影。）

我们发现，与其用 $d_{\text{model}}$ 维的键、值与查询只做一次注意力函数，不如把查询、键与值用不同的、学到的线性投影分别投影 $h$ 次到 $d_k$、$d_k$ 与 $d_v$ 维，这样效果更好。

对每一份投影后的查询、键与值，我们并行地执行注意力函数，得到 $d_v$ 维的输出值。这些输出被拼接（concatenated）起来，再做一次投影，得到最终的值，如图 2 所示。

多头注意力（multi-head attention）让模型能在不同位置上同时关注来自不同表示子空间（representation subspaces）的信息。若只有一个注意力头，平均操作会妨碍这一点。

$$\mathrm{MultiHead}(Q, K, V) = \mathrm{Concat}(\mathrm{head}_1, \ldots, \mathrm{head}_h) W^O$$

$$\text{其中}\quad \mathrm{head}_i = \mathrm{Attention}(QW^Q_i, KW^K_i, VW^V_i)$$

（公式说明：每个头先把 Q、K、V 各乘自己的投影矩阵，降到 $d_k$ 或 $d_v$ 维再做公式（1）的缩放点积注意力；$h$ 个头的输出拼接后乘 $W^O$ 投影回 $d_{\text{model}}$ 维。）

其中各投影是参数矩阵 $W^Q_i \in \mathbb{R}^{d_{\text{model}} \times d_k}$、$W^K_i \in \mathbb{R}^{d_{\text{model}} \times d_k}$、$W^V_i \in \mathbb{R}^{d_{\text{model}} \times d_v}$ 与 $W^O \in \mathbb{R}^{hd_v \times d_{\text{model}}}$。

本工作使用 $h = 8$ 个并行的注意力层（即头）。每个头取 $d_k = d_v = d_{\text{model}} / h = 64$。

由于每个头的维度降低了，总计算成本与使用完整维度的单头注意力相近。

#### 3.2.3 注意力在本模型中的三种用法

Transformer 以三种不同方式使用多头注意力：

- 在"编码器—解码器注意力"层中，查询来自前一个解码器层，记忆的键与值来自编码器的输出。这让解码器中的每个位置都能注意到输入序列的所有位置。这模仿了序列到序列模型中典型的编码器—解码器注意力机制，如 [38, 2, 9]。
- 编码器包含自注意力层。在自注意力层中，所有的键、值与查询都来自同一处——在这里就是编码器中前一层的输出。编码器中的每个位置都能注意到编码器前一层的所有位置。
- 与此类似，解码器中的自注意力层让解码器中每个位置注意到解码器中截至该位置（包括该位置）的所有位置。我们需要阻止解码器中的信息向左流动，以保持自回归性质。这一点的实现是在缩放点积注意力内部，把 softmax 输入中对应于非法连接的那些值掩码掉（设为 $-\infty$）。见图 2。

### 3.3 逐位置前馈网络

除了注意力子层，编码器与解码器的每一层都还包含一个全连接前馈网络，它对每个位置分别地、以相同方式施加。这个网络由两个线性变换组成，中间有一个 ReLU 激活：

$$\mathrm{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2 \tag{2}$$

（公式说明：先用 $W_1$ 把 $d_{\text{model}}$ 维升到 $d_{ff}$ 维，过 ReLU，再用 $W_2$ 降回 $d_{\text{model}}$ 维；每个位置用的参数相同，不同层之间参数不同。）

线性变换在不同位置之间是相同的，但不同层之间使用不同的参数。另一种描述方式是：它等价于两次核大小为 1 的卷积。输入与输出维度为 $d_{\text{model}} = 512$，内层维度为 $d_{ff} = 2048$。

### 3.4 嵌入与 Softmax

与其他序列转换模型类似，我们使用学到的嵌入（embeddings）把输入标记（tokens）与输出标记转换为 $d_{\text{model}}$ 维的向量。我们还使用通常的学到的线性变换与 softmax 函数，把解码器输出转换为对下一个标记的预测概率。在我们的模型中，两个嵌入层与 softmax 之前的线性变换共享同一个权重矩阵，与 [30] 类似。在嵌入层中，我们把这些权重乘以 $\sqrt{d_{\text{model}}}$。

### 3.5 位置编码

由于我们的模型既不含循环也不含卷积，为了让模型能利用序列中标记的顺序，必须注入一些关于标记在序列中相对或绝对位置的信息。为此，我们把"位置编码"（positional encodings）加到编码器与解码器堆栈底部的输入嵌入上。位置编码与嵌入维度相同，都是 $d_{\text{model}}$，因此两者可以相加。位置编码有许多种选择，有学到的也有固定的 [9]。

本工作使用不同频率的正弦与余弦函数：

$$PE_{(pos, 2i)} = \sin\left(pos\,/\,10000^{2i/d_{\text{model}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(pos\,/\,10000^{2i/d_{\text{model}}}\right)$$

（公式说明：$pos$ 是位置，$i$ 是维度下标；偶数维用正弦、奇数维用余弦，每个维度对应一条正弦曲线，波长从 $2\pi$ 到 $10000 \cdot 2\pi$ 构成几何级数。）

我们选择这个函数，是因为我们猜想它能让模型容易地学会按相对位置做注意力：对任意固定偏移 $k$，$PE_{pos+k}$ 都可以表示为 $PE_{pos}$ 的线性函数。

我们也试验过改用学到的位置嵌入 [9]，发现两种版本的结果几乎相同（见表 3 的 (E) 行）。我们选择正弦版本，是因为它或许能让模型外推到比训练时遇到的更长的序列长度。

## 4 为什么用自注意力

本节把自注意力层与循环层、卷积层作比较。这两类层常用于把一列符号表示 $(x_1, \ldots, x_n)$ 映射到另一列等长的表示 $(z_1, \ldots, z_n)$，其中 $x_i, z_i \in \mathbb{R}^d$，例如典型序列转换编码器或解码器中的一个隐藏层。我们考虑三项期望性质来说明采用自注意力的理由。

一是每层的总计算复杂度。

二是可并行化的计算量，以所需的最少顺序操作数衡量。

三是网络中长距离依赖之间的路径长度。学习长距离依赖是许多序列转换任务中的关键挑战。影响学习这类依赖能力的一个关键因素，是信号在网络中前向与后向传播所要走的路径长度。输入与输出序列任意两个位置组合之间的路径越短，学长距离依赖就越容易 [12]。因此我们还比较了由不同层类型构成的网络中，任意两个输入与输出位置之间的最大路径长度。

**表 1** 不同层类型的最大路径长度、每层复杂度与最少顺序操作数。$n$ 是序列长度，$d$ 是表示维度，$k$ 是卷积的核大小，$r$ 是受限自注意力中邻域的大小。

| 层类型 | 每层复杂度 | 顺序操作数 | 最大路径长度 |
|---|---|---|---|
| 自注意力 | $O(n^2 \cdot d)$ | $O(1)$ | $O(1)$ |
| 循环 | $O(n \cdot d^2)$ | $O(n)$ | $O(n)$ |
| 卷积 | $O(k \cdot n \cdot d^2)$ | $O(1)$ | $O(\log_k(n))$ |
| 自注意力（受限） | $O(r \cdot n \cdot d)$ | $O(1)$ | $O(n/r)$ |

如表 1 所示，自注意力层用常数个顺序执行的操作就连接所有位置，而循环层需要 $O(n)$ 个顺序操作。

计算复杂度方面，当序列长度 $n$ 小于表示维度 $d$ 时，自注意力层比循环层更快——在机器翻译的最先进模型所用的句子表示中，如词片（word-piece）[38] 与字节对（byte-pair）[31] 表示，这正是最常见的情况。

对于涉及很长序列的任务，为改善计算表现，可以把自注意力限制为只考虑输入序列中以对应输出位置为中心、大小为 $r$ 的一个邻域。这会把最大路径长度增加到 $O(n/r)$。我们计划在今后的工作中进一步研究这一方法。

单个卷积层若核宽 $k < n$，无法连接所有输入与输出位置对。要做到这一点，在连续核的情形下需要堆叠 $O(n/k)$ 个卷积层，在空洞卷积（dilated convolutions）[18] 的情形下需要 $O(\log_k(n))$ 层，从而拉长网络中任意两个位置之间最长的路径。

卷积层通常比循环层贵 $k$ 倍。但可分离卷积（separable convolutions）[6] 把复杂度大幅降到 $O(k \cdot n \cdot d + n \cdot d^2)$。不过，即便取 $k = n$，可分离卷积的复杂度也等于一个自注意力层加上一个逐位置前馈层的组合——这正是我们模型采用的做法。

附带的好处是，自注意力或许能带来可解释性更强的模型。我们检查了自己模型的注意力分布，在附录中给出并讨论了一些例子。各个注意力头不仅明显学会了执行不同的任务，许多头似乎还表现出与句子的句法和语义结构相关的行为。

## 5 训练

本节描述我们模型的训练方案。

### 5.1 训练数据与批处理

我们在标准的 WMT 2014 英德数据集上训练，它包含约 450 万个句子对。句子用字节对编码（byte-pair encoding）[3] 编码，源语言与目标语言共享一个约 37,000 个标记的词表。对英法任务，我们使用规模大得多的 WMT 2014 英法数据集，包含 3,600 万个句子，标记被切分为 32,000 个词片（word-piece）的词表 [38]。句子对按近似的序列长度组批。每个训练批次包含一组句子对，其中约含 25,000 个源标记与 25,000 个目标标记。

### 5.2 硬件与日程

我们的模型在一台有 8 块 NVIDIA P100 GPU 的机器上训练。对使用贯穿全文所述超参数的基础模型，每个训练步约需 0.4 秒。基础模型总共训练 100,000 步，即 12 小时。对我们的大模型（见表 3 最后一行所述），每步时间为 1.0 秒。大模型训练 300,000 步（3.5 天）。

### 5.3 优化器

我们使用 Adam 优化器 [20]，取 $\beta_1 = 0.9$、$\beta_2 = 0.98$、$\epsilon = 10^{-9}$。我们在训练过程中按下面的公式改变学习率：

$$lrate = d_{\text{model}}^{-0.5} \cdot \min\left(step\_num^{-0.5},\ step\_num \cdot warmup\_steps^{-1.5}\right) \tag{3}$$

（公式说明：学习率在最初的 $warmup\_steps$ 步内线性上升，此后按步数的平方根倒数比例下降。$step\_num$ 是当前步数。）

我们取 $warmup\_steps = 4000$。

### 5.4 正则化

训练中我们使用三类正则化：

**残差 Dropout。** 我们对每个子层的输出施加 dropout [33]，施加的时机是在它与子层输入相加并归一化之前。此外，我们还对编码器与解码器堆栈中嵌入与位置编码之和施加 dropout。对基础模型，我们使用的比率为 $P_{drop} = 0.1$。

**标签平滑。** 训练中我们使用取值为 $\epsilon_{ls} = 0.1$ 的标签平滑（label smoothing）[36]。这会损害困惑度（perplexity），因为模型学着变得更不确定，但能提高准确率与 BLEU 分数。

（译注：5.4 节开头原文说"三类正则化"，其下实际只列出"残差 Dropout"与"标签平滑"两个小节标题——这是原文本来如此，译文如实保持。）

## 6 结果

### 6.1 机器翻译

**表 2** Transformer 在英德与英法 newstest2014 测试集上取得比先前最先进模型更好的 BLEU 分数，训练成本却只是它们的一小部分。

| 模型 | BLEU（英德 EN-DE） | BLEU（英法 EN-FR） | 训练成本 FLOPs（英德） | 训练成本 FLOPs（英法） |
|---|---|---|---|---|
| ByteNet [18] | 23.75 |  |  |  |
| Deep-Att + PosUnk [39] |  | 39.2 |  | $1.0 \cdot 10^{20}$ |
| GNMT + RL [38] | 24.6 | 39.92 | $2.3 \cdot 10^{19}$ | $1.4 \cdot 10^{20}$ |
| ConvS2S [9] | 25.16 | 40.46 | $9.6 \cdot 10^{18}$ | $1.5 \cdot 10^{20}$ |
| MoE [32] | 26.03 | 40.56 | $2.0 \cdot 10^{19}$ | $1.2 \cdot 10^{20}$ |
| Deep-Att + PosUnk Ensemble [39] |  | 40.4 |  | $8.0 \cdot 10^{20}$ |
| GNMT + RL Ensemble [38] | 26.30 | 41.16 | $1.8 \cdot 10^{20}$ | $1.1 \cdot 10^{21}$ |
| ConvS2S Ensemble [9] | 26.36 | 41.29 | $7.7 \cdot 10^{19}$ | $1.2 \cdot 10^{21}$ |
| Transformer（基础模型） | 27.3 | 38.1 | $3.3 \cdot 10^{18}$ |  |
| Transformer（大模型） | 28.4 | 41.8 | $2.3 \cdot 10^{19}$ |  |

（译注：表 2 中空缺的格子，是原文表中本来就未给出的数字，译文照原样留空。训练成本只对部分模型估算到英德或英法一侧，另一侧未给。）

在 WMT 2014 英德翻译任务上，大 Transformer 模型（表 2 中的 Transformer (big)）超过此前报告的最好模型（包括集成模型）达 2.0 BLEU 以上，创下 28.4 的新最先进 BLEU 分数。这个模型的配置列于表 3 的最后一行。训练在 8 块 P100 GPU 上花了 3.5 天。即使是我们的基础模型，也超过此前发表的全部模型与集成模型，而训练成本只是那些有竞争力的模型的一小部分。

在 WMT 2014 英法翻译任务上，我们的大模型取得 41.8 的 BLEU 分数（译注：arXiv HTML 版此处正文作 41.0，与摘要及表 2 的 41.8 不一致；译文按摘要与表 2 的 41.8 译出），超过此前发表的全部单模型，而训练成本不到先前最先进模型的四分之一。用于英法任务的 Transformer (big) 模型用了 $P_{drop} = 0.1$ 的 dropout 率，而不是 0.3。

对基础模型，我们用最后一个模型，即对按 10 分钟间隔写出的最后 5 个检查点（checkpoints）取平均得到的单个模型。对大模型，我们对最后 20 个检查点取平均。我们使用束搜索（beam search），束大小为 4，长度惩罚（length penalty）$\alpha = 0.6$ [38]。这些超参数是在开发集上做实验之后选定的。推理时我们把最大输出长度设为输入长度加 50，但在可能时提前终止 [38]。

表 2 汇总了我们的结果，并把我们的翻译质量与训练成本和文献中的其他模型架构作了比较。我们估计训练一个模型所用的浮点运算数的方法是：把训练时间、所用的 GPU 数量，与每块 GPU 持续的单精度浮点运算能力估计值相乘²。

（脚注 2：K80、K40、M40 与 P100 的取值分别为 2.8、3.7、6.0 与 9.5 TFLOPS。）

### 6.2 模型变体

**表 3** Transformer 架构的变体。未列出的取值与基础模型相同。所有指标都在英德翻译开发集 newstest2013 上取得。所列困惑度是按我们的字节对编码计的每词片（per-wordpiece）困惑度，不应与每词（per-word）困惑度相比。（译注：原表是一张多列宽表，列为 $N$、$d_{\text{model}}$、$d_{ff}$、$h$、$d_k$、$d_v$、$P_{drop}$、$\epsilon_{ls}$、训练步数、PPL（dev）、BLEU（dev）、参数量（$\times 10^6$）；(A)–(E) 是原表中的分组标记，同一行只改动所标记的那组参数。译文按分组逐行列出每行实际改动的量与结果，基础模型的完整一行列在最前，其余行未列出的量即与基础模型相同。）

| 行 | 改动的量 | 训练步数 | PPL（dev） | BLEU（dev） | 参数量 $\times 10^6$ |
|---|---|---|---|---|---|
| 基础模型 | $N=6$，$d_{\text{model}}=512$，$d_{ff}=2048$，$h=8$，$d_k=64$，$d_v=64$，$P_{drop}=0.1$，$\epsilon_{ls}=0.1$ | 100K | 4.92 | 25.8 | 65 |
| (A) | $h=1$，$d_k=512$，$d_v=512$ |  | 5.29 | 24.9 |  |
| (A) | $h=4$，$d_k=128$，$d_v=128$ |  | 5.00 | 25.5 |  |
| (A) | $h=16$，$d_k=32$，$d_v=32$ |  | 4.91 | 25.8 |  |
| (A) | $h=32$，$d_k=16$，$d_v=16$ |  | 5.01 | 25.4 |  |
| (B) | $d_k=16$ |  | 5.16 | 25.1 | 58 |
| (B) | $d_k=32$ |  | 5.01 | 25.4 | 60 |
| (C) | $N=2$ |  | 6.11 | 23.7 | 36 |
| (C) | $N=4$ |  | 5.19 | 25.3 | 50 |
| (C) | $N=8$ |  | 4.88 | 25.5 | 80 |
| (C) | $d_{\text{model}}=256$，$d_k=32$，$d_v=32$ |  | 5.75 | 24.5 | 28 |
| (C) | $d_{\text{model}}=1024$，$d_k=128$，$d_v=128$ |  | 4.66 | 26.0 | 168 |
| (C) | $d_{ff}=1024$ |  | 5.12 | 25.4 | 53 |
| (C) | $d_{ff}=4096$ |  | 4.75 | 26.2 | 90 |
| (D) | $P_{drop}=0.0$ |  | 5.77 | 24.6 |  |
| (D) | $P_{drop}=0.2$ |  | 4.95 | 25.5 |  |
| (D) | $\epsilon_{ls}=0.0$ |  | 4.67 | 25.3 |  |
| (D) | $\epsilon_{ls}=0.2$ |  | 5.47 | 25.7 |  |
| (E) | 用位置嵌入代替正弦位置编码 |  | 4.92 | 25.7 |  |
| 大模型 | $N=6$，$d_{\text{model}}=1024$，$d_{ff}=4096$，$h=16$，$P_{drop}=0.3$ | 300K | 4.33 | 26.4 | 213 |

为了评估 Transformer 各组成部分的重要性，我们以不同方式变动基础模型，测量其在英德翻译开发集 newstest2013 上表现的变化。我们使用上一节所述的束搜索，但不做检查点平均。结果见表 3。

在表 3 的 (A) 行中，我们按第 3.2.2 节所述、保持计算量恒定，变动注意力头的数目与注意力键、值的维度。单头注意力比最好的设定差 0.9 BLEU；而头太多时质量也会下降。

在表 3 的 (B) 行中，我们观察到，减小注意力键的大小 $d_k$ 会损害模型质量。这说明确定相容性并不容易，比点积更精细的相容性函数或许有益。我们还在 (C) 行与 (D) 行中观察到，正如预期，更大的模型更好，dropout 对避免过拟合非常有帮助。在 (E) 行中，我们用学到的位置嵌入 [9] 替换正弦位置编码，观察到与基础模型几乎相同的结果。

### 6.3 英语成分句法分析

**表 4** Transformer 能很好地推广到英语成分句法分析（结果在 WSJ 第 23 节上）。

| 分析器 | 训练 | WSJ 23 F1 |
|---|---|---|
| Vinyals & Kaiser et al. (2014) [37] | 仅 WSJ，判别式 | 88.3 |
| Petrov et al. (2006) [29] | 仅 WSJ，判别式 | 90.4 |
| Zhu et al. (2013) [40] | 仅 WSJ，判别式 | 90.4 |
| Dyer et al. (2016) [8] | 仅 WSJ，判别式 | 91.7 |
| Transformer（4 层） | 仅 WSJ，判别式 | 91.3 |
| Zhu et al. (2013) [40] | 半监督 | 91.3 |
| Huang & Harper (2009) [14] | 半监督 | 91.3 |
| McClosky et al. (2006) [26] | 半监督 | 92.1 |
| Vinyals & Kaiser et al. (2014) [37] | 半监督 | 92.1 |
| Transformer（4 层） | 半监督 | 92.7 |
| Luong et al. (2015) [23] | 多任务 | 93.0 |
| Dyer et al. (2016) [8] | 生成式 | 93.3 |

（译注：表 4 中 "Vinyals & Kaiser et al." 的写法是原文如此。）

为了评估 Transformer 能否推广到其他任务，我们在英语成分句法分析（English constituency parsing）上做了实验。这个任务有其特殊的难点：输出受到很强的结构约束，且明显长于输入。

此外，RNN 序列到序列模型在小数据情形下一直无法取得最先进的结果 [37]。

我们在宾夕法尼亚树库（Penn Treebank）[25] 的华尔街日报（Wall Street Journal，WSJ）部分上训练了一个 4 层 Transformer，$d_{\text{model}} = 1024$，训练句子约 40K 条。我们还在半监督设定下训练它，使用规模更大的高置信度语料与 BerkeleyParser 语料，约 1,700 万个句子 [37]。仅 WSJ 的设定使用 16K 个标记的词表，半监督设定使用 32K 个标记的词表。

我们只做了少量实验，来在第 22 节开发集上选择 dropout（包括注意力与残差的，见 5.4 节）、学习率与束大小，其他所有参数与英德基础翻译模型保持不变。推理时，我们把最大输出长度增加到输入长度加 300。在仅 WSJ 与半监督两种设定下，我们都使用束大小 21 与 $\alpha = 0.3$。

表 4 中我们的结果表明，尽管没有针对任务做专门调参，我们的模型表现意外地好，超过此前报告的所有模型，唯一的例外是循环神经网络语法（Recurrent Neural Network Grammar）[8]。

与 RNN 序列到序列模型 [37] 相比，Transformer 即使只在 40K 个句子的 WSJ 训练集上训练，也超过了 BerkeleyParser [29]。

## 7 结论

在本文中，我们提出了 Transformer——第一个完全基于注意力的序列转换模型，用多头自注意力取代了编码器—解码器架构中最常用的循环层。

对翻译任务而言，Transformer 的训练速度能显著快于基于循环层或卷积层的架构。在 WMT 2014 英德与 WMT 2014 英法两个翻译任务上，我们都取得了新的最先进水平。在英德任务上，我们最好的模型甚至超过此前报告的全部集成模型。

我们对基于注意力的模型的未来满怀期待，计划把它们应用到其他任务上。我们计划把 Transformer 扩展到涉及文本之外的输入与输出模态的问题，并研究局部的、受限的注意力机制，以高效处理图像、音频与视频这类大规模的输入与输出。

让生成过程更少地依赖顺序，也是我们的研究目标之一。

我们用于训练与评测模型的代码见 https://github.com/tensorflow/tensor2tensor 。

### 致谢

我们感谢 Nal Kalchbrenner 与 Stephan Gouws 富有成果的评论、修正与启发。

## 参考文献（原文照录）

[1] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. *arXiv preprint arXiv:1607.06450*, 2016.

[2] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. *CoRR*, abs/1409.0473, 2014.

[3] Denny Britz, Anna Goldie, Minh-Thang Luong, and Quoc V. Le. Massive exploration of neural machine translation architectures. *CoRR*, abs/1703.03906, 2017.

[4] Jianpeng Cheng, Li Dong, and Mirella Lapata. Long short-term memory-networks for machine reading. *arXiv preprint arXiv:1601.06733*, 2016.

[5] Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. Learning phrase representations using rnn encoder-decoder for statistical machine translation. *CoRR*, abs/1406.1078, 2014.

[6] Francois Chollet. Xception: Deep learning with depthwise separable convolutions. *arXiv preprint arXiv:1610.02357*, 2016.

[7] Junyoung Chung, Çaglar Gülçehre, Kyunghyun Cho, and Yoshua Bengio. Empirical evaluation of gated recurrent neural networks on sequence modeling. *CoRR*, abs/1412.3555, 2014.

[8] Chris Dyer, Adhiguna Kuncoro, Miguel Ballesteros, and Noah A. Smith. Recurrent neural network grammars. In *Proc. of NAACL*, 2016.

[9] Jonas Gehring, Michael Auli, David Grangier, Denis Yarats, and Yann N. Dauphin. Convolutional sequence to sequence learning. *arXiv preprint arXiv:1705.03122v2*, 2017.

[10] Alex Graves. Generating sequences with recurrent neural networks. *arXiv preprint arXiv:1308.0850*, 2013.

[11] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, pages 770–778, 2016.

[12] Sepp Hochreiter, Yoshua Bengio, Paolo Frasconi, and Jürgen Schmidhuber. Gradient flow in recurrent nets: the difficulty of learning long-term dependencies, 2001.

[13] Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. *Neural computation*, 9(8):1735–1780, 1997.

[14] Zhongqiang Huang and Mary Harper. Self-training PCFG grammars with latent annotations across languages. In *Proceedings of the 2009 Conference on Empirical Methods in Natural Language Processing*, pages 832–841. ACL, August 2009.

[15] Rafal Jozefowicz, Oriol Vinyals, Mike Schuster, Noam Shazeer, and Yonghui Wu. Exploring the limits of language modeling. *arXiv preprint arXiv:1602.02410*, 2016.

[16] Łukasz Kaiser and Samy Bengio. Can active memory replace attention? In *Advances in Neural Information Processing Systems*, (NIPS), 2016.

[17] Łukasz Kaiser and Ilya Sutskever. Neural GPUs learn algorithms. In *International Conference on Learning Representations (ICLR)*, 2016.

[18] Nal Kalchbrenner, Lasse Espeholt, Karen Simonyan, Aaron van den Oord, Alex Graves, and Koray Kavukcuoglu. Neural machine translation in linear time. *arXiv preprint arXiv:1610.10099v2*, 2017.

[19] Yoon Kim, Carl Denton, Luong Hoang, and Alexander M. Rush. Structured attention networks. In *International Conference on Learning Representations*, 2017.

[20] Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In *ICLR*, 2015.

[21] Oleksii Kuchaiev and Boris Ginsburg. Factorization tricks for LSTM networks. *arXiv preprint arXiv:1703.10722*, 2017.

[22] Zhouhan Lin, Minwei Feng, Cicero Nogueira dos Santos, Mo Yu, Bing Xiang, Bowen Zhou, and Yoshua Bengio. A structured self-attentive sentence embedding. *arXiv preprint arXiv:1703.03130*, 2017.

[23] Minh-Thang Luong, Quoc V. Le, Ilya Sutskever, Oriol Vinyals, and Lukasz Kaiser. Multi-task sequence to sequence learning. *arXiv preprint arXiv:1511.06114*, 2015.

[24] Minh-Thang Luong, Hieu Pham, and Christopher D Manning. Effective approaches to attention-based neural machine translation. *arXiv preprint arXiv:1508.04025*, 2015.

[25] Mitchell P Marcus, Mary Ann Marcinkiewicz, and Beatrice Santorini. Building a large annotated corpus of english: The penn treebank. *Computational linguistics*, 19(2):313–330, 1993.

[26] David McClosky, Eugene Charniak, and Mark Johnson. Effective self-training for parsing. In *Proceedings of the Human Language Technology Conference of the NAACL, Main Conference*, pages 152–159. ACL, June 2006.

[27] Ankur Parikh, Oscar Täckström, Dipanjan Das, and Jakob Uszkoreit. A decomposable attention model. In *Empirical Methods in Natural Language Processing*, 2016.

[28] Romain Paulus, Caiming Xiong, and Richard Socher. A deep reinforced model for abstractive summarization. *arXiv preprint arXiv:1705.04304*, 2017.

[29] Slav Petrov, Leon Barrett, Romain Thibaux, and Dan Klein. Learning accurate, compact, and interpretable tree annotation. In *Proceedings of the 21st International Conference on Computational Linguistics and 44th Annual Meeting of the ACL*, pages 433–440. ACL, July 2006.

[30] Ofir Press and Lior Wolf. Using the output embedding to improve language models. *arXiv preprint arXiv:1608.05859*, 2016.

[31] Rico Sennrich, Barry Haddow, and Alexandra Birch. Neural machine translation of rare words with subword units. *arXiv preprint arXiv:1508.07909*, 2015.

[32] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. *arXiv preprint arXiv:1701.06538*, 2017.

[33] Nitish Srivastava, Geoffrey E Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. Dropout: a simple way to prevent neural networks from overfitting. *Journal of Machine Learning Research*, 15(1):1929–1958, 2014.

[34] Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, and Rob Fergus. End-to-end memory networks. In C. Cortes, N. D. Lawrence, D. D. Lee, M. Sugiyama, and R. Garnett, editors, *Advances in Neural Information Processing Systems 28*, pages 2440–2448. Curran Associates, Inc., 2015.

[35] Ilya Sutskever, Oriol Vinyals, and Quoc VV Le. Sequence to sequence learning with neural networks. In *Advances in Neural Information Processing Systems*, pages 3104–3112, 2014.

[36] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. *CoRR*, abs/1512.00567, 2015.

[37] Vinyals & Kaiser, Koo, Petrov, Sutskever, and Hinton. Grammar as a foreign language. In *Advances in Neural Information Processing Systems*, 2015.

[38] Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, et al. Google's neural machine translation system: Bridging the gap between human and machine translation. *arXiv preprint arXiv:1609.08144*, 2016.

[39] Jie Zhou, Ying Cao, Xuguang Wang, Peng Li, and Wei Xu. Deep recurrent models with fast-forward connections for neural machine translation. *CoRR*, abs/1606.04199, 2016.

[40] Muhua Zhu, Yue Zhang, Wenliang Chen, Min Zhang, and Jingbo Zhu. Fast and accurate shift-reduce constituent parsing. In *Proceedings of the 51st Annual Meeting of the ACL (Volume 1: Long Papers)*, pages 434–443. ACL, August 2013.

## 附录 注意力可视化（Attention Visualizations）

**图 3** 一个例子：在 6 层中的第 5 层（layer 5 of 6），注意力机制在编码器自注意力中跟随长距离依赖。这里只显示词 "making" 的注意力。许多注意力头都注意到动词 "making" 的一个远距离依赖，从而补全短语 "making … more difficult"。不同颜色代表不同的头。彩色查看效果最佳。（译注：原图是把该词与其余各词之间的注意力权重画成连线／色块的可视化图，图像本身未嵌入本文件。）

**图 4** 两个注意力头，同样在 6 层中的第 5 层，似乎参与了指代消解（anaphora resolution）。上：头 5 的完整注意力。下：仅就词 "its" 孤立显示头 5 与头 6 的注意力。注意，对这个词的注意力非常集中（sharp）。

**图 5** 许多注意力头表现出的行为似乎与句子的结构有关。上面给出两个这样的例子，来自编码器自注意力 6 层中的第 5 层的两个不同头。这些头明显学会了执行不同的任务。（译注：图 5 在原文中即以这段说明收尾，没有更长的正文。）
