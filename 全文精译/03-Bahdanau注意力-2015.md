# 通过联合学习对齐与翻译的神经机器翻译（全文精译）

## 书目信息

- **原题**：Neural Machine Translation by Jointly Learning to Align and Translate
- **作者**：Dzmitry Bahdanau、Kyunghyun Cho、Yoshua Bengio
- **发表**：《学习表征国际会议》（International Conference on Learning Representations，ICLR）2015 年会议论文集（Conference Track Proceedings），2015 年 5 月 7–9 日，美国圣迭戈
- **arXiv**：arXiv:1409.0473 [cs.CL]；v1 于 2014 年 9 月 1 日提交，最新版 v7 修订于 2016 年 5 月 19 日
- **原文链接（核实过的真实来源链）**：
  - arXiv 摘要页（核对版本信息所据）：https://arxiv.org/abs/1409.0473 （页面载明：Submitted on 1 Sep 2014 (v1), last revised 19 May 2016 (this version, v7)；作者与题名与本条一致）
  - arXiv HTML 全文版（本译文取全文所据版本）：https://arxiv.org/html/1409.0473v7
- **版本说明**：ICLR 2015 会议版与 arXiv 各修订版同属一篇论文；本译文以 arXiv 当前版 v7 的 HTML 全文为底本逐节译出。作者单位按 v7 首页照录（第一作者单位在后期修订版中已更新为雅各布大学，早期版本为蒙特利尔大学，属版本差异，读时留意）。
- **对应导读**：见《AI 顶尖论文谱系》★ 标记篇目（Bahdanau 注意力机制）
- **译本说明**：本译文据上述 HTML 全文逐节译出。正文第 1–7 节、图 1–图 3 图注、表 1–表 3、全部脚注、致谢，以及附录 A（模型架构）、附录 B（训练过程）、附录 C（长句翻译样例）均已译出或按下方说明处理；公式以 LaTeX 呈现并附中文说明；参考文献保留英文原样。原文图 1–图 3 的图像本身未嵌入本文件，图注已全部译出。第 5.2.2 节与附录 C 中的英、法语句子是论文展示的翻译样例本身，属于实验数据，故保留原文语言不译，另以译注说明。译者注一律写作"（译注：……）"，不与原文混排。本文件为 Ezra 个人学习收藏，仅本地存档。

---

## 通过联合学习对齐与翻译的神经机器翻译

**Dzmitry Bahdanau**

雅各布大学（Jacobs University），德国不来梅

**Kyunghyun Cho、Yoshua Bengio†**

蒙特利尔大学（Université de Montréal）

（脚注 †：CIFAR 高级研究员（CIFAR Senior Fellow）。译注：作者单位按 arXiv v7 首页的排版照录；三位作者在论文写作时同属蒙特利尔大学的实验室，v7 中第一作者单位已改为雅各布大学。）

### 摘要

神经机器翻译（neural machine translation）是新近提出的一种机器翻译方法。与传统的统计机器翻译（statistical machine translation）不同，神经机器翻译的目标是构建单个神经网络，并对它做联合调优（jointly tuned），使翻译性能最大化。近来提出的神经机器翻译模型大多属于编码器—解码器（encoder–decoder）这一家族：由一个编码器把源句（source sentence）编码成一个定长向量（fixed-length vector），解码器再从这个向量生成译文。本文提出一个猜想：使用定长向量，是这种基本编码器—解码器架构提升性能的一个瓶颈（bottleneck）。我们建议对它加以扩展——让模型在预测一个目标词时，能自动地（软性地）（(soft-)search）在源句中搜索与之相关的那些部分，而不必把这些部分显式地构成一个硬性的片段（hard segment）。用这一新方法，我们在英译法任务上取得了与现有最先进的基于短语的系统（state-of-the-art phrase-based system）可比的翻译性能。定性分析还表明，模型找到的（软）对齐（(soft-)alignment）与我们的直觉吻合得很好。

## 1 引言

神经机器翻译是新近兴起的一种机器翻译方法，由 Kalchbrenner 与 Blunsom（2013）、Sutskever 等人（2014）和 Cho 等人（2014b）提出。传统的基于短语的翻译系统（Koehn 等人，2003）由许多分别调优的小组件构成；神经机器翻译与之不同，它试图构建并训练一个单一的大型神经网络，读入一个句子，输出正确的译文。

已提出的神经机器翻译模型大多属于编码器—解码器家族（Sutskever 等人，2014；Cho 等人，2014a）：每种语言各有一个编码器和一个解码器；或者对每个句子施加一个语言专属的编码器，再比较它们的输出（Hermann 与 Blunsom，2014）。编码器神经网络读取源句，把它编码成一个定长向量。解码器随后从这个编码向量输出译文。整个编码器—解码器系统——即某个语言对的编码器与解码器——被联合训练，以最大化在给定源句时产出正确译文的概率。

这种编码器—解码器方法有一个潜在问题：神经网络必须把源句的全部必要信息压缩进一个定长向量。对长句来说这可能很难处理，尤其是那些比训练语料中的句子更长的句子。Cho 等人（2014b）已经表明，基本编码器—解码器的性能确实会随着输入句子变长而迅速恶化。

为处理这一问题，我们给编码器—解码器模型引入一个扩展，让它学会联合地对齐与翻译（learn to align and translate jointly）。所提模型每生成译文中的一个词，就（软性地）搜索源句中一个位置集合——最相关的信息集中的地方。随后模型根据与这些源位置相联系的上下文向量（context vectors），以及此前已经生成的全部目标词，来预测下一个目标词。

这一方法与基本编码器—解码器最关键的区别在于：它不试图把整个输入句子编码成单个定长向量。它把输入句子编码成一个向量序列，并在解码译文时自适应地从中选取一个子集。这样，神经翻译模型就不必再把源句的全部信息——无论句子多长——都挤进一个定长向量。我们将表明，这使模型能更好地应对长句。

本文表明，所提的"联合学习对齐与翻译"方法比基本编码器—解码器方法取得了显著更好的翻译性能。句子越长，改进越明显；但任何长度的句子上都能观察到改进。在英译法任务上，所提方法只用单个模型，就取得了与传统的基于短语的系统可比、或接近的翻译性能。定性分析还表明，所提模型在源句与对应目标句之间找到了在语言学上说得通的（软）对齐。

## 2 背景：神经机器翻译

从概率的视角看，翻译等价于：找一个目标句 $\mathbf{y}$，使它在给定源句 $\mathbf{x}$ 时的条件概率最大，即 $\arg\max_{\mathbf{y}} p(\mathbf{y} \mid \mathbf{x})$。在神经机器翻译中，我们用平行训练语料拟合一个参数化模型，使句对的条件概率最大化。翻译模型一旦学到了这个条件分布，给定一个源句，就可以通过搜索条件概率最大的句子来生成对应的译文。

近来已有多篇论文提出用神经网络直接学习这一条件分布（Kalchbrenner 与 Blunsom，2013；Cho 等人，2014a；Sutskever 等人，2014；Cho 等人，2014b；另见 Forcada 与 Ñeco，1997 等）。这种神经机器翻译方法通常由两个部件组成：一个部件把源句 $\mathbf{x}$ 编码，另一个部件解码出目标句 $\mathbf{y}$。例如，Cho 等人（2014a）和 Sutskever 等人（2014）用两个循环神经网络（recurrent neural networks，RNN），把一个变长源句编码成定长向量，再把这个向量解码成变长的目标句。

神经机器翻译虽然还很新，已经显示出有希望的结果。Sutskever 等人（2014）报告说，基于带长短期记忆（long short-term memory，LSTM）单元的 RNN 的神经机器翻译，在英译法任务上取得了接近传统的基于短语的机器翻译系统之最先进性能的成绩¹。

（脚注 1：这里所说的最先进性能，指不使用任何基于神经网络的组件的传统短语系统所达到的性能。把神经组件加入现有翻译系统——例如给短语表中的短语对打分（Cho 等人，2014a），或对候选译文重排序（Sutskever 等人，2014）——已经可以超越此前最先进的性能水平。）

### 2.1 RNN 编码器—解码器

这里简要描述 Cho 等人（2014a）和 Sutskever 等人（2014）提出的底层框架——RNN 编码器—解码器（RNN Encoder–Decoder）。我们提出的新架构、即能同时学会对齐与翻译的架构，就建立在它之上。

在编码器—解码器框架中，编码器读取输入句子——一个向量序列 $\mathbf{x} = (x_1, \cdots, x_{T_x})$——并把它读成一个向量 $c$²。最常见的做法是使用一个 RNN，使

$$h_t = f(x_t, h_{t-1}) \tag{1}$$

（公式说明：$h_t$ 是时刻 $t$ 的隐状态（hidden state）；$x_t$ 是输入序列的第 $t$ 个向量；$f$ 是某个非线性函数。）

并且

$$c = q\left(\{h_1, \cdots, h_{T_x}\}\right),$$

（公式说明：$c$ 是由隐状态序列生成的向量；$h_t \in \mathbb{R}^n$；$f$ 与 $q$ 都是某个非线性函数。例如 Sutskever 等人（2014）用 LSTM 作 $f$，并取 $q(\{h_1, \cdots, h_T\}) = h_T$。）

（脚注 2：此前多数工作（Cho 等人，2014a；Sutskever 等人，2014；Kalchbrenner 与 Blunsom，2013 等）都把变长输入句编码成定长向量；但这并非必需，用变长向量甚至可能更好——后文将表明这一点。）

解码器通常被训练为：在给定上下文向量 $c$ 和此前已预测出的全部词 $\{y_1, \cdots, y_{t'-1}\}$ 时，预测下一个词 $y_{t'}$。换言之，解码器通过把联合概率分解为有序的条件概率，来定义译文 $\mathbf{y}$ 上的一个概率：

$$p(\mathbf{y}) = \prod_{t=1}^{T} p(y_t \mid \{y_1, \cdots, y_{t-1}\}, c), \tag{2}$$

（公式说明：$\mathbf{y} = (y_1, \cdots, y_{T_y})$。）

用 RNN 时，每个条件概率被建模为

$$p(y_t \mid \{y_1, \cdots, y_{t-1}\}, c) = g(y_{t-1}, s_t, c), \tag{3}$$

（公式说明：$g$ 是一个非线性的、可能多层的函数，输出 $y_t$ 的概率；$s_t$ 是 RNN 的隐状态。原文还指出，也可以使用其他架构，例如 RNN 与反卷积神经网络的混合架构（Kalchbrenner 与 Blunsom，2013）。）

## 3 学习对齐与翻译

本节提出一种新的神经机器翻译架构。新架构由一个作编码器的双向 RNN（bidirectional RNN）（见第 3.2 节）和一个解码器构成；这个解码器在解码译文时，模拟在源句中进行搜索的过程（见第 3.1 节）。

### 3.1 解码器：一般描述

**图 1** 所提模型的图示：模型在给定源句 $(x_1, x_2, \dots, x_T)$ 时，尝试生成第 $t$ 个目标词 $y_t$。（译注：图 1 的图像未嵌入本文件；图中左侧是编码器对源句各位置给出的标注 $h_1, \dots, h_{T_x}$，右侧解码器在每一步用对齐权重 $\alpha_{tj}$ 对标注加权得到上下文 $c_t$，再结合上一个词 $y_{t-1}$ 与隐状态生成 $y_t$。）

在新模型架构中，我们把公式（2）中的每个条件概率定义为：

$$p(y_i \mid y_1, \ldots, y_{i-1}, \mathbf{x}) = g(y_{i-1}, s_i, c_i), \tag{4}$$

（公式说明：$s_i$ 是时刻 $i$ 的 RNN 隐状态，按下式计算。）

$$s_i = f(s_{i-1}, y_{i-1}, c_i).$$

要注意，与已有的编码器—解码器方法（见公式（2））不同，这里的概率以一个专属于每个目标词 $y_i$ 的、彼此不同的上下文向量 $c_i$ 为条件。

上下文向量 $c_i$ 依赖于编码器把输入句子映射到的一串标注（annotations）$(h_1, \cdots, h_{T_x})$。每个标注 $h_i$ 都包含关于整个输入序列的信息，但重心强地落在输入序列第 $i$ 个词周围的部分上。标注如何计算，在下一节详细解释。

上下文向量 $c_i$ 随后按这些标注 $h_j$ 的加权和计算：

$$c_i = \sum_{j=1}^{T_x} \alpha_{ij} h_j. \tag{5}$$

（公式说明：这是全篇的核心公式之一——每个目标词都有自己的上下文向量，由全部标注按对齐权重加权求和得到；权重大的源位置对当前目标词的生成影响更大。）

每个标注 $h_j$ 的权重 $\alpha_{ij}$ 由下式计算：

$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x} \exp(e_{ik})}, \tag{6}$$

（公式说明：这是对各源位置的打分 $e_{ij}$ 做 softmax 归一化，使全部 $\alpha_{ij}$ 之和为 1，可以读作目标词 $y_i$ 对齐到源词 $x_j$ 的概率。）

其中

$$e_{ij} = a(s_{i-1}, h_j)$$

（公式说明：$a$ 是一个对齐模型（alignment model），给"位置 $j$ 周围的输入"与"位置 $i$ 处的输出"彼此匹配的程度打分。打分依据是 RNN 隐状态 $s_{i-1}$（即刚要发出 $y_i$ 之前的状态，见公式（4））与输入句子的第 $j$ 个标注 $h_j$。对齐模型的具体函数形式在附录 A.1.2 与 A.2.2 中给出：$e_{ij} = v_a^{\top} \tanh(W_a s_{i-1} + U_a h_j)$，是一个单层前馈网络。）

我们把对齐模型 $a$ 参数化为一个前馈神经网络（feedforward neural network），它与所提系统的所有其他部件联合训练。要注意，与传统机器翻译不同，这里的对齐不被看作隐变量（latent variable）。对齐模型直接计算出一个软对齐（soft alignment），于是代价函数的梯度可以经由它反向传播。这个梯度既可以用来训练对齐模型，也可以用来联合训练整个翻译模型。

把"对全部标注取加权和"这种做法理解为计算一个期望标注（expected annotation）是有益的：期望是对可能的对齐取的。设 $\alpha_{ij}$ 是目标词 $y_i$ 对齐到源词 $x_j$、或说由源词 $x_j$ 翻译而来的概率，那么第 $i$ 个上下文向量 $c_i$ 就是在全部标注上、以 $\alpha_{ij}$ 为概率取的期望标注。

概率 $\alpha_{ij}$、或与它相联系的能量 $e_{ij}$，反映了在决定下一个状态 $s_i$ 并生成 $y_i$ 时，标注 $h_j$ 相对于前一个隐状态 $s_{i-1}$ 的重要程度。直观地说，这就在解码器里实现了一个注意力机制（mechanism of attention）。解码器自己决定要去注意源句的哪些部分。让解码器拥有注意力机制，就把编码器从"必须把源句全部信息编码进一个定长向量"的负担中解放出来。在这一新方法下，信息可以散布在整个标注序列之中，解码器随后可以有选择地把它们取回。

（译注：后世所称的"注意力机制"（attention mechanism）即源于本节这一构造；文中当时用的说法是让解码器"（软性地）搜索"源句，$\alpha_{ij}$ 即注意力权重，$a(\cdot,\cdot)$ 即打分函数。）

### 3.2 编码器：用于标注序列的双向 RNN

通常的 RNN（见公式（1））按顺序读取输入序列 $\mathbf{x}$，从第一个符号 $x_1$ 读到最后一个 $x_{T_x}$。但在所提方案中，我们希望每个词的标注不仅概括它前面的词，也概括它后面的词。

因此我们建议使用双向 RNN（Schuster 与 Paliwal，1997，BiRNN），它近来已在语音识别中成功应用（例如 Graves 等人，2013）。

一个 BiRNN 由一个前向 RNN 和一个后向 RNN 组成。前向 RNN $\overrightarrow{f}$ 按原有顺序（从 $x_1$ 到 $x_{T_x}$）读取输入序列，计算出一串前向隐状态 $(\overrightarrow{h}_1, \cdots, \overrightarrow{h}_{T_x})$。后向 RNN $\overleftarrow{f}$ 按相反顺序（从 $x_{T_x}$ 到 $x_1$）读取序列，得到一串后向隐状态 $(\overleftarrow{h}_1, \cdots, \overleftarrow{h}_{T_x})$。

我们把前向隐状态 $\overrightarrow{h}_j$ 与后向隐状态 $\overleftarrow{h}_j$ 拼接起来，得到词 $x_j$ 的一个标注，即 $h_j = [\overrightarrow{h}_j^{\top}; \overleftarrow{h}_j^{\top}]^{\top}$。这样，标注 $h_j$ 同时包含它前面词与后面词的概括。由于 RNN 倾向于更好地表征较近的输入，标注 $h_j$ 会聚焦在 $x_j$ 周围的词上。这一串标注随后被解码器与对齐模型用来计算上下文向量（公式（5）–（6））。

所提模型的图示见图 1。

## 4 实验设置

我们在英译法任务上评估所提方法。我们使用 ACL WMT '14 提供的双语平行语料³。作为对照，我们还报告 Cho 等人（2014a）新近提出的 RNN 编码器—解码器的性能。两个模型使用相同的训练过程和相同的数据集⁴。

（脚注 3：http://www.statmt.org/wmt14/translation-task.html 。）

（脚注 4：实现见 https://github.com/lisa-groundhog/GroundHog 。）

**图 2** 生成译文在测试集上的 BLEU 分数与句子长度的关系。结果取自完整测试集，其中包括含有模型未知词的句子。（译注：图 2 的图像未嵌入本文件；图中横轴为源句长度分组、纵轴为 BLEU，RNNencdec 的曲线随长度增加急剧下降，RNNsearch-30 与 RNNsearch-50 的曲线明显更平、更高，RNNsearch-50 在 50 词以上仍未见下降。）

### 4.1 数据集

WMT '14 包含以下英法平行语料：Europarl（6100 万词）、新闻评论（news commentary，550 万词）、联合国语料（UN，4.21 亿词），以及两个网络爬取语料，分别为 9000 万词和 2.725 亿词，合计 8.5 亿词。

按照 Cho 等人（2014a）所描述的过程，我们用 Axelrod 等人（2011）的数据选择方法⁵，把合并语料的规模削减到 3.48 亿词。除了上述平行语料，我们不使用任何单语数据——尽管用规模大得多的单语语料预训练一个编码器也许是可行的。我们把 news-test-2012 与 news-test-2013 拼接起来构成开发集（验证集）（development (validation) set），并在 WMT '14 的测试集（news-test-2014）上评估模型；该测试集由 3003 个不在训练数据中出现过的句子组成。

（脚注 5：在线获取地址：http://www-lium.univ-lemans.fr/~schwenk/cslm_joint_paper/ 。）

经过通常的分词（tokenization）⁶之后，我们用每种语言中出现频率最高的 30,000 个词构成的短表（shortlist）来训练模型。不在短表中的任何词都被映射到一个特殊标记（[UNK]）。我们不对数据施加任何其他特殊预处理，例如小写化（lowercasing）或词干化（stemming）。

（脚注 6：分词脚本来自开源机器翻译包 Moses。）

**图 3** RNNsearch-50 找到的四个样例对齐。每个图的横轴与纵轴分别对应源句（英语）与生成译文（法语）中的词。每个像素以灰度显示标注权重 $\alpha_{ij}$——即第 $j$ 个源词的标注对第 $i$ 个目标词的权重（见公式（6））（0 为黑，1 为白）。(a) 一个任意选取的句子。(b–d) 从测试集中随机选出的三个样例，取自不含任何未知词、长度在 10 到 20 个词之间的句子。（译注：图 3 的对齐矩阵图像未嵌入本文件；四个矩阵的强权重总体沿对角线分布，个别短语处出现明显的非对角跳跃，对应第 5.2.1 节讨论的名词—形容词词序差异。）

### 4.2 模型

我们训练两类模型。一类是 RNN 编码器—解码器（Cho 等人，2014a，下称 RNNencdec），另一类是所提模型，下称 RNNsearch。每类模型各训练两次：一次用长度不超过 30 个词的句子（RNNencdec-30、RNNsearch-30），一次用长度不超过 50 个词的句子（RNNencdec-50、RNNsearch-50）。

RNNencdec 的编码器与解码器各有 1000 个隐单元⁷。RNNsearch 的编码器由前向与后向循环神经网络组成，各有 1000 个隐单元；它的解码器有 1000 个隐单元。

（脚注 7：本文中所说的"隐单元"，一律指门控隐单元（gated hidden unit），见附录 A.1.1。）

两种情形下，我们都用一个带单个 maxout（Goodfellow 等人，2013）隐层的多层网络来计算每个目标词的条件概率（Pascanu 等人，2014）。

我们用小批量随机梯度下降（minibatch stochastic gradient descent，SGD）算法结合 Adadelta（Zeiler，2012）来训练每个模型。每个 SGD 更新方向用一个由 80 个句子组成的小批量计算。每个模型训练了大约 5 天。

模型训练完成后，我们用束搜索（beam search）去找一个近似最大化条件概率的译文（例如 Graves，2012；Boulanger-Lewandowski 等人，2013）。Sutskever 等人（2014）也用这种方法从他们的神经机器翻译模型生成译文。

实验所用模型的架构与训练过程的更多细节，见附录 A 与附录 B。

## 5 结果

### 5.1 定量结果

**表 1** 在测试集上计算得到的各已训练模型的 BLEU 分数。第二列与第三列分别是全部句子上的分数，以及源句本身与参考译文中都不含任何未知词的那些句子上的分数。注意，RNNsearch-50⋆ 的训练时间长得多，一直训练到开发集上的性能不再提升为止。（∘）在只评估不含未知词的句子时（最后一列），我们不允许模型生成 [UNK] 标记。

| 模型 | 全部（All） | 无 UNK（No UNK ∘） |
|---|---|---|
| RNNencdec-30 | 13.93 | 24.19 |
| RNNsearch-30 | 21.50 | 31.44 |
| RNNencdec-50 | 17.82 | 26.71 |
| RNNsearch-50 | 26.75 | 34.16 |
| RNNsearch-50⋆ | 28.45 | 36.15 |
| Moses | 33.30 | 35.63 |

（译注：BLEU 分数照原文照录，未做任何改动；"全部"一列含带未知词的句子，因此整体低于"无 UNK"一列。）

表 1 列出了以 BLEU 分数衡量的翻译性能。从表中可以清楚地看到，在所有情形下，所提的 RNNsearch 都优于传统的 RNNencdec。更重要的是，当只考虑由已知词构成的句子时，RNNsearch 的性能与传统的基于短语的翻译系统（Moses）一样高。考虑到 Moses 除了我们用来训练 RNNsearch 与 RNNencdec 的平行语料之外，还额外使用了一个单独的单语语料（4.18 亿词），这是一个相当可观的成果。

所提方法背后的动机之一，正是基本编码器—解码器方法使用了定长上下文向量。我们猜想，这一限制可能使基本编码器—解码器在长句上表现不佳。从图 2 可以看到，RNNencdec 的性能随着句子变长而急剧下降。相比之下，RNNsearch-30 与 RNNsearch-50 对句子长度都稳健得多。尤其是 RNNsearch-50，即便对长度在 50 词或更长的句子，也没有表现出性能退化。RNNsearch-30 甚至还胜过 RNNencdec-50（见表 1），这进一步证实了所提模型对基本编码器—解码器的这一优势。

### 5.2 定性分析

#### 5.2.1 对齐

所提方法给出了一种直观的方式，来检视生成译文中的词与源句中的词之间的（软）对齐。做法是把公式（6）中的标注权重 $\alpha_{ij}$ 可视化，如图 3 所示。每个图中矩阵的每一行，表示与各标注相联系的权重。由此我们可以看到，在生成某个目标词时，源句中的哪些位置被认为更重要。

从图 3 的对齐可以看到，英语词与法语词之间的对齐大体上是单调的（monotonic）。我们看到每个矩阵的对角线上都有强权重。但我们也观察到若干不平凡的、非单调的对齐。形容词与名词在法语和英语中的词序通常不同，图 3 (a) 中就有一个例子。从该图可以看到，模型正确地把短语「European Economic Area」翻译成「zone économique européen」。RNNsearch 能正确地把「zone」与「Area」对齐，跳过两个词（「European」与「Economic」），然后一次回看一个词，补齐整个短语「zone économique européenne」。（译注：原文此处法语短语在正文与 5.2.1 节另一处分别写作 européen 与 européenne，系原文自身的不一致，译文按上下文照录。）

软对齐相对于硬对齐（hard-alignment）的长处，例如可以从图 3 (d) 看出。考虑源短语「the man」，它被翻译成「l'homme」。任何硬对齐都会把「the」映射到「l'」、「man」映射到「homme」。这对翻译没有帮助，因为要决定「the」应当译成「le」「la」「les」还是「l'」，必须考虑它后面的词。我们的软对齐自然地解决了这个问题：它让模型同时看「the」与「man」；在这个例子中我们看到，模型确实正确地把「the」译成了「l'」。在图 3 展示的全部样例中，我们都观察到了类似的行为。软对齐的另一个好处是，它自然地处理长度不同的源短语与目标短语，不需要用一种反直觉的方式把某些词映射到无处（[NULL]）或从无处映射出来（例如见 Koehn，2010，第 4 章与第 5 章）。

#### 5.2.2 长句

从图 2 可以清楚地看到，所提模型（RNNsearch）在翻译长句上比传统模型（RNNencdec）好得多。这很可能是因为，RNNsearch 不需要把一个长句完美地编码成一个定长向量，而只需要准确地编码输入句中某个特定词周围的部分。

举一个例子。考虑测试集中的这个源句（译注：以下英、法语句子是论文展示的翻译样例本身，保留原文语言）：

> An admitting privilege is the right of a doctor to admit a patient to a hospital or a medical centre to carry out a diagnosis or a procedure, based on his status as a health care worker at a hospital.

RNNencdec-50 把这句译为：

> Un privilège d'admission est le droit d'un médecin de reconnaître un patient à l'hôpital ou un centre médical d'un diagnostic ou de prendre un diagnostic en fonction de son état de santé.

RNNencdec-50 正确地翻译了源句直到「a medical center」为止。但从那里起（原文中加下划线的部分），它就偏离了源句的原意。例如，它把源句中的「based on his status as a health care worker at a hospital」替换成了「en fonction de son état de santé」（意为"根据他的健康状况"）。

而 RNNsearch-50 生成了下面这个正确的译文，完整保留了输入句的全部意思，没有遗漏任何细节：

> Un privilège d'admission est le droit d'un médecin d'admettre un patient à un hôpital ou un centre médical pour effectuer un diagnostic ou une procédure, selon son statut de travailleur des soins de santé à l'hôpital.

再看测试集中的另一个句子：

> This kind of experience is part of Disney's efforts to "extend the lifetime of its series and build new relationships with audiences via digital platforms that are becoming ever more important," he added.

RNNencdec-50 给出的译文是：

> Ce type d'expérience fait partie des initiatives du Disney pour "prolonger la durée de vie de ses nouvelles et de développer des liens avec les lecteurs numériques qui deviennent plus complexes.

与前一个例子一样，RNNencdec 在生成大约 30 个词之后开始偏离源句的实际意思（见原文中加下划线的短语）。从那一点起，译文质量退化，出现诸如缺少收尾引号这类基本错误。

RNNsearch-50 同样能正确地翻译这个长句：

> Ce genre d'expérience fait partie des efforts de Disney pour "prolonger la durée de vie de ses séries et créer de nouvelles relations avec des publics via des plateformes numériques de plus en plus importantes", a-t-il ajouté.

结合已经给出的定量结果，这些定性观察证实了我们的假设：RNNsearch 架构在长句上能实现的翻译，可靠性远高于标准的 RNNencdec 模型。

在附录 C 中，我们还给出了几个长源句的样例翻译，分别由 RNNencdec-50、RNNsearch-50 与 Google Translate 生成，并附参考译文。

## 6 相关工作

### 6.1 学习对齐

Graves（2013）新近在手写合成（handwriting synthesis）的背景下，提出了一种把输出符号与输入符号对齐的类似方法。手写合成是要求模型生成给定字符序列之手写体的任务。在他的工作中，他用一组高斯核的混合（mixture of Gaussian kernels）来计算标注的权重，其中每个核的位置、宽度与混合系数都由一个对齐模型预测。更具体地说，他的对齐被限制为只能预测这样的位置：位置是单调递增的。

与我们的方法的主要区别在于，在 Graves（2013）中，标注权重的众数（modes）只能朝一个方向移动。在机器翻译的背景下，这是一个严重的限制，因为生成语法正确的译文常常需要（长距离的）重排序（reordering）（例如英译德）。

我们的方法则要求：对译文中的每个词，都计算源句中每个词的标注权重。当输入、输出句子大多只有 15–40 个词时——翻译任务正是如此——这个缺点并不严重。但它可能限制所提方案在其他任务上的适用性。

### 6.2 用于机器翻译的神经网络

自 Bengio 等人（2003）提出神经概率语言模型（neural probabilistic language model）——用神经网络建模一个词在给定固定数目前文词时的条件概率——以来，神经网络已被广泛用于机器翻译。不过，神经网络的角色在很大程度上局限于：给现有统计机器翻译系统提供单个特征，或者对现有系统给出的候选译文列表做重排序。

例如，Schwenk（2012）提出用一个前馈神经网络计算一对源短语与目标短语的分数，并把这个分数用作基于短语的统计机器翻译系统中的一个额外特征。更近一些，Kalchbrenner 与 Blunsom（2013）和 Devlin 等人（2014）报告了把神经网络用作现有翻译系统的一个子组件并取得成功。传统上，作为目标侧语言模型训练的神经网络被用来对候选译文列表重打分或重排序（例如 Schwenk 等人，2006）。

上述方法已被证明能把翻译性能提升到当时的最先进水平之上；但我们感兴趣的是一个更有野心的目标：设计一个全新的、基于神经网络的翻译系统。因此，本文所考虑的神经机器翻译方法，是与这些早期工作的彻底决裂。我们的模型不把神经网络用作现有系统的一部分，而是独立工作，直接从源句生成译文。

## 7 结论

神经机器翻译的传统方法，称为编码器—解码器方法，把整个输入句子编码成一个定长向量，再从中解码出译文。我们基于 Cho 等人（2014b）与 Pouget-Abadie 等人（2014）新近报告的实证研究提出猜想：使用定长上下文向量，在翻译长句时是有问题的。

本文提出了一种处理这一问题的新架构。我们扩展了基本编码器—解码器：让模型在生成每个目标词时，（软性地）搜索一组输入词、或由编码器计算出的它们的标注。这样，模型就不必把整个源句编码进一个定长向量，也可以只聚焦于与生成下一个目标词相关的信息。这对神经机器翻译系统在较长句子上给出好结果的能力，有重大的正面影响。与传统机器翻译系统不同，翻译系统的全部部件——包括对齐机制——都被联合训练，朝着"产出正确译文的对数概率更高"这一目标优化。

我们在英译法任务上测试了所提模型（称为 RNNsearch）。实验表明，所提的 RNNsearch 显著优于传统的编码器—解码器模型（RNNencdec），与句子长度无关；并且它对源句长度的稳健性强得多。在定性分析中，我们考察了 RNNsearch 生成的（软）对齐，由此可以得出结论：模型在生成正确译文的同时，能把每个目标词与源句中相关的词、或它们的标注正确地对齐起来。

也许更重要的是，所提方法取得了与现有的基于短语的统计机器翻译可比的翻译性能。考虑到所提架构、乃至整个神经机器翻译家族，都只是今年才刚被提出来的，这是一个引人注目的结果。我们相信，这里提出的架构，是朝着更好的机器翻译、以及更好地理解自然语言迈出的有希望的一步。

留待未来处理的一个挑战，是更好地处理未知词、或罕见词。模型要得到更广泛的使用、并在一切背景下都达到当前最先进机器翻译系统的性能，这一点是必需的。

## 致谢

作者感谢 Theano 的开发者们（Bergstra 等人，2010；Bastien 等人，2012）。我们感谢以下机构在研究经费与计算支持上的资助：NSERC、Calcul Québec、Compute Canada、加拿大研究讲席（Canada Research Chairs）与 CIFAR。Bahdanau 感谢 Planet Intelligent Systems GmbH 的支持。我们还感谢 Felix Hill、Bart van Merriénboer、Jean Pouget-Abadie、Coline Devin 与 Tae-Ho Kim。

## 参考文献（原文照录）

Axelrod, A., He, X., and Gao, J. (2011). Domain adaptation via pseudo in-domain data selection. In *Proceedings of the ACL Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pages 355–362. Association for Computational Linguistics.

Bastien, F., Lamblin, P., Pascanu, R., Bergstra, J., Goodfellow, I. J., Bergeron, A., Bouchard, N., and Bengio, Y. (2012). Theano: new features and speed improvements. *Deep Learning and Unsupervised Feature Learning NIPS 2012 Workshop*.

Bengio, Y., Simard, P., and Frasconi, P. (1994). Learning long-term dependencies with gradient descent is difficult. *IEEE Transactions on Neural Networks*, 5(2), 157–166.

Bengio, Y., Ducharme, R., Vincent, P., and Janvin, C. (2003). A neural probabilistic language model. *J. Mach. Learn. Res.*, 3, 1137–1155.

Bergstra, J., Breuleux, O., Bastien, F., Lamblin, P., Pascanu, R., Desjardins, G., Turian, J., Warde-Farley, D., and Bengio, Y. (2010). Theano: a CPU and GPU math expression compiler. In *Proceedings of the Python for Scientific Computing Conference (SciPy)*. Oral Presentation.

Boulanger-Lewandowski, N., Bengio, Y., and Vincent, P. (2013). Audio chord recognition with recurrent neural networks. In *ISMIR*.

Cho, K., van Merrienboer, B., Gulcehre, C., Bougares, F., Schwenk, H., and Bengio, Y. (2014a). Learning phrase representations using RNN encoder-decoder for statistical machine translation. In *Proceedings of the Empiricial Methods in Natural Language Processing (EMNLP 2014)*. to appear.

Cho, K., van Merriënboer, B., Bahdanau, D., and Bengio, Y. (2014b). On the properties of neural machine translation: Encoder–Decoder approaches. In *Eighth Workshop on Syntax, Semantics and Structure in Statistical Translation*. to appear.

Devlin, J., Zbib, R., Huang, Z., Lamar, T., Schwartz, R., and Makhoul, J. (2014). Fast and robust neural network joint models for statistical machine translation. In *Association for Computational Linguistics*.

Forcada, M. L. and Ñeco, R. P. (1997). Recursive hetero-associative memories for translation. In J. Mira, R. Moreno-Díaz, and J. Cabestany, editors, *Biological and Artificial Computation: From Neuroscience to Technology*, volume 1240 of *Lecture Notes in Computer Science*, pages 453–462. Springer Berlin Heidelberg.

Goodfellow, I., Warde-Farley, D., Mirza, M., Courville, A., and Bengio, Y. (2013). Maxout networks. In *Proceedings of The 30th International Conference on Machine Learning*, pages 1319–1327.

Graves, A. (2012). Sequence transduction with recurrent neural networks. In *Proceedings of the 29th International Conference on Machine Learning (ICML 2012)*.

Graves, A. (2013). Generating sequences with recurrent neural networks. *arXiv: 1308.0850 [cs.NE]*.

Graves, A., Jaitly, N., and Mohamed, A.-R. (2013). Hybrid speech recognition with deep bidirectional LSTM. In *Automatic Speech Recognition and Understanding (ASRU), 2013 IEEE Workshop on*, pages 273–278.

Hermann, K. and Blunsom, P. (2014). Multilingual distributed representations without word alignment. In *Proceedings of the Second International Conference on Learning Representations (ICLR 2014)*.

Hochreiter, S. (1991). *Untersuchungen zu dynamischen neuronalen Netzen*. Diploma thesis, Institut für Informatik, Lehrstuhl Prof. Brauer, Technische Universität München.

Hochreiter, S. and Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735–1780.

Kalchbrenner, N. and Blunsom, P. (2013). Recurrent continuous translation models. In *Proceedings of the ACL Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pages 1700–1709. Association for Computational Linguistics.

Koehn, P. (2010). *Statistical Machine Translation*. Cambridge University Press, New York, NY, USA.

Koehn, P., Och, F. J., and Marcu, D. (2003). Statistical phrase-based translation. In *Proceedings of the 2003 Conference of the North American Chapter of the Association for Computational Linguistics on Human Language Technology - Volume 1, NAACL '03*, pages 48–54, Stroudsburg, PA, USA. Association for Computational Linguistics.

Pascanu, R., Mikolov, T., and Bengio, Y. (2013a). On the difficulty of training recurrent neural networks. In *ICML'2013*.

Pascanu, R., Mikolov, T., and Bengio, Y. (2013b). On the difficulty of training recurrent neural networks. In *Proceedings of the 30th International Conference on Machine Learning (ICML 2013)*.

Pascanu, R., Gulcehre, C., Cho, K., and Bengio, Y. (2014). How to construct deep recurrent neural networks. In *Proceedings of the Second International Conference on Learning Representations (ICLR 2014)*.

Pouget-Abadie, J., Bahdanau, D., van Merriënboer, B., Cho, K., and Bengio, Y. (2014). Overcoming the curse of sentence length for neural machine translation using automatic segmentation. In *Eighth Workshop on Syntax, Semantics and Structure in Statistical Translation*. to appear.

Schuster, M. and Paliwal, K. K. (1997). Bidirectional recurrent neural networks. *Signal Processing, IEEE Transactions on*, 45(11), 2673–2681.

Schwenk, H. (2012). Continuous space translation models for phrase-based statistical machine translation. In M. Kay and C. Boitet, editors, *Proceedings of the 24th International Conference on Computational Linguistics (COLIN)*, pages 1071–1080. Indian Institute of Technology Bombay.

Schwenk, H., Dchelotte, D., and Gauvain, J.-L. (2006). Continuous space language models for statistical machine translation. In *Proceedings of the COLING/ACL on Main conference poster sessions*, pages 723–730. Association for Computational Linguistics.

Sutskever, I., Vinyals, O., and Le, Q. (2014). Sequence to sequence learning with neural networks. In *Advances in Neural Information Processing Systems (NIPS 2014)*.

Zeiler, M. D. (2012). ADADELTA: An adaptive learning rate method. *arXiv: 1212.5701 [cs.LG]*.

## 附录 A 模型架构

### A.1 架构选择

第 3 节所提的方案是一个通用框架，其中例如 RNN 的激活函数 $f$ 与对齐模型 $a$ 都可以自由定义。这里描述我们在本文实验中所做的选择。

#### A.1.1 循环神经网络

对 RNN 的激活函数 $f$，我们使用 Cho 等人（2014a）新近提出的门控隐单元（gated hidden unit）。门控隐单元是常规简单单元（例如逐元素 $\tanh$）的一个替代。它与 Hochreiter 与 Schmidhuber（1997）更早提出的长短期记忆（LSTM）单元相似，共同点是能更好地建模并学习长期依赖（long-term dependencies）。其实现方式是：在展开后的 RNN 中，存在一些计算路径，沿这些路径导数的乘积接近 1。这些路径让梯度容易地向后流动，不至于过多地遭受梯度消失（vanishing）效应之苦（Hochreiter，1991；Bengio 等人，1994；Pascanu 等人，2013a）。因此，也可以使用 LSTM 单元来代替这里描述的门控隐单元，正如 Sutskever 等人（2014）在类似背景下所做的那样。

采用 $n$ 个门控隐单元的 RNN，其新状态 $s_i$⁸ 按下式计算：

（脚注 8：这里给出的是解码器的公式。同一个公式同样可用于编码器，只需忽略上下文向量 $c_i$ 及与之相关的项。）

$$s_i = f(s_{i-1}, y_{i-1}, c_i) = (1 - z_i) \circ s_{i-1} + z_i \circ \tilde{s}_i,$$

（公式说明：$\circ$ 表示逐元素乘法；$z_i$ 是更新门（update gates）的输出，见下文。）

提议的更新状态 $\tilde{s}_i$ 由下式计算：

$$\tilde{s}_i = \tanh\left(W e(y_{i-1}) + U [r_i \circ s_{i-1}] + C c_i\right),$$

（公式说明：$e(y_{i-1}) \in \mathbb{R}^m$ 是词 $y_{i-1}$ 的一个 $m$ 维嵌入（embedding）；$r_i$ 是重置门（reset gates）的输出，见下文。当 $y_i$ 表示为 1-of-$K$ 向量时，$e(y_i)$ 就是嵌入矩阵 $E \in \mathbb{R}^{m \times K}$ 的一列。为使公式不至于太杂乱，在可能之处我们省略偏置项。）

更新门 $z_i$ 让每个隐单元能保持它先前的激活，重置门 $r_i$ 控制先前状态中有多少、以及哪些信息应当被重置。它们按下式计算：

$$z_i = \sigma\left(W_z e(y_{i-1}) + U_z s_{i-1} + C_z c_i\right),$$

$$r_i = \sigma\left(W_r e(y_{i-1}) + U_r s_{i-1} + C_r c_i\right),$$

（公式说明：$\sigma(\cdot)$ 是逻辑斯蒂 sigmoid 函数。（译注：这组门控公式即后来通称的 GRU（门控循环单元，gated recurrent unit）的结构，本附录原文只称"门控隐单元"。））

在解码器的每一步，我们把输出概率（公式（4））计算为一个多层函数（Pascanu 等人，2014）。我们使用一个由 maxout 单元构成的单隐层，并用 softmax 函数对输出概率（每个词一个）做归一化（见公式（6））。

#### A.1.2 对齐模型

设计对齐模型时，要考虑到：对每个长度分别为 $T_x$ 与 $T_y$ 的句对，模型需要被求值 $T_x \times T_y$ 次。为了减少计算量，我们使用一个单层多层感知机（multilayer perceptron），使

$$a(s_{i-1}, h_j) = v_a^{\top} \tanh\left(W_a s_{i-1} + U_a h_j\right),$$

（公式说明：这就是对齐模型打分函数的具体形式，后世称为"加性注意力"（additive attention）的打分方式。其中 $W_a \in \mathbb{R}^{n \times n}$、$U_a \in \mathbb{R}^{n \times 2n}$、$v_a \in \mathbb{R}^n$ 都是权重矩阵。）

由于 $U_a h_j$ 不依赖于 $i$，可以预先计算它，以把计算成本降到最低。

### A.2 模型的详细描述

#### A.2.1 编码器

本节详细描述实验中使用的所提模型（RNNsearch）的架构（见第 4–5 节）。从这里起，为便于阅读，我们省略全部偏置项。

模型以一个由 1-of-K 编码的词向量组成的源句为输入

$$\mathbf{x} = (x_1, \ldots, x_{T_x}), \quad x_i \in \mathbb{R}^{K_x}$$

并输出一个由 1-of-K 编码的词向量组成的译句

$$\mathbf{y} = (y_1, \ldots, y_{T_y}), \quad y_i \in \mathbb{R}^{K_y},$$

其中 $K_x$ 与 $K_y$ 分别是源语言与目标语言的词表大小。$T_x$ 与 $T_y$ 分别表示源句与目标句的长度。

计算双向循环神经网络（BiRNN）的前向状态：

$$\overrightarrow{h}_i = \begin{cases} (1 - \overrightarrow{z}_i) \circ \overrightarrow{h}_{i-1} + \overrightarrow{z}_i \circ \overrightarrow{\underline{h}}_i, & \text{若 } i > 0 \\ 0, & \text{若 } i = 0 \end{cases}$$

其中

$$\overrightarrow{\underline{h}}_i = \tanh\left(\overrightarrow{W} \overline{E} x_i + \overrightarrow{U} [\overrightarrow{r}_i \circ \overrightarrow{h}_{i-1}]\right)$$

$$\overrightarrow{z}_i = \sigma\left(\overrightarrow{W}_z \overline{E} x_i + \overrightarrow{U}_z \overrightarrow{h}_{i-1}\right)$$

$$\overrightarrow{r}_i = \sigma\left(\overrightarrow{W}_r \overline{E} x_i + \overrightarrow{U}_r \overrightarrow{h}_{i-1}\right).$$

（公式说明：$\overline{E} \in \mathbb{R}^{m \times K_x}$ 是词嵌入矩阵。$\overrightarrow{W}, \overrightarrow{W}_z, \overrightarrow{W}_r \in \mathbb{R}^{n \times m}$，$\overrightarrow{U}, \overrightarrow{U}_z, \overrightarrow{U}_r \in \mathbb{R}^{n \times n}$ 是权重矩阵。$m$ 与 $n$ 分别是词嵌入维度与隐单元数。$\sigma(\cdot)$ 一如通常，是逻辑斯蒂 sigmoid 函数。）

后向状态 $(\overleftarrow{h}_1, \cdots, \overleftarrow{h}_{T_x})$ 以类似方式计算。

前向与后向 RNN 共用词嵌入矩阵 $\overline{E}$，但权重矩阵不共用。

我们拼接前向与后向状态，得到标注 $(h_1, h_2, \cdots, h_{T_x})$，其中

$$h_i = \begin{bmatrix} \overrightarrow{h}_i \\ \overleftarrow{h}_i \end{bmatrix}$$

#### A.2.2 解码器

给定来自编码器的标注，解码器的隐状态 $s_i$ 按下式计算：

$$s_i = (1 - z_i) \circ s_{i-1} + z_i \circ \tilde{s}_i,$$

其中

$$\tilde{s}_i = \tanh\left(W E y_{i-1} + U [r_i \circ s_{i-1}] + C c_i\right)$$

$$z_i = \sigma\left(W_z E y_{i-1} + U_z s_{i-1} + C_z c_i\right)$$

$$r_i = \sigma\left(W_r E y_{i-1} + U_r s_{i-1} + C_r c_i\right)$$

（公式说明：$E$ 是目标语言的词嵌入矩阵。$W, W_z, W_r \in \mathbb{R}^{n \times m}$，$U, U_z, U_r \in \mathbb{R}^{n \times n}$，$C, C_z, C_r \in \mathbb{R}^{n \times 2n}$ 是权重。$m$ 与 $n$ 分别是词嵌入维度与隐单元数。）

初始隐状态 $s_0$ 由 $s_0 = \tanh\left(W_s \overleftarrow{h}_1\right)$ 计算，其中 $W_s \in \mathbb{R}^{n \times n}$。

上下文向量 $c_i$ 在每一步都由对齐模型重新计算：

$$c_i = \sum_{j=1}^{T_x} \alpha_{ij} h_j,$$

其中

$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x} \exp(e_{ik})}$$

$$e_{ij} = v_a^{\top} \tanh\left(W_a s_{i-1} + U_a h_j\right),$$

（公式说明：$h_j$ 是源句中的第 $j$ 个标注（见公式（A.2.1））。$v_a \in \mathbb{R}^{n'}$、$W_a \in \mathbb{R}^{n' \times n}$、$U_a \in \mathbb{R}^{n' \times 2n}$ 是权重矩阵。原文注：若把 $c_i$ 固定为 $\overrightarrow{h}_{T_x}$，模型就退化为 RNN 编码器—解码器（Cho 等人，2014a）。）

有了解码器状态 $s_{i-1}$、上下文 $c_i$ 与上一个生成的词 $y_{i-1}$，我们把目标词 $y_i$ 的概率定义为

$$p(y_i \mid s_i, y_{i-1}, c_i) \propto \exp\left(y_i^{\top} W_o t_i\right),$$

其中

$$t_i = [\max\{\tilde{t}_{i,2j-1}, \tilde{t}_{i,2j}\}]_{j=1, \ldots, l}^{\top}$$

而 $\tilde{t}_{i,k}$ 是向量 $\tilde{t}_i$ 的第 $k$ 个元素，$\tilde{t}_i$ 由下式计算：

$$\tilde{t}_i = U_o s_{i-1} + V_o E y_{i-1} + C_o c_i.$$

（公式说明：$W_o \in \mathbb{R}^{K_y \times l}$、$U_o \in \mathbb{R}^{2l \times n}$、$V_o \in \mathbb{R}^{2l \times m}$、$C_o \in \mathbb{R}^{2l \times 2n}$ 是权重矩阵。这可以理解为带单个 maxout 隐层的深度输出（deep output）（Pascanu 等人，2014；maxout 见 Goodfellow 等人，2013）。）

#### A.2.3 模型规模

对本文使用的全部模型，隐层大小 $n$ 为 1000，词嵌入维度 $m$ 为 620，深度输出中 maxout 隐层的大小 $l$ 为 500。对齐模型中的隐单元数 $n'$ 为 1000。

**表 2** 学习统计与相关信息。每次更新（update）对应使用单个小批量更新一次参数。一个轮次（epoch）指完整遍历一遍训练集。NLL 是训练集或开发集中句子的平均条件对数概率。注意，各集合中句子的长度不同。（译注：表中模型名 RNNenc 是原文的写法，与正文的 RNNencdec 指同一类模型。）

| 模型 | 更新次数（×10⁵） | 轮次（Epochs） | 小时（Hours） | GPU | 训练 NLL | 开发 NLL |
|---|---|---|---|---|---|---|
| RNNenc-30 | 8.46 | 6.4 | 109 | TITAN BLACK | 28.1 | 53.0 |
| RNNenc-50 | 6.00 | 4.5 | 108 | Quadro K-6000 | 44.0 | 43.6 |
| RNNsearch-30 | 4.71 | 3.6 | 113 | TITAN BLACK | 26.7 | 47.2 |
| RNNsearch-50 | 2.88 | 2.2 | 111 | Quadro K-6000 | 40.7 | 38.1 |
| RNNsearch-50⋆ | 6.67 | 5.0 | 252 | Quadro K-6000 | 36.7 | 35.2 |

## 附录 B 训练过程

### B.1 参数初始化

我们把循环权重矩阵 $U, U_z, U_r, \overleftarrow{U}, \overleftarrow{U}_z, \overleftarrow{U}_r, \overrightarrow{U}, \overrightarrow{U}_z$ 与 $\overrightarrow{U}_r$ 初始化为随机正交矩阵（random orthogonal matrices）。

对 $W_a$ 与 $U_a$，我们通过从均值为 0、方差为 $0.001^2$ 的高斯分布中采样每个元素来初始化。$V_a$ 的全部元素与全部偏置向量都初始化为零。任何其他权重矩阵，都通过从均值为 0、方差为 $0.01^2$ 的高斯分布中采样来初始化。

（译注：原文此处写作 $V_a$，与 A.1.2 中的 $v_a$ 应为同一组参数，系原文记号不一致，译文照录。）

### B.2 训练

我们使用随机梯度下降（SGD）算法。用 Adadelta（Zeiler，2012）自动适配每个参数的学习率（$\epsilon = 10^{-6}$，$\rho = 0.95$）。每当我们计算时，都显式地把代价函数梯度的 $L_2$ 范数归一化到至多为预先定下的阈值 1——当范数大于该阈值时才这样做（Pascanu 等人，2013b）。每个 SGD 更新方向用一个由 80 个句子组成的小批量计算。

在每次更新时，我们的实现所需时间正比于小批量中最长句子的长度。因此，为把计算浪费降到最低，在每第 20 次更新之前，我们取出 1600 个句对，按长度排序，并把它们切成 20 个小批量。训练数据在训练开始前被打乱一次，随后以这种方式顺序遍历。

表 2 给出了与训练实验所用全部模型有关的统计信息。

## 附录 C 长句的翻译

（译注：本附录的表 3 是翻译样例本身——源句为英语，各系统输出与参考译文为法语。样例的价值正在于对照原文，故整表保留原文语言不译；表注译为中文。）

**表 3** RNNenc-50 与 RNNsearch-50 对从测试集中选出的长源句（30 词或以上）生成的翻译。对每个源句，我们同时给出黄金标准（gold-standard）译文。Google Translate 的译文制作于 2014 年 8 月 27 日。

| 项目 | 内容 |
|---|---|
| Source | An admitting privilege is the right of a doctor to admit a patient to a hospital or a medical centre to carry out a diagnosis or a procedure, based on his status as a health care worker at a hospital. |
| Reference | Le privilège d'admission est le droit d'un médecin, en vertu de son statut de membre soignant d'un hôpital, d'admettre un patient dans un hôpital ou un centre médical afin d'y délivrer un diagnostic ou un traitement. |
| RNNenc-50 | Un privilège d'admission est le droit d'un médecin de reconnaître un patient à l'hôpital ou un centre médical d'un diagnostic ou de prendre un diagnostic en fonction de son état de santé. |
| RNNsearch-50 | Un privilège d'admission est le droit d'un médecin d'admettre un patient à un hôpital ou un centre médical pour effectuer un diagnostic ou une procédure, selon son statut de travailleur des soins de santé à l'hôpital. |
| Google Translate | Un privilège admettre est le droit d'un médecin d'admettre un patient dans un hôpital ou un centre médical pour effectuer un diagnostic ou une procédure, fondée sur sa situation en tant que travailleur de soins de santé dans un hôpital. |

| 项目 | 内容 |
|---|---|
| Source | This kind of experience is part of Disney's efforts to "extend the lifetime of its series and build new relationships with audiences via digital platforms that are becoming ever more important," he added. |
| Reference | Ce type d'expérience entre dans le cadre des efforts de Disney pour "étendre la durée de vie de ses séries et construire de nouvelles relations avec son public grâce à des plateformes numériques qui sont de plus en plus importantes", a-t-il ajouté. |
| RNNenc-50 | Ce type d'expérience fait partie des initiatives du Disney pour "prolonger la durée de vie de ses nouvelles et de développer des liens avec les lecteurs numériques qui deviennent plus complexes. |
| RNNsearch-50 | Ce genre d'expérience fait partie des efforts de Disney pour "prolonger la durée de vie de ses séries et créer de nouvelles relations avec des publics via des plateformes numériques de plus en plus importantes", a-t-il ajouté. |
| Google Translate | Ce genre d'expérience fait partie des efforts de Disney à "étendre la durée de vie de sa série et construire de nouvelles relations avec le public par le biais des plates-formes numériques qui deviennent de plus en plus important", at-il ajouté. |

| 项目 | 内容 |
|---|---|
| Source | In a press conference on Thursday, Mr Blair stated that there was nothing in this video that might constitute a "reasonable motive" that could lead to criminal charges being brought against the mayor. |
| Reference | En conférence de presse, jeudi, M. Blair a affirmé qu'il n'y avait rien dans cette vidéo qui puisse constituer des "motifs raisonnables" pouvant mener au dépôt d'une accusation criminelle contre le maire. |
| RNNenc-50 | Lors de la conférence de presse de jeudi, M. Blair a dit qu'il n'y avait rien dans cette vidéo qui pourrait constituer une "motivation raisonnable" pouvant entraîner des accusations criminelles portées contre le maire. |
| RNNsearch-50 | Lors d'une conférence de presse jeudi, M. Blair a déclaré qu'il n'y avait rien dans cette vidéo qui pourrait constituer un "motif raisonnable" qui pourrait conduire à des accusations criminelles contre le maire. |
| Google Translate | Lors d'une conférence de presse jeudi, M. Blair a déclaré qu'il n'y avait rien dans cette vido qui pourrait constituer un "motif raisonnable" qui pourrait mener à des accusations criminelles portes contre le maire. |

（译注：原文至此结束。附录 C 的法语句子为各系统在 2014 年的实际输出，其中个别拼写与重音符号的缺失——例如 Google Translate 输出中的「vido」「portes」——都是原文照录的系统错误，不是译文笔误。）
