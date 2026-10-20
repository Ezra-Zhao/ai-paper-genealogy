# 用深度卷积神经网络做 ImageNet 分类（全文精译）

## 书目信息

- **原题**：ImageNet Classification with Deep Convolutional Neural Networks
- **作者**：Alex Krizhevsky、Ilya Sutskever、Geoffrey E. Hinton
- **单位**：多伦多大学（University of Toronto），计算机科学系
- **发表**：《神经信息处理系统进展》（Advances in Neural Information Processing Systems，NIPS，现称 NeurIPS）第 25 卷，2012 年，论文编号 4824，页 1097–1105
- **原文链接（核实过的真实来源链）**：
  - NeurIPS 官方 proceedings 摘要页：https://proceedings.neurips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html （该页为会议早期稿的摘要存档，数字为 top-1 39.7%、top-5 18.9%，与下述正式 PDF 定稿数字不同，见译注）
  - NeurIPS 官方 proceedings 正式 PDF：https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
  - 多伦多大学作者主页托管本：http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf （Hinton 课题组 Fritz 主页镜像，与正式 PDF 正文一致）
  - 本译文实际取全文所据版本：杜克大学课程站托管的同一正式 PDF 镜像（https://courses.cs.duke.edu/cps274/compsci527/cps274/compsci527/spring19/papers/Krizhevsky.pdf ，逐页文本与上述正式 PDF 一致，已与 proceedings 摘要页交叉核对书目信息）
- **arXiv 说明**：该文没有 arXiv 版本，不要与 2017 年《美国计算机协会通讯》（Communications of the ACM）同题重刊版混淆；本译文译的是 2012 年 NIPS 定稿。
- **对应导读**：见《AI 顶尖论文谱系》第二卷《深度学习革命》AlexNet 篇（★）
- **译本说明**：本译文据上述正式 PDF（共 9 页）逐节译出。正文第 1–7 节、图 1–图 4 图注、表 1–表 2、脚注 1–5 全部译出；公式以 LaTeX 呈现并附中文说明；参考文献 [1]–[26] 保留英文原样。原文没有单独的致谢（Acknowledgements）一节，也没有附录——正文之后即讨论与参考文献，译文如实保持这一结构，不增补。原文图 1–图 4 的图像本身未嵌入本文件，图注已全部译出，图内容在图注译文中概述。译者注一律写作“（译注：……）”，不与原文混排。本文件为 Ezra 个人学习收藏，仅本地存档。

（译注：proceedings 摘要页存档的数字 39.7%/18.9%、神经元数 500,000、“两个全局连接层”等表述，属于投稿早期稿；正式出版的 PDF 定稿改为 top-1 37.5%、top-5 17.0%、650,000 个神经元、三个全连接层，并补入了 ILSVRC-2012 的 15.3% 结果。两处不一致是版本差异，不是笔误，读时以 PDF 定稿为准。）

---

## 用深度卷积神经网络做 ImageNet 分类

**Alex Krizhevsky、Ilya Sutskever、Geoffrey E. Hinton**

多伦多大学，多伦多大学，多伦多大学
kriz@cs.utoronto.ca，ilya@cs.utoronto.ca，hinton@cs.utoronto.ca

### 摘要

我们训练了一个大型的深度卷积神经网络（deep convolutional neural network），把 ImageNet LSVRC-2010 竞赛中的 120 万张高分辨率图像分到 1000 个不同类别中。在测试数据上，我们取得了 top-1 错误率 37.5%、top-5 错误率 17.0%，明显优于此前的最佳水平（state-of-the-art）。这个神经网络有 6000 万个参数和 650,000 个神经元，由五个卷积层（convolutional layers）构成——其中一部分后面跟着最大池化层（max-pooling layers）——再加三个全连接层（fully-connected layers），末端是一个 1000 路的 softmax。为了让训练更快，我们使用了非饱和神经元（non-saturating neurons）和一个非常高效的卷积运算 GPU 实现。为了减轻全连接层的过拟合（overfitting），我们采用了一种新近提出的正则化（regularization）方法，叫“dropout”，效果被证明非常好。我们还把这个模型的一个变体送去参加 ILSVRC-2012 竞赛，取得了获胜的 top-5 测试错误率 15.3%，第二名的成绩是 26.2%。

（译注：top-1 错误率指模型概率最高的那个预测错了的比例；top-5 错误率指正确标签没有进入模型认为最可能的五个标签之内的比例。下文不再逐处加注。）

## 1 引言

当前的目标识别（object recognition）方法在本质上都离不开机器学习方法。要提升它们的表现，可以收集更大的数据集、学习更强大的模型，并使用更好的防过拟合技术。直到不久前，带标签的图像数据集还相对较小——大约是几万张图像的量级（例如 NORB [16]、Caltech-101/256 [8, 9] 和 CIFAR-10/100 [12]）。用这个规模的数据集，简单的识别任务已经可以做得相当好，尤其是再用保持标签不变的变换（label-preserving transformations）去扩增数据之后。例如，MNIST 数字识别任务上目前最好的错误率（<0.3%）已经接近人类水平 [4]。

但真实场景中的物体表现出相当大的可变性，所以要学会识别它们，必须使用大得多的训练集。事实上，小型图像数据集的短处已经得到广泛承认（例如 Pinto 等人 [21]），但直到最近，收集带有上百万张图像的带标签数据集才成为可能。新的、更大的数据集包括 LabelMe [23]——由几十万张完整分割好的图像组成——和 ImageNet [6]——由 22,000 多个类别中超过 1500 万张带标签的高分辨率图像组成。

要从上百万张图像中学习几千种物体，我们需要一个学习容量（learning capacity）很大的模型。然而，目标识别任务极其复杂，这意味着即便是 ImageNet 这样大的数据集也无法把问题完全定死，所以我们的模型还应当带有大量的先验知识（prior knowledge），来弥补我们没有的那些数据。卷积神经网络（convolutional neural networks，CNNs）正是这样一类模型 [16, 11, 13, 18, 15, 22, 26]。它们的容量可以通过改变深度和宽度来控制，而且它们对图像的性质也作出了强有力且大体正确的假设（即统计量的平稳性（stationarity of statistics）和像素依赖的局部性（locality of pixel dependencies））。因此，与层规模相近的标准前馈神经网络相比，CNN 的连接和参数要少得多，训练起来更容易，而它们理论上能达到的最佳性能大概只略差一点。

尽管 CNN 有这些吸引人的性质，尽管其局部结构相对高效，要在高分辨率图像上大规模地应用它们，代价此前仍然高得令人却步。所幸，当前的 GPU 配上高度优化的二维卷积实现，已经强大到足以支撑起有趣规模的 CNN 的训练；而 ImageNet 这类新近出现的数据集，包含的带标签样本也已经足够多，可以在不出现严重过拟合的情况下训练这样的模型。

本文的具体贡献如下：我们在 ILSVRC-2010 和 ILSVRC-2012 竞赛 [2] 所用的 ImageNet 子集上，训练了迄今为止最大的卷积神经网络之一，并在这两个数据集上取得了迄今为止报道过的、遥遥领先的最佳结果。我们写了一个高度优化的卷积神经网络训练 GPU 实现——包括二维卷积在内的全部运算——并把它公开出来¹。我们的网络包含若干新的、不寻常的特征，它们提升了性能、缩短了训练时间，细节在第 3 节中说明。网络的规模使过拟合成为一个显著的问题，即便有 120 万个带标签的训练样本也不例外，所以我们使用了几种有效的防过拟合技术，见第 4 节。我们的最终网络包含五个卷积层和三个全连接层，而这个深度看来很重要：我们发现，去掉任何一个卷积层（每个卷积层包含的参数都不超过模型总参数的 1%），都会导致性能变差。

归根结底，网络的规模主要受限于当前 GPU 上可用的显存量，以及我们愿意容忍的训练时长。我们的网络在两块 GTX 580 3GB GPU 上训练需要五到六天。我们所有的实验都表明：只要等到更快的 GPU 和更大的数据集出现，我们的结果就能得到改进。

（脚注 1：http://code.google.com/p/cuda-convnet/ 。译注：该 Google Code 链接现已失效，代码后世通称 cuda-convnet。）

## 2 数据集

ImageNet 是一个包含超过 1500 万张带标签高分辨率图像的数据集，归属大约 22,000 个类别。这些图像从网络上收集而来，由人类标注者使用亚马逊 Mechanical Turk 众包工具打上标签。从 2010 年起，作为帕斯卡视觉目标挑战赛（Pascal Visual Object Challenge）的一部分，一项名为 ImageNet 大规模视觉识别挑战赛（ImageNet Large-Scale Visual Recognition Challenge，ILSVRC）的年度竞赛开始举办。ILSVRC 使用 ImageNet 的一个子集，每个类别大约 1000 张图像，分属 1000 个类别。总计大约有 120 万张训练图像、50,000 张验证图像和 150,000 张测试图像。

ILSVRC-2010 是唯一一个测试集标签公开可得的 ILSVRC 版本，所以我们的大多数实验都是在这个版本上做的。由于我们也把模型送去参加了 ILSVRC-2012 竞赛，第 6 节中我们还会报告在这个版本数据集上的结果，它的测试集标签不可得。在 ImageNet 上，惯例是报告两种错误率：top-1 和 top-5，其中 top-5 错误率是测试图像中正确标签没有出现在模型认为最可能的五个标签之内的那部分所占的比例。

ImageNet 中的图像分辨率各不相同，而我们的系统要求输入维度恒定。因此，我们把图像下采样（down-sampled）到固定的 256×256 分辨率。具体做法是：给定一张矩形图像，先把它缩放，使其较短的一边长度为 256，再从缩放后的图像中裁出中央的 256×256 块。除了从每个像素中减去训练集上的平均活动（mean activity）之外，我们不对图像做任何其他预处理。所以，我们是在（已中心化的）像素原始 RGB 值上训练网络的。

## 3 架构

我们网络的架构概括在图 2 中。它包含八个带权重的层——前五个是卷积层，其余三个是全连接层。下面我们描述这个网络架构中一些新颖或不寻常的特征。第 3.1–3.4 节按我们估计的重要性排序，最重要的排在最前面。

### 3.1 ReLU 非线性

把神经元的输出 $f$ 建模为其输入 $x$ 的函数，标准做法是用 $f(x) = \tanh(x)$ 或 $f(x) = (1 + e^{-x})^{-1}$。

$$f(x) = \tanh(x) \quad \text{或} \quad f(x) = \frac{1}{1 + e^{-x}}$$

（公式说明：前者是双曲正切，后者是逻辑斯蒂 sigmoid 函数；两者都是饱和非线性（saturating nonlinearities），输入绝对值变大时输出趋于平坦、梯度趋于零。）

就梯度下降的训练时间而言，这些饱和非线性比非饱和非线性（non-saturating nonlinearity）$f(x) = \max(0, x)$ 要慢得多。

$$f(x) = \max(0, x)$$

（公式说明：输入为负时输出 0，为正时原样输出；这就是修正线性单元的激活函数。）

沿用 Nair 与 Hinton [20] 的叫法，我们把使用这种非线性的神经元称为修正线性单元（Rectified Linear Units，ReLUs）。带 ReLU 的深度卷积神经网络，训练速度比带 tanh 单元的等价网络快好几倍。图 1 展示了这一点：图中给出，在 CIFAR-10 数据集上，一个特定的四层卷积网络达到 25% 训练错误率所需的迭代次数。如果没有使用 ReLU，我们在这项工作中根本不可能拿这么大的神经网络做实验。

**图 1** 一个带 ReLU 的四层卷积网络（实线）在 CIFAR-10 上达到 25% 训练错误率的速度，比带 tanh 神经元的等价网络（虚线）快六倍。每个网络的学习率都是各自独立选取的，以使训练尽可能快。没有使用任何形式的正则化。这里展示的效应大小会随网络架构而变化，但带 ReLU 的网络始终比带饱和神经元的等价网络学习快好几倍。（译注：图 1 是一条训练误差随迭代次数下降的曲线图，横轴为迭代次数、纵轴为训练错误率；实线明显先于虚线到达 25% 这条水平线。）

我们并不是最早在 CNN 中考虑传统神经元模型替代方案的人。例如，Jarrett 等人 [11] 声称，非线性 $f(x) = |\tanh(x)|$ 在他们的对比度归一化（contrast normalization）接局部平均池化的做法下，在 Caltech-101 数据集上效果特别好。然而，在那个数据集上，首要关切是防止过拟合，所以他们观察到的效应，与我们这里报告的使用 ReLU 时拟合训练集被加速的效应不是一回事。更快的学习，对在大数据集上训练的大模型表现有着很大的影响。

### 3.2 在多块 GPU 上训练

单块 GTX 580 GPU 只有 3GB 显存，这限制了能在它上面训练的网络的最大规模。事实证明，120 万个训练样本足以训练大到一块 GPU 装不下的网络。因此，我们把网络摊到两块 GPU 上。当前的 GPU 特别适合跨 GPU 并行，因为它们能够直接读写彼此的显存，不必经过主机内存。我们采用的并行方案，实质上是把一半的核（kernels，或神经元）放在每块 GPU 上，并多用了一个技巧：GPU 之间只在某些层通信。这意味着，举例来说，第 3 层的核从第 2 层的所有核图（kernel maps）取输入，而第 4 层的核只从位于同一块 GPU 上的第 3 层核图取输入。选择连接模式是一个需要交叉验证（cross-validation）的问题，但这让我们可以精确地调整通信量，直到它占计算量的比例达到可接受的程度。

由此得到的架构，与 Cireşan 等人 [5] 使用的“柱状”CNN 有几分相似，区别在于我们的柱子并不独立（见图 2）。与在单块 GPU 上训练、每个卷积层核数减半的网络相比，这个方案把 top-1 和 top-5 错误率分别降低了 1.7% 和 1.2%。双 GPU 网络的训练时间比单 GPU 网络略短²。

（脚注 2：单 GPU 网络在最后一个卷积层中，实际上与双 GPU 网络拥有相同数量的核。这是因为网络的大部分参数位于第一个全连接层中，而该层以最后一个卷积层为输入。所以为了让两个网络的参数量大致相同，我们没有把最后一个卷积层的规模减半（其后的全连接层也没有减半）。因此，这个比较对单 GPU 网络有利，因为它比双 GPU 网络的“一半规模”要大。）

### 3.3 局部响应归一化

ReLU 有一个可取的性质：它们不需要对输入做归一化来防止饱和。只要至少有一些训练样本能给一个 ReLU 带来正输入，学习就会在那个神经元中发生。不过，我们仍然发现，下面这种局部归一化方案有助于泛化（generalization）。用 $a^i_{x,y}$ 表示在位置 $(x, y)$ 施加核 $i$ 再施加 ReLU 非线性之后算出的神经元活动，那么响应归一化后的活动 $b^i_{x,y}$ 由下式给出：

$$b^i_{x,y} = a^i_{x,y} \Big/ \left(k + \alpha \sum_{j=\max(0,\, i-n/2)}^{\min(N-1,\, i+n/2)} \left(a^j_{x,y}\right)^2 \right)^{\beta}$$

（公式说明：求和是在同一空间位置上“相邻”的核图上进行的，$N$ 是该层核的总数。核图的排序是任意的，在训练开始之前就确定下来。这种响应归一化实现了一种侧抑制（lateral inhibition）——其灵感来自真实神经元中发现的那种形式——它让用不同核计算出的神经元输出之间，为大的活动值展开竞争。常数 $k$、$n$、$\alpha$ 和 $\beta$ 是超参数（hyper-parameters），其取值用验证集确定；我们使用 $k = 2$、$n = 5$、$\alpha = 10^{-4}$、$\beta = 0.75$。我们在某些层中、施加 ReLU 非线性之后应用这种归一化（见第 3.5 节）。原文公式中的求和上下限在 PDF 排版中写作 $\min(N-1, i+n/2)$ 与 $\max(0, i-n/2)$，译文按此呈现。）

这个方案与 Jarrett 等人 [11] 的局部对比度归一化（local contrast normalization）方案有几分相似，但我们的方案更准确地讲应当叫“亮度归一化”（brightness normalization），因为我们没有减去平均活动。响应归一化把我们的 top-1 和 top-5 错误率分别降低了 1.4% 和 1.2%。我们也在 CIFAR-10 数据集上验证了这个方案的有效性：一个四层 CNN 在没有归一化时取得 13% 的测试错误率，有归一化时取得 11%³。

（脚注 3：由于篇幅所限，我们无法详细描述这个网络，但它由这里提供的代码和参数文件精确指定：http://code.google.com/p/cuda-convnet/ 。）

### 3.4 重叠池化

CNN 中的池化层（pooling layers）汇总同一张核图中相邻神经元组的输出。传统上，相邻池化单元所汇总的邻域互不重叠（例如 [17, 11, 4]）。说得更精确些，一个池化层可以看作由一张池化单元网格构成，单元之间相隔 $s$ 个像素，每个单元汇总一个以该池化单元位置为中心、大小为 $z \times z$ 的邻域。如果令 $s = z$，就得到 CNN 中常用的传统局部池化；如果令 $s < z$，就得到重叠池化（overlapping pooling）。这正是我们在整个网络中使用的方案，取 $s = 2$、$z = 3$。与输出维度等价的非重叠方案 $s = 2$、$z = 2$ 相比，这个方案把 top-1 和 top-5 错误率分别降低了 0.4% 和 0.3%。我们在训练中普遍观察到，带重叠池化的模型稍微更难过拟合一些。

### 3.5 整体架构

现在我们可以描述 CNN 的整体架构了。如图 2 所示，这个网络包含八个带权重的层；前五个是卷积层，其余三个是全连接层。最后一个全连接层的输出送入一个 1000 路的 softmax，它产生一个在 1000 个类别标签上的分布。我们的网络最大化的是多项逻辑回归（multinomial logistic regression）目标，这等价于在训练样本上最大化正确标签在预测分布下的对数概率的平均值。

第二、第四和第五个卷积层的核，只连接到前一层中位于同一块 GPU 上的那些核图（见图 2）。第三个卷积层的核连接到第二层的所有核图。全连接层中的神经元连接到前一层的所有神经元。响应归一化层跟在第一和第二个卷积层之后。第 3.4 节所描述的那种最大池化层，跟在两个响应归一化层之后，也跟在第五个卷积层之后。ReLU 非线性被施加到每一个卷积层和全连接层的输出上。

**图 2** 我们 CNN 架构的示意图，明确画出了两块 GPU 之间的职责划分。一块 GPU 运行图中上半部分的层部件，另一块运行下半部分的层部件。GPU 只在某些层通信。网络的输入是 150,528 维的，其余各层中神经元的数量依次为 253,440–186,624–64,896–64,896–43,264–4096–4096–1000。（译注：150,528 = 224×224×3；图中网络自左向右横向展开，上、下两条通路分别对应两块 GPU，在第三个卷积层与全连接层处汇合。）

第一个卷积层用 96 个大小为 11×11×3 的核，以 4 个像素的步幅（stride）（译注：步幅是核图中相邻神经元感受野中心之间的距离）对 224×224×3 的输入图像做滤波。第二个卷积层以第一个卷积层（经响应归一化和池化后的）输出为输入，用 256 个大小为 5×5×48 的核对它做滤波。第三、第四和第五个卷积层彼此相连，中间没有任何池化层或归一化层。第三个卷积层有 384 个大小为 3×3×256 的核，连接到第二个卷积层（经归一化、池化后的）输出。第四个卷积层有 384 个大小为 3×3×192 的核，第五个卷积层有 256 个大小为 3×3×192 的核。全连接层各有 4096 个神经元。


## 4 减轻过拟合

我们的神经网络架构有 6000 万个参数。尽管 ILSVRC 的 1000 个类别使每个训练样本对从图像到标签的映射施加了 10 比特的约束，事实证明，这仍不足以在没有相当程度过拟合的情况下学会这么多参数。下面我们描述我们对抗过拟合的两种主要办法。

### 4.1 数据扩增

减轻图像数据过拟合最简单、也最常用的办法，是用保持标签不变的变换人工地扩大数据集（例如 [25, 4, 5]）。我们采用了两种不同形式的数据扩增（data augmentation），两者都能以很少的计算从原始图像生成变换后的图像，所以变换后的图像不需要存到磁盘上。在我们的实现中，变换后的图像由 CPU 上的 Python 代码生成，与此同时 GPU 正在前一批图像上训练。所以这些数据扩增方案实际上是不花计算成本的。

第一种数据扩增，是生成图像平移和水平翻转。我们的做法是，从 256×256 的图像中抽取随机的 224×224 块（以及它们的水平翻转），并在这些抽取出的块上训练网络⁴。这把训练集的规模扩大了 2048 倍，尽管由此得到的训练样本当然是高度相互依赖的。没有这个方案，我们的网络会遭受严重的过拟合，那就迫使我们使用小得多的网络。在测试时，网络通过抽取五个 224×224 块（四个角上的块和中央块）以及它们的水平翻转（因此总共十个块）来作出预测，并对网络的 softmax 层在这十个块上作出的预测取平均。

（脚注 4：这就是图 2 中输入图像是 224×224×3 维的原因。）

第二种数据扩增，是改变训练图像中 RGB 通道的强度。具体做法是，我们对整个 ImageNet 训练集上的 RGB 像素值集合做主成分分析（PCA）。对每张训练图像，我们加上所找到的主成分的若干倍，其大小正比于相应的特征值，再乘以一个从均值为零、标准差为 0.1 的高斯分布中抽取的随机变量。因此，对每个 RGB 图像像素 $I^{xy} = [I^{xy}_R, I^{xy}_G, I^{xy}_B]^T$，我们加上下面这个量：

$$\left[\mathbf{p}_1, \mathbf{p}_2, \mathbf{p}_3\right] \left[\alpha_1 \lambda_1, \alpha_2 \lambda_2, \alpha_3 \lambda_3\right]^T$$

（公式说明：$\mathbf{p}_i$ 与 $\lambda_i$ 分别是 RGB 像素值 3×3 协方差矩阵的第 $i$ 个特征向量（eigenvector）和特征值（eigenvalue），$\alpha_i$ 是前面所说的那个随机变量。每个 $\alpha_i$ 对一张特定的训练图像的所有像素只抽取一次，直到该图像再次被用于训练时才重新抽取。）

这个方案大致抓住了自然图像的一个重要性质：物体的身份不随照明强度和颜色的变化而改变。这个方案把 top-1 错误率降低了 1% 以上。

### 4.2 Dropout

把许多不同模型的预测组合起来，是降低测试误差的一种非常成功的办法 [1, 3]，但对于已经需要训练好几天的大型神经网络来说，这样做似乎太昂贵了。不过，有一种非常高效的模型组合版本，训练时只多花大约两倍的成本。这一新近提出的技术叫“dropout” [10]，其做法是以 0.5 的概率把每个隐藏神经元的输出置为零。以这种方式被“丢弃”（dropped out）的神经元不参与前向传递（forward pass），也不参与反向传播（back-propagation）。于是，每呈现一个输入，神经网络就采样出一个不同的架构，但所有这些架构共享权重。这种技术减少了神经元之间复杂的相互适应（co-adaptations），因为一个神经元不能依赖其他特定神经元的存在。因此，它被迫去学习更鲁棒的特征——这些特征在与许多不同的、其他神经元的随机子集协同工作时都有用。在测试时，我们使用全部神经元，但把它们的输出乘以 0.5，这是对指数级多个 dropout 网络所产生的预测分布取几何平均的一个合理近似。

我们在图 2 的前两个全连接层中使用 dropout。没有 dropout，我们的网络表现出严重的过拟合。Dropout 使收敛所需的迭代次数大致翻倍。

## 5 学习的细节

我们使用随机梯度下降（stochastic gradient descent）训练模型，批大小（batch size）为 128 个样本，动量（momentum）为 0.9，权重衰减（weight decay）为 0.0005。我们发现，这么小量的权重衰减对模型的学习很重要。换句话说，此处的权重衰减不只是一个正则化项：它降低了模型的训练误差。权重 $w$ 的更新规则是：

$$v_{i+1} := 0.9 \cdot v_i - 0.0005 \cdot \epsilon \cdot w_i - \epsilon \cdot \left\langle \frac{\partial L}{\partial w} \Big|_{w_i} \right\rangle_{D_i}$$

$$w_{i+1} := w_i + v_{i+1}$$

（公式说明：$i$ 是迭代下标，$v$ 是动量变量，$\epsilon$ 是学习率（learning rate），$\left\langle \frac{\partial L}{\partial w} |_{w_i} \right\rangle_{D_i}$ 是目标函数关于 $w$ 的导数在 $w_i$ 处的取值在第 $i$ 个批 $D_i$ 上的平均值。）

**图 3** 第一个卷积层在 224×224×3 输入图像上学到的 96 个大小为 11×11×3 的卷积核。上面 48 个核是在 GPU 1 上学到的，下面 48 个核是在 GPU 2 上学到的。详见第 6.1 节。（译注：图 3 是 96 个小方块拼成的网格图，每个小方块是一个 11×11 的彩色滤波器图案。）

我们用均值为零、标准差为 0.01 的高斯分布初始化每一层的权重。我们把第二、第四和第五个卷积层以及全连接隐藏层中神经元的偏置（biases）初始化为常数 1。这种初始化通过给 ReLU 提供正输入，加速了学习的早期阶段。我们把其余层中神经元的偏置初始化为常数 0。我们对所有层使用相同的学习率，并在整个训练过程中手动调整它。我们遵循的启发式规则是：当验证错误率在当前学习率下不再改善时，就把学习率除以 10。学习率初始值设为 0.01，在终止前共减小了三次。我们在 120 万张图像的训练集上训练了大约 90 个循环（cycles），在两块 NVIDIA GTX 580 3GB GPU 上花了五到六天。

## 6 结果

我们在 ILSVRC-2010 上的结果汇总在表 1 中。我们的网络取得了 top-1 测试集错误率 37.5% 和 top-5 测试集错误率 17.0%⁵。ILSVRC-2010 竞赛期间取得的最佳成绩是 47.1% 和 28.2%，所用方法是对在不同特征上训练的六个稀疏编码（sparse-coding）模型所作的预测取平均 [2]；此后发表的最佳结果是 45.7% 和 25.7%，所用方法是对两个分类器的预测取平均，这两个分类器是在从两种密集采样特征算出的 Fisher 向量（Fisher Vectors，FVs）上训练的 [24]。

**表 1** ILSVRC-2010 测试集上的结果比较。斜体是他人取得的最佳结果。

| 模型 | Top-1 | Top-5 |
|---|---|---|
| 稀疏编码 [2] | 47.1% | 28.2% |
| SIFT + FVs [24] | *45.7%* | *25.7%* |
| CNN（本文） | **37.5%** | **17.0%** |

（脚注 5：不按第 4.1 节所述对十个块的预测取平均时，错误率是 39.0% 和 18.3%。）

我们还把模型送去参加了 ILSVRC-2012 竞赛，结果报告在表 2 中。由于 ILSVRC-2012 的测试集标签不公开，我们无法报告我们尝试过的所有模型的测试错误率。在本段余下部分中，我们把验证错误率和测试错误率交替使用，因为按我们的经验，两者相差不超过 0.1%（见表 2）。本文描述的这个 CNN 取得了 18.2% 的 top-5 错误率。对五个类似 CNN 的预测取平均，错误率为 16.4%。训练一个 CNN，在最后一个池化层之上多加第六个卷积层，先让它对整个 ImageNet 2011 年秋季发布版（1500 万张图像、22,000 个类别）做分类，再在 ILSVRC-2012 上对它做“微调”（fine-tuning），得到 16.6% 的错误率。把两个在整个 2011 年秋季发布版上预训练过的 CNN 的预测，与前面提到的五个 CNN 的预测一起取平均，得到 15.3% 的错误率。第二名的参赛作品取得了 26.2% 的错误率，其方法是对若干分类器的预测取平均，这些分类器是在从不同类型的密集采样特征算出的 FV 上训练的 [7]。

**表 2** ILSVRC-2012 验证集和测试集上的错误率比较。斜体是他人取得的最佳结果。带星号 * 的模型经过了“预训练”，即先学会对整个 ImageNet 2011 年秋季发布版做分类。详见第 6 节。

| 模型 | Top-1（验证） | Top-5（验证） | Top-5（测试） |
|---|---|---|---|
| SIFT + FVs [7] | — | — | *26.2%* |
| 1 个 CNN | 40.7% | 18.2% | — |
| 5 个 CNN | 38.1% | 16.4% | 16.4% |
| 1 个 CNN* | 39.0% | 16.6% | — |
| 7 个 CNN* | 36.7% | 15.4% | **15.3%** |

最后，我们还报告在 ImageNet 2009 年秋季版本上的错误率，该版本有 10,184 个类别和 890 万张图像。在这个数据集上，我们沿用文献中的惯例，用一半图像训练、一半图像测试。由于没有既定的测试集，我们的划分必然与先前作者所用的划分不同，但这不会对结果有可察觉的影响。我们在这个数据集上的 top-1 和 top-5 错误率分别为 67.4% 和 40.9%，是由上面描述的那个网络、但在最后一个池化层之上多加了第六个卷积层后取得的。这个数据集上已发表的最佳结果是 78.1% 和 60.9% [19]。

### 6.1 定性评估

图 3 展示了网络的两个数据连接层所学到的卷积核。这个网络学到了各种各样的频率与方向选择性（frequency- and orientation-selective）的核，以及各种彩色斑块。注意两块 GPU 所表现出的分工，这是第 3.5 节所述受限连接方式的结果。GPU 1 上的核大体上与颜色无关，而 GPU 2 上的核大体上是颜色特异的。这种分工在每一次运行中都会出现，并且与任何特定的随机权重初始化无关（只差 GPU 重新编号）。

**图 4** （左）八张 ILSVRC-2010 测试图像，以及我们的模型认为最可能的五个标签。每张图像下方写着正确标签，赋予正确标签的概率也用一根红条标出（如果它恰好进入了前五名）。（右）第一列是五张 ILSVRC-2010 测试图像。其余各列展示的是六张训练图像——它们的最后一个隐藏层特征向量，与测试图像的特征向量之间的欧氏距离最小。（译注：左图的八个例子在原文图像中依次含螨虫（mite）、蘑菇、樱桃、豹（leopard）、格栅（grille）等；红条长度表示概率大小。右图的检索结果按行排列，每行第一张是查询图、后面六张是检索到的训练图。）

在图 4 的左半部分，我们通过计算网络在八张测试图像上的 top-5 预测，定性地评估网络学到了什么。注意，即便是偏离中心的目标，例如左上角的螨虫，也能被网络识别出来。大多数 top-5 标签看起来都合理。例如，对豹来说，被认为可能的标签只有其他类型的猫科动物。在某些情况下（格栅、樱桃），关于照片的预期焦点究竟是什么，确实存在歧义。

探究网络视觉知识的另一种办法，是考察一张图像在最后一个 4096 维隐藏层上诱导出的特征激活。如果两张图像产生的特征激活向量之间欧氏距离很小，我们就可以说，神经网络的较高层认为它们是相似的。图 4 展示了测试集中的五张图像，以及按这一度量与它们各自最相似的六张训练集图像。注意，在像素层面上，检索到的训练图像与第一列的查询图像在 L2 距离上一般并不接近。例如，检索到的狗和大象以各种姿势出现。我们在补充材料（supplementary material）中给出了更多测试图像的结果。（译注：补充材料未随正式 PDF 正文一同收录，本译文无此部分可译，特此注明。）

用两个 4096 维实值向量之间的欧氏距离来计算相似度，效率很低，但可以通过训练一个自编码器（auto-encoder）把这些向量压缩成短的二进制码来变得高效。这应当能产生一种比把自编码器直接用到原始像素上 [14] 好得多的图像检索方法，因为后者没有利用图像标签，因而倾向于检索边缘图案相似的图像，不管它们在语义上是否相似。

## 7 讨论

我们的结果表明，一个大型的深度卷积神经网络，仅仅使用监督学习（supervised learning），就能够在一个极具挑战性的数据集上取得破纪录的成果。值得注意的是，如果去掉单个卷积层，我们网络的性能就会下降。例如，去掉任何一个中间层，会让网络的 top-1 性能损失大约 2%。所以，深度对取得我们的结果确实重要。

为了简化实验，我们没有使用任何无监督预训练（unsupervised pre-training），尽管我们预期它会有帮助——尤其是当我们获得足够的计算能力、能在不相应增加带标签数据量的情况下大幅扩大网络规模时。到目前为止，我们把网络做得更大、训练得更久，结果就随之改善，但要赶上人类视觉系统的颞下通路（infero-temporal pathway），我们仍有许多个数量级的差距。最终，我们希望在视频序列上使用非常大、非常深的卷积网络，因为在视频中，时间结构提供了非常有帮助的信息，而这些信息在静态图像中是缺失的，或者远没有那么明显。

（译注：原文至此结束，正文之后直接是参考文献，没有致谢一节，也没有附录。这一点与本文件书目信息中的译本说明一致。）

## 参考文献（原文照录）

[1] R.M. Bell and Y. Koren. Lessons from the netflix prize challenge. *ACM SIGKDD Explorations Newsletter*, 9(2):75–79, 2007.

[2] A. Berg, J. Deng, and L. Fei-Fei. Large scale visual recognition challenge 2010. www.imagenet.org/challenges. 2010.

[3] L. Breiman. Random forests. *Machine learning*, 45(1):5–32, 2001.

[4] D. Cireşan, U. Meier, and J. Schmidhuber. Multi-column deep neural networks for image classification. *Arxiv preprint arXiv:1202.2745*, 2012.

[5] D.C. Cireşan, U. Meier, J. Masci, L.M. Gambardella, and J. Schmidhuber. High-performance neural networks for visual object classification. *Arxiv preprint arXiv:1102.0183*, 2011.

[6] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei. ImageNet: A Large-Scale Hierarchical Image Database. In *CVPR09*, 2009.

[7] J. Deng, A. Berg, S. Satheesh, H. Su, A. Khosla, and L. Fei-Fei. ILSVRC-2012, 2012. URL http://www.image-net.org/challenges/LSVRC/2012/.

[8] L. Fei-Fei, R. Fergus, and P. Perona. Learning generative visual models from few training examples: An incremental bayesian approach tested on 101 object categories. *Computer Vision and Image Understanding*, 106(1):59–70, 2007.

[9] G. Griffin, A. Holub, and P. Perona. Caltech-256 object category dataset. Technical Report 7694, California Institute of Technology, 2007. URL http://authors.library.caltech.edu/7694.

[10] G.E. Hinton, N. Srivastava, A. Krizhevsky, I. Sutskever, and R.R. Salakhutdinov. Improving neural networks by preventing co-adaptation of feature detectors. *arXiv preprint arXiv:1207.0580*, 2012.

[11] K. Jarrett, K. Kavukcuoglu, M. A. Ranzato, and Y. LeCun. What is the best multi-stage architecture for object recognition? In *International Conference on Computer Vision*, pages 2146–2153. IEEE, 2009.

[12] A. Krizhevsky. Learning multiple layers of features from tiny images. Master's thesis, Department of Computer Science, University of Toronto, 2009.

[13] A. Krizhevsky. Convolutional deep belief networks on cifar-10. Unpublished manuscript, 2010.

[14] A. Krizhevsky and G.E. Hinton. Using very deep autoencoders for content-based image retrieval. In *ESANN*, 2011.

[15] Y. Le Cun, B. Boser, J.S. Denker, D. Henderson, R.E. Howard, W. Hubbard, L.D. Jackel, et al. Handwritten digit recognition with a back-propagation network. In *Advances in neural information processing systems*, 1990.

[16] Y. LeCun, F.J. Huang, and L. Bottou. Learning methods for generic object recognition with invariance to pose and lighting. In *Computer Vision and Pattern Recognition, 2004. CVPR 2004. Proceedings of the 2004 IEEE Computer Society Conference on*, volume 2, pages II–97. IEEE, 2004.

[17] Y. LeCun, K. Kavukcuoglu, and C. Farabet. Convolutional networks and applications in vision. In *Circuits and Systems (ISCAS), Proceedings of 2010 IEEE International Symposium on*, pages 253–256. IEEE, 2010.

[18] H. Lee, R. Grosse, R. Ranganath, and A.Y. Ng. Convolutional deep belief networks for scalable unsupervised learning of hierarchical representations. In *Proceedings of the 26th Annual International Conference on Machine Learning*, pages 609–616. ACM, 2009.

[19] T. Mensink, J. Verbeek, F. Perronnin, and G. Csurka. Metric Learning for Large Scale Image Classification: Generalizing to New Classes at Near-Zero Cost. In *ECCV - European Conference on Computer Vision*, Florence, Italy, October 2012.

[20] V. Nair and G. E. Hinton. Rectified linear units improve restricted boltzmann machines. In *Proc. 27th International Conference on Machine Learning*, 2010.

[21] N. Pinto, D.D. Cox, and J.J. DiCarlo. Why is real-world visual object recognition hard? *PLoS computational biology*, 4(1):e27, 2008.

[22] N. Pinto, D. Doukhan, J.J. DiCarlo, and D. D. Cox. A high-throughput screening approach to discovering good forms of biologically inspired visual representation. *PLoS computational biology*, 5(11):e1000579, 2009.

[23] B.C. Russell, A. Torralba, K.P. Murphy, and W.T. Freeman. Labelme: a database and web-based tool for image annotation. *International journal of computer vision*, 77(1):157–173, 2008.

[24] J. Sánchez and F. Perronnin. High-dimensional signature compression for large-scale image classification. In *Computer Vision and Pattern Recognition (CVPR), 2011 IEEE Conference on*, pages 1665–1672. IEEE, 2011.

[25] P.Y. Simard, D. Steinkraus, and J.C. Platt. Best practices for convolutional neural networks applied to visual document analysis. In *Proceedings of the Seventh International Conference on Document Analysis and Recognition*, volume 2, pages 958–962, 2003.

[26] S.C. Turaga, J.F. Murray, V. Jain, F. Roth, M. Helmstaedter, K. Briggman, W. Denk, and H.S. Seung. Convolutional networks can learn to generate affinity graphs for image segmentation. *Neural Computation*, 22(2):511–538, 2010.
