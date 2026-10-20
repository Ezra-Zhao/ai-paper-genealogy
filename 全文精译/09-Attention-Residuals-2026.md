# 注意力残差（全文精译）

## 书目信息

- **原题**：Attention Residuals
- **作者**：Kimi Team（Moonshot AI）（译注：按本谱系统一口径，条目只署团队名，不展开个人作者细节。）
- **文类**：技术报告（Technical Report of Attention Residuals）
- **发表**：预印本 arXiv:2603.15031 [cs.CL]，2026 年 3 月 16 日提交
- **原文链接**：https://arxiv.org/abs/2603.15031 （摘要页）；全文 PDF：https://arxiv.org/pdf/2603.15031 ；官方代码仓库：https://github.com/MoonshotAI/Attention-Residuals
- **来源链与底本版本**：本译文的底本为 arXiv 官方发布的 arXiv:2603.15031v1 [cs.CL]，2026 年 3 月 16 日版。译前已在 arXiv 摘要页核对题名、团队署名、提交日期（Submitted on 16 Mar 2026）与摘要原文；全文取自 arXiv 官方 PDF（21 页）及其 LaTeX 源文件（e-print 源包，含正文 0–8 节、附录、全部图与表的源文件），公式、表格数字与图注均以源文件为准逐项核对；官方代码仓库 MoonshotAI/Attention-Residuals 的 README 与其中的伪代码、下游成绩表已作交叉确认，与论文底本一致。截至本译文写作时，arXiv 上只有 v1 一个版本。
- **对应导读**：见《AI 顶尖论文谱系》前沿卷 Attention Residuals 篇（★）
- **译本说明**：正文第 1–7 节、结论、全部图注与全部表格逐节精译；公式以 LaTeX 呈现并附中文说明；Block AttnRes 的 PyTorch 风格伪代码照原文录入；参考文献 73 条保留英文原样（见文末，未逐条翻译）；附录 A、附录 B 均译出，其中附录 A 原为作者姓名名单，按上述团队署名口径不逐一译录，已在该处如实注明，未作静默删节。原文无单独致谢一节。图 1 等各图的图像本身未附入本文件，图注全部译出，图内关键读数在译注中说明。本文件为 Ezra 个人学习收藏，仅本地存档。

---

## 摘要

带前归一化（PreNorm）的残差连接（residual connections）已是现代大语言模型的标准配置，可它以固定的单位权重累积所有层的输出。这种均匀聚合让隐藏状态随深度不受控地增长，逐步稀释每一层的贡献。本文提出注意力残差（attention residuals，AttnRes）：把这种固定累积换成对在先各层输出的 softmax 注意力（softmax attention），让每一层以学到的、随输入而变的权重，有选择地聚合更早的表征。为应对大规模模型训练中对全部在先层输出做注意力所带来的内存与通信开销，本文又提出分块注意力残差（Block AttnRes）：把各层划分为若干块，对块级表征做注意力，在压低内存占用的同时保住完整 AttnRes 的大部分收益。再配上基于缓存的流水线通信与两阶段计算策略，Block AttnRes 就能以极小的额外开销，作标准残差连接的即插即用替代品。

缩放律（scaling law）实验确认，改进在各种模型规模上都成立；消融实验也证实了按内容在深度方向做选择的好处。本文还把 AttnRes 接入 Kimi Linear 架构（总参数 48B／激活参数 3B），在 1.4T 个词元（tokens）上预训练。结果是 AttnRes 缓解了 PreNorm 的稀释问题，让各深度上的输出幅值与梯度分布都更均匀，下游表现在全部受评任务上都有提升。

**图 1** 注意力残差总览。**(a) 标准残差（Standard Residuals）**：标准残差连接，均匀地加法累积。**(b) 完整 AttnRes（Full AttnRes）**：每一层借学到的注意力权重，有选择地聚合全部先前层的输出。**(c) 分块 AttnRes（Block AttnRes）**：各层被分组成块，内存从 $O(Ld)$ 降到 $O(Nd)$。（译注：图中单个 Transformer 块内，自注意力层与 MoE 层各自算作一层、各自有一次残差聚合；输出端在图上标为 Output，底端为词嵌入 Embedding。）

## 1 引言

标准残差连接是现代大语言模型事实上的基本构件。更新式 $\bm{h}_l = \bm{h}_{l-1} + f_{l-1}(\bm{h}_{l-1})$ 常被理解为一条梯度高速公路（gradient highway）：梯度可以借恒等映射绕过变换，使深层训练保持稳定。但残差还担着第二个角色，得到的关注却少得多。把递推展开就会看到，每一层收到的都是全部先前层输出的同一个均匀加权和；也就是说，残差定义了信息怎样沿深度聚合。序列混合与专家路由如今都已用上可学习的、随输入而变的加权，唯独这种深度方向的聚合仍由固定单位权重支配，没有任何机制能有选择地强调或压低某一层的贡献。

实践中，PreNorm 已成主流范式，可它的无权重累积使隐藏状态幅值随深度按 $O(L)$ 增长，逐步稀释每一层的相对贡献。浅层信息被埋住，无法被有选择地取回；经验上，相当一部分层剪掉后损失也很小。近来的一些办法——缩放残差路径、多流递推——仍被绑在加法递推上；另一些确实引入跨层访问的办法又难以扩展。这一处境与注意力机制给出另一条路之前，循环神经网络（recurrent neural networks, RNNs）在序列维度上遇到的难题很像。

本文看到，深度方向的累积与 RNN 中的序列递推之间存在一种形式上的对偶（duality）。据此提出**注意力残差（AttnRes）**：把固定累积 $\bm{h}_l = \sum_i \bm{v}_i$ 换成 $\bm{h}_l = \sum_i \alpha_{i \to l} \cdot \bm{v}_i$，其中 $\alpha_{i \to l}$ 是 softmax 注意力权重，由每层单个学到的伪查询（pseudo-query）$\bm{w}_l \in \mathbb{R}^d$ 算出。这个轻量机制只用每层一个 $d$ 维向量，就能在深度方向做有选择的、感知内容的检索。事实上，标准残差连接与此前基于递推的各种变体，都可以看成在做深度方向的*线性*注意力（linear attention）；AttnRes 把它们推广为深度方向的 *softmax* 注意力，在深度上走完了当年在序列上带来变革的那步从线性到 softmax 的过渡（见第 6.2 节、第 6.1 节）。

常规训练里，完整 AttnRes（Full AttnRes）增加的开销可以忽略，因为它需要的层输出本就为反向传播保留着。可到了大规模，激活重计算（activation recomputation）与流水线并行（pipeline parallelism）成了常规做法，这些激活就必须被显式留存，还要跨流水线阶段传送。本文提出*分块 AttnRes（Block AttnRes）*来守住这一情形下的效率：把各层划分为 $N$ 个块，每块经标准残差归约为单个表征，跨块注意力只对这 $N$ 个块级摘要施加。这样内存与通信都降到 $O(Nd)$；再加上基础设施优化（第 4 节），Block AttnRes 就成了标准残差连接的即插即用替代品，训练成本只多出边际量，推理延迟开销也可忽略。

缩放律实验证实，在各个算力预算下 AttnRes 都稳定优于基线，其中 Block AttnRes 能追平一个多用了 $1.25\times$ 算力训练的基线的损失。本文还把 AttnRes 接入 Kimi Linear 架构（总参数 48B／激活参数 3B），在 1.4T 个词元上预训练。对训练动态的分析显示，AttnRes 缓解了 PreNorm 稀释：输出幅值在深度上保持有界，梯度范数在各层间分布得更均匀。下游基准上，最终模型在全部受评任务上都优于基线。

**本文贡献**

- **注意力残差。** 提出 AttnRes，用学到的、沿深度的 softmax 注意力取代固定残差累积；并给出可扩展的变体 Block AttnRes，把内存与通信从 $O(Ld)$ 降到 $O(Nd)$。借一套统一的结构化矩阵分析，本文表明标准残差与此前的递推类变体对应于深度方向的*线性*注意力，而 AttnRes 做的是深度方向的 *softmax* 注意力。
- **面向规模的基础设施。** 给出一组系统优化，让 Block AttnRes 在大规模下实用而高效：跨阶段缓存消除了流水线并行下的冗余传输；两阶段推理策略借在线 softmax（online softmax）摊薄跨块注意力。由此带来的训练开销只是边际量，典型推理负载下推理延迟开销低于 2%。
- **完整的评测与分析。** 经缩放律实验、组件消融，以及在一个 48B 参数、用 1.4T 词元预训练的模型上的下游基准，验证 AttnRes 相对标准残差连接有稳定改进。训练动态分析进一步显示，AttnRes 缓解了 PreNorm 稀释，带来有界的隐藏状态幅值与沿深度更均匀的梯度分布。

## 2 动机

**记号。** 考虑一批输入序列，形状为 $B \times T \times d$，其中 $B$ 是批大小，$T$ 是序列长度，$d$ 是隐藏维度。为表述清楚，公式只写单个词元：$\bm{h}_l \in \mathbb{R}^{d}$ 表示进入第 $l$ 层的隐藏状态，$l \in \{1, \ldots, L\}$ 是层下标，$L$ 是总层数。词元嵌入（token embedding）就是 $\bm{h}_1$。函数 $f_l$ 表示第 $l$ 层施加的变换。在 Transformer 模型里，本文把每一个自注意力（self-attention）或 MLP 都当作单独的一*层*。

### 2.1 借残差训练深层网络

**残差学习。** 残差学习（residual learning）已被证明是训练深层网络的一项关键技术，它让梯度绕过变换。具体地，每一层这样更新隐藏状态：

$$\bm{h}_{l} = \bm{h}_{l-1} + f_{l-1}(\bm{h}_{l-1})$$

（公式说明：$\bm{h}_{l-1}$ 是上一层的隐藏状态，$f_{l-1}(\bm{h}_{l-1})$ 是上一层的变换输出，两者直接相加。）

展开这一递推，第 $l$ 层的隐藏状态是嵌入与全部先前层输出之和：$\bm{h}_l = \bm{h}_1 + \sum_{i=1}^{l-1} f_i(\bm{h}_i)$。残差连接的关键洞见是*恒等映射*（identity mapping）：每层都为信息与梯度留一条不加改变的直通路径。反向传播时，损失对某个中间隐藏状态的梯度为：

$$\frac{\partial \mathcal{L}}{\partial \bm{h}_l} = \frac{\partial \mathcal{L}}{\partial \bm{h}_{L}} \cdot \prod_{j=l}^{L-1} \left( \mathbf{I} + \frac{\partial f_j}{\partial \bm{h}_j} \right)$$

（公式说明：$\mathcal{L}$ 是损失，$\mathbf{I}$ 是单位矩阵，$\partial f_j/\partial \bm{h}_j$ 是第 $j$ 层变换的雅可比矩阵。）

把这个连乘展开，得到 $\mathbf{I}$ 加上涉及各层雅可比的高阶项。恒等项始终保留，于是不论深度多深，都有一条从损失直达任意层的梯度通路。

**推广残差。** 残差更新里固定的单位系数虽然有效，却把每层的贡献一律同等对待，没有机制让深度方向的混合随情况调整。高速公路网络（Highway networks）借学到的逐元素门控（gates）放松了这一点：

$$\bm{h}_{l} = (1 - \bm{g}_{l}) \odot \bm{h}_{l-1} + \bm{g}_{l} \odot f_{l-1}(\bm{h}_{l-1})$$

（公式说明：$\bm{g}_l \in [0,1]^d$ 是门控向量，$\odot$ 是逐元素乘法；门控在变换路径与恒等路径之间插值。）

更一般地，两者都是加权递推 $\bm{h}_{l} = \alpha_{l} \cdot \bm{h}_{l-1} + \beta_{l} \cdot f_{l-1}(\bm{h}_{l-1})$ 的特例：残差取 $\alpha_{l} = \beta_{l} = 1$，Highway 取 $\alpha_{l} = 1 - \bm{g}_{l}$、$\beta_{l} = \bm{g}_{l}$。

**局限。** 不论固定还是门控，这两种做法都受同一个根本约束：每层只能访问它的直接输入 $\bm{h}_{l-1}$——一个把全部更早层输出揉在一起的单一压缩状态，而不是那些输出本身。由此带来几重局限：（1）*无从选择性访问*：不同类型的层（比如注意力层与 MLP 层）收到的是同一个聚合状态，尽管它们本可能受益于不同的加权；（2）*不可逆的信息损失*：聚合中丢掉的信息，在更深的层里无法被有选择地找回；（3）*输出膨胀*：靠后的层为了在累积起来的残差上争得影响力，会学出越来越大的输出，这会让训练失稳。这几重局限呼唤一种机制，让每层都能从全部先前层里有选择地聚合信息。

## 3 注意力残差：时间与深度的统一视角

上文讨论的局限，让人想起序列建模里相似的瓶颈，提示本文到深度维度上找相似的解法。

**时间与深度的对偶。** 残差连接沿深度把全部先前信息压进单一状态 $\bm{h}_l$，正如 RNN 沿时间所做的那样。在序列建模上，Transformer 用注意力取代递推得到改进，让每个位置都能以随数据而变的权重，有选择地访问全部先前位置。本文对深度提出同样的方法论：

$$\bm{h}_{l} = \alpha_{0 \to l} \cdot \bm{h}_1 + \sum_{i=1}^{l-1} \alpha_{i \to l} \cdot f_i(\bm{h}_{i}) \tag{1}$$

（公式说明：$\alpha_{i \to l}$ 是第 $l$ 层分给第 $i$ 个来源的注意力权重，满足 $\sum_{i=0}^{l-1} \alpha_{i \to l} = 1$；$i=0$ 对应词嵌入 $\bm{h}_1$。）

序列长度可以长到数百万个词元，网络深度却通常不大（$L < 1000$），因此在深度上做 $O(L^2)$ 注意力在算力上可行。本文把这一做法称为*注意力残差*，简称 *AttnRes*。

### 3.1 完整注意力残差

注意力权重可以写成 $\alpha_{i \to l} = \phi(\bm{q}_{l},\, \bm{k}_{i})$，其中 $\phi\colon \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}_{\geq 0}$ 是核函数（kernel function），$\bm{q}_l$ 与 $\bm{k}_i$ 分别是查询（query）与键（key）向量。$\phi$ 的不同取法对应不同的残差变体（见第 6.2 节）；本文取 $\phi(\bm{q}, \bm{k}) = \exp\left(\bm{q}^\top\operatorname{RMSNorm}(\bm{k})\right)$ 并作归一化，得到深度方向的 softmax 注意力：

$$\alpha_{i \to l} = \frac{\phi\left(\bm{q}_{l}, \bm{k}_{i}\right)}{\sum_{j=0}^{l-1} \phi\left(\bm{q}_{l}, \bm{k}_{j}\right)} \tag{2}$$

（公式说明：分子是第 $l$ 层的查询与第 $i$ 个来源的键之间的 $\phi$ 值，分母是对全部 $l$ 个来源求和，这正是 softmax 的归一化形式；$\operatorname{RMSNorm}$ 是均方根归一化。）

对每一层 $l$，定义：

$$\bm{q}_{l} = \bm{w}_{l}, \quad \quad \bm{k}_{i} = \bm{v}_{i} = \begin{cases} \bm{h}_1 & i = 0 \\ f_i(\bm{h}_{i}) & 1 \leq i \leq l-1 \end{cases} \tag{3}$$

（公式说明：查询 $\bm{q}_l=\bm{w}_l$ 是逐层学到的向量，$\bm{w}_l \in \mathbb{R}^{d}$；键与值取同一个向量 $\bm{v}_i$，即嵌入或先前层的原始输出。）

其中 $\phi$ 内部的 $\operatorname{RMSNorm}$ 防止输出幅值大的层主导注意力权重。第 $l$ 层的输入于是是：

$$\bm{h}_{l} = \sum_{i=0}^{l-1} \alpha_{i \to l} \cdot \bm{v}_{i} \tag{4}$$

本文把这一形式称为*完整注意力残差*（full attention residuals）。对每个词元，完整 AttnRes 需要 $O(L^2 d)$ 的运算量与 $O(Ld)$ 的内存来存各层输出。深度远小于序列长度，运算代价并不大。

**开销。** $O(Ld)$ 内存与反向传播本就要保留的激活完全重叠，因此完整 AttnRes 在普通训练里不增加额外内存开销。但到了大规模，激活重计算与流水线并行被广泛采用：本来会被释放、再重算的层输出，如今必须为全部后续层一直留存；在流水线并行下，每个输出还要跨阶段边界传送。于是内存与通信开销都长成 $O(Ld)$。

**分块优化。** 完整 AttnRes 里有一处刻意的设计：伪查询（pseudo-query）$\bm{w}_l$ 是学到的参数，与该层的前向计算解耦。正因这份独立性，任意一组层的注意力权重都能并行算出，不必等它们的顺序输出；具体说，可以把 $L$ 层分成 $N$ 个块、每块 $S$ 层，把块内注意力计算批处理起来，每层内存读写就从 $O(Ld)$ 降到 $O((S{+}N)d)$（两阶段策略的细节见第 4 节）。不过在当前的分布式训练格局下，主导成本不是本地内存带宽，而是流水线并行下的跨阶段通信：每个层输出仍要在阶段之间传送，这份 $O(Ld)$ 通信开销靠本地批处理消不掉。这正是下文 Block AttnRes 变体的动机——它把跨阶段表征的数量从 $L$ 降到 $N$。本文预期，将来互连条件改善后，完整的 $O(Ld)$ 通信会变得可行，完整 AttnRes 的潜力能充分发挥。

### 3.2 分块注意力残差

本文提出*分块注意力残差*（Block Attention Residuals，Block AttnRes）：把 $L$ 层划分为 $N$ 个块；块内，各层输出经求和归约为单个表征；跨块，只对 $N$ 个块级表征与词嵌入施加完整注意力。这样内存与通信开销都从 $O(Ld)$ 降到 $O(Nd)$。

**块内累积。** 具体做法是把 $L$ 层分成 $N$ 个块，每块 $S = L/N$ 层，这里设 $L$ 能被 $N$ 整除；若不能整除，最后一块装余下的 $L \bmod N$ 层。记 $\mathcal{B}_n$ 为第 $n$ 个块（$n = 1, \ldots, N$）的层下标集合。构成一个块，就是把块内全部层输出求和：

$$\bm{b}_n = \sum_{j \in \mathcal{B}_n} f_j(\bm{h}_j) \tag{5}$$

（公式说明：$\bm{b}_n$ 是第 $n$ 个块的块级表征。）

再记 $\bm{b}_n^i$ 为 $\mathcal{B}_n$ 内前 $i$ 层的部分和（partial sum），于是 $\bm{b}_n = \bm{b}_n^{S}$。当 $L$ 不能被 $N$ 整除时，最后一个部分和就取作最后一块的表征。与完整 AttnRes 一样，$\phi$ 内部的 $\operatorname{RMSNorm}$ 防止完整块与部分和之间的幅值差异扭曲注意力权重。

**块间注意力。** 在完整 AttnRes 里，第 $l$ 层的输入是对直至 $f_{l-1}(\bm{h}_{l-1})$ 的全部输出做注意力算出的。分块变体把这些单个输出换成块表征，并定义 $\bm{b}_0 = \bm{h}_1$，使词嵌入始终作为一个来源。对第 $n$ 块内的第 $i$ 层，值矩阵（value matrix）为：

$$\mathbf{V} = \begin{cases} [\bm{b}_0, \bm{b}_1, \ldots, \bm{b}_{n-1}]^\top & \text{若 } i = 1 \text{（块 } n \text{ 的第一层）} \\ [\bm{b}_0, \bm{b}_1, \ldots, \bm{b}_{n-1}, \bm{b}_n^{i-1}]^\top & \text{若 } i \geq 2 \text{（其后各层）} \end{cases} \tag{6}$$

（公式说明：块内第一层只能看到此前各块；其后各层还多看到本块截至上一层的部分和 $\bm{b}_n^{i-1}$。）

键与注意力权重按公式（3）与公式（2）算。网络最第一层的输入是词嵌入，即 $\bm{b}_0 = \bm{h}_1$。每个块内，第一层收到先前各块的表征与词嵌入，其后各层还额外对部分和 $\bm{b}_n^{i-1}$ 做注意力。最后的输出层聚合全部 $N$ 个块表征。图 2（清单）给出 Block AttnRes 的 PyTorch 风格伪代码。

**图 2** Block Attention Residuals 的 PyTorch 风格伪代码。`block_attn_res` 用学到的伪查询 $\bm{w}_l$ 对块表征计算 softmax 注意力；`forward` 是单层的一趟计算，维护 `partial_block`（$\bm{b}_n^i$，块内残差）与 `blocks`（$[\bm{b}_0, \ldots, \bm{b}_{n-1}]$，块间历史）。

```python
def block_attn_res(blocks: list[Tensor], partial_block: Tensor, proj: Linear, norm: RMSNorm) -> Tensor:
    """
    Inter-block attention: attend over block reps + partial sum.
    blocks:
        N tensors of shape [B, T, D]: completed block representations for each previous block
    partial_block:
        [B, T, D]:    intra-block partial sum (b_n^i)
    """
    V = torch.stack(blocks + [partial_block])  # [N+1, B, T, D]
    K = norm(V)
    logits = torch.einsum('d, n b t d -> n b t', proj.weight.squeeze(), K)
    h = torch.einsum('n b t, n b t d -> b t d', logits.softmax(0), V)
    return h

def forward(self, blocks: list[Tensor], hidden_states: Tensor) -> tuple[list[Tensor], Tensor]:
    partial_block = hidden_states
    # apply block attnres before attn
    # blocks already include token embedding
    h = block_attn_res(blocks, partial_block, self.attn_res_proj, self.attn_res_norm)

    # if reaches block boundary, start new block
    # block_size counts ATTN + MLP; each transformer layer has 2
    if self.layer_number % (self.block_size // 2) == 0:
        blocks.append(partial_block)
        partial_block = None

    # self-attention layer
    attn_out = self.attn(self.attn_norm(h))
    partial_block = partial_block + attn_out if partial_block is not None else attn_out

    # apply block attnres before MLP
    h = block_attn_res(blocks, partial_block, self.mlp_res_proj, self.mlp_res_norm)

    # MLP layer
    mlp_out = self.mlp(self.mlp_norm(h))
    partial_block = partial_block + mlp_out

    return blocks, partial_block
```

（译注：伪代码内注释为原文英文，照录不译；`proj.weight` 即该层的伪查询 $\bm{w}_l$。）

**效率。** 每层如今是对 $N$ 个块表征而非 $L$ 个单个输出做注意力，内存从 $O(L)$ 降到 $O(N)$，计算从 $O(L^2)$ 降到 $O(N^2)$。块数 $N$ 在两个极端之间插值：$N = L$ 时回到完整 AttnRes，$N = 1$ 时退化为标准残差连接，只是嵌入被单独隔离为 $\bm{b}_0$。经验上，本文发现 $N \approx 8$ 在各模型规模上都能收回大部分收益，每个词元只需存 8 个隐藏状态（见第 5 节）。

块结构对推理延迟也有好处，超出内存与计算之外：块边界决定了第 3.1 节所述分块优化的调度粒度，固定的块数 $N$ 给 KV 缓存（KV cache）大小封了顶。块间的并行结果与块内顺序的部分和，经在线 softmax 合并，保持精确等价（见第 4 节）。

## 4 基础设施设计

与标准残差连接相比，Block AttnRes 带来了额外的系统挑战。大规模模型训练中，块表征必须跨流水线阶段传播，朴素实现会造成沉重通信。推理时，反复访问累积的块表征会抬高延迟，长上下文预填充（prefilling）又会放大缓存块表征的内存代价。本文的应对是：训练中用跨阶段缓存（cross-stage caching）；推理中用两阶段计算策略，再加一套省内存的预填充方案。

### 4.1 训练

小规模训练时，AttnRes 只增加很小的计算开销，不增加额外内存，因为激活反正要为反向传播保存。大规模分布式训练下，流水线并行是 AttnRes 面临的主要基础设施挑战。完整 AttnRes 要求全部 $L$ 个层输出跨阶段传送；Block AttnRes 把这件事降到 $N$ 个块表征，下面的优化再把余下的开销压到最小。

**流水线通信。** 用标准残差连接时，流水线并行在相邻阶段之间传送固定大小的隐藏状态，与流水线深度无关。Block AttnRes 则要求每个阶段都备齐全部累积块表征以做块间注意力；若每次交接都把完整历史传一遍，就会产生冗余通信。

考虑交错流水线调度（interleaved pipeline schedule）：有 $P$ 个物理阶段，每个物理阶段有 $V$ 个虚拟阶段。为简单计，设每个物理阶段平均每个词元产生 $N_p$ 个维度为 $d$ 的块表征。（译注：原文脚注说，实践中块边界不必与物理阶段边界对齐；例如在图 3 中，每个块横跨两个物理阶段，因此只有每隔一次的交接才涉及一个新完成的块。）总块数（chunks）为 $C = PV$（每个物理阶段乘每个虚拟阶段），第 $j$ 个块累积了 $jN_p$ 个块。若每次交接都朴素地传送全部累积块，每个词元的通信代价为：

$$\mathrm{Comm}_{\text{naïve}} = \sum_{j=1}^{C-1} jN_p \cdot d = \frac{C(C{-}1)}{2}\,N_p d. \tag{7}$$

**跨阶段缓存。** 每个物理阶段会连续处理多个虚拟阶段，因此可以把块缓存在本地来消除这份冗余：较早虚拟阶段收到过的块留在本地内存里，不必重传。第一个虚拟阶段（$v = 1$）没有缓存、照常累积；$v \geq 2$ 时，每次交接只传送自接收方在上一虚拟阶段对应块以来新累积的约 $P N_p$ 个增量块。总通信降为：

$$\mathrm{Comm}_{\text{cached}} = \underbrace{\frac{P(P{-}1)}{2}\, N_p d}_{\text{第一个虚拟阶段}} + \underbrace{(V{-}1)\, P^2\, N_p d}_{\text{其后各虚拟阶段}}. \tag{8}$$

**图 3** 基于缓存的流水线通信示例：4 个物理秩（physical ranks）、每秩 2 个虚拟阶段，带条纹的框表示 AttnRes 块的结束处。数字是微批（micro-batch）下标。每个秩缓存此前收到过的块；阶段交接只传送增量块（$+[\bm{b}_1, \bm{b}_2]$），而不是完整历史。

缓存把每次交接的峰值代价从 $O(C)$ 降到 $O(P)$，是 $V$ 倍的改进，并使稳态 1F1B 下通信能与计算完全重叠。反向传播同样受益于这套方案。图 3 以 $P{=}4$、$V{=}2$ 说明这一优化：在第二个虚拟阶段，缓存消除了 6 次冗余的块传输。

**内存开销。** 有了跨阶段缓存，每个块在全部 $V$ 个虚拟阶段中恰好只存一份，相对标准的逐层激活缓存可以忽略。关键在于，逐层激活占用与标准架构完全相同，因为激活检查点（activation checkpointing）消除了全部块间注意力的中间量，被检查点保存的输入 $\bm{p}_l$ 与它所替代的隐藏状态 $\bm{h}_l$ 内存大小相同。

按墙钟时间计，不启用流水线并行时，Block AttnRes 增加的训练开销可以忽略；启用流水线并行时，实测端到端开销低于 4%。

### 4.2 推理

下文的两阶段计算策略对完整与 Block AttnRes 都适用：两种情形下，各层都被分组成大小为 $S$ 的块，阶段 1 批量处理块间查询，阶段 2 处理块内的顺序回看。对完整 AttnRes，这把每层读写从 $O(Ld)$ 降到 $O((S{+}N)d)$（详细推导见附录 B）；Block AttnRes 还进一步把所存表征从 $L$ 个降到 $N$ 个，因为每块被压成单个向量。下文聚焦 Block AttnRes，细述两阶段计算策略与一套面向长上下文输入的、按序列分片的预填充方案。

**两阶段计算策略。** Block AttnRes 的逐层注意力计算很像自回归解码：块表征充当一份被各层共用的 KV 缓存。朴素实现会在每层都算一遍注意力残差，每次都要完整扫过全部先前块，内存访问达 $O(L \cdot N)$。伪查询向量与前向计算解耦（见第 3 节），因此一个块内全部 $S = L/N$ 个查询可以批成一次矩阵乘法，把内存访问从 $S$ 次读摊薄为 1 次。

算法 1 是利用这一性质的两阶段计算策略：

- **阶段 1** 对块内全部 $S$ 层同时做块间注意力：用一次批量查询对缓存的块表征计算，同时返回输出与 softmax 统计量（最大值与对数和指数 log-sum-exp）。这摊薄了内存访问代价，每块从读 $S$ 次降到只读 1 次。
- **阶段 2** 对每层用不断演进的部分和顺序地做块内注意力，再经在线 softmax 与阶段 1 的输出合并。在线 softmax 合并是逐元素的，这一阶段天然能与周围运算做核融合（kernel fusion），进一步压低读写开销。

**算法 1** 第 $n$ 块的两阶段计算

输入：伪查询 $\{\bm{w}_l\}_{l \in \mathcal{B}_n}$，块表征 $\{\bm{b}_0, \ldots, \bm{b}_{n-1}\}$

- 阶段 1：并行块间注意力
  - $\mathbf{Q} \gets [\bm{w}_l]_{l \in \mathcal{B}_n}$（形状 $[S, d]$）
  - $\mathbf{K}, \mathbf{V} \gets [\bm{b}_0; \ldots; \bm{b}_{n-1}]$（形状 $[n, d]$）
  - $\{\bm{o}_l^{(1)}, m_l^{(1)}, \ell_l^{(1)}\}_{l \in \mathcal{B}_n} \gets \textsc{AttnWithStats}(\mathbf{Q}, \mathbf{K}, \mathbf{V})$（返回 LSE，即对数和指数）
- 阶段 2：顺序块内注意力 + 在线 softmax 合并
  - $i \gets 0$
  - 对每个 $l \in \mathcal{B}_n$：
    - 若 $i = 0$：$\bm{h}_l \gets \bm{o}_l^{(1)} / \ell_l^{(1)}$（只有块间部分）
    - 否则：$\bm{o}_l^{(2)}, m_l^{(2)}, \ell_l^{(2)} \gets \textsc{AttnWithStats}(\bm{w}_l, \bm{b}_n^i, \bm{b}_n^i)$（块内部分）；$m_l \gets \max(m_l^{(1)}, m_l^{(2)})$；$\bm{h}_l \gets \dfrac{e^{m_l^{(1)} - m_l} \bm{o}_l^{(1)} + e^{m_l^{(2)} - m_l} \bm{o}_l^{(2)}}{e^{m_l^{(1)} - m_l} \ell_l^{(1)} + e^{m_l^{(2)} - m_l} \ell_l^{(2)}}$（在线 softmax 合并）
    - $i \gets i + 1$
    - $\bm{b}_n^{i} \gets \bm{b}_n^{i-1} + f_l(\bm{h}_l)$（更新部分和；$\bm{b}_n^0 \coloneqq \bm{0}$）
  - 返回 $\{\bm{h}_l\}_{l \in \mathcal{B}_n}$

（公式说明：$m$ 是 softmax 的最大值统计量，$\ell$ 是对数和指数相关的归一化统计量，上标 (1)、(2) 分别对应块间与块内两部分；合并式是在线 softmax 的精确合并，不是近似。）

有了两阶段设计，阶段 2 的读写占用与标准残差连接相近，主要额外成本来自阶段 1 的块间注意力。这些块间读取经批处理摊到块内全部层上，总的每层内存访问代价只剩 $(\frac{N}{S} + 3)d$ 次读与 $2d$ 次写（表 1）。这远低于此前残差推广方案如 (m)HC 在典型设定下的残差流读写。实践中，阶段 1 还能与块内第一层的计算部分重叠，进一步减小墙钟影响。结果是，典型推理负载下端到端推理延迟开销低于 2%。

**表 1** 各方案下残差机制在每个词元、每层上造成的内存访问代价。不含层函数 $f_l$ 自身的内部读写。AttnRes 的完整与分块两变体都用附录 B 的两阶段推理调度；摊薄代价按一个块内 $N$ 层取平均。典型取值：$L{=}128$、$N{=}8$、$S{=}L/N{=}16$、$m{=}4$。

| 方案 | 操作 | 读 | 写 | 总读写（符号） | 总读写（典型值） |
|---|---|---|---|---|---|
| 标准残差 | 残差合并 | $2d$ | $d$ | $3d$ | $3d$ |
| mHC（$m$ 条流） | 计算 $\bm{\alpha}_{l}$、$\bm{\beta}_{l}$、$\mathbf{A}_{l}$ | $md$ | $m^2{+}2m$ | $(8m{+}2)d{+}2m^2{+}4m$（五项合计） | $34d$ |
| mHC | 施加 $\bm{\alpha}_{l}$ | $md{+}m$ | $d$ | — | — |
| mHC | 施加 $\bm{\beta}_{l}$ | $d{+}m$ | $md$ | — | — |
| mHC | 施加 $\mathbf{A}_{l}$ | $md{+}m^2$ | $md$ | — | — |
| mHC | 残差合并 | $2md$ | $md$ | — | — |
| AttnRes 完整（朴素） | 注意力 | $(L{+}1)d$ | $d$ | $(L{+}2)d$ | $130d$ |
| AttnRes 完整 | 阶段 1（摊薄） | $(N{-}1)d$ | $d$ | $(S{+}N)d$（两阶段合计） | $24d$ |
| AttnRes 完整 | 阶段 2 | $(S{-}1)d$ | $d$ | — | — |
| AttnRes 分块 | 阶段 1（摊薄） | $\frac{N}{S}d$ | $d$ | $\left(\frac{N}{S}{+}5\right)d$（两阶段合计） | $5.5d$ |
| AttnRes 分块 | 阶段 2 | $3d$ | $d$ | — | — |

（译注：原表把完整 AttnRes 的朴素行以注释形式保留，此处一并列出以便对照；mHC 一行的总读写为其五行操作合计。）

**省内存的预填充。** 预填充时存块表征需要 $N \cdot T \cdot d$ 个元素：对一个 128K 词元的序列、8 个块而言，这要占 15 GB 内存。本文的缓解办法是把这些表征沿序列维度分片到 $P$ 台张量并行（tensor-parallel）设备上，让阶段 1 能在本地序列分片上独立执行。阶段 2 的在线 softmax 合并则接入标准 TP 全规约（all-reduce）通信路径：输出先做规约分发（reduce-scatter），在本地合并，再经全收集（all-gather）重构，并天然能与 $\operatorname{RMSNorm}$ 之类运算做核融合。这样每台设备的内存占用降到 $N \cdot (T/P) \cdot d$——128K 上下文的例子就从 15 GB 降到每台设备约 1.9 GB。再结合分块预填充（如 16K 的块大小），开销进一步降到每台设备 0.3 GB 以下。

## 5 实验

**架构细节。** 本文的架构与 Kimi Linear 完全相同：一个专家混合（Mixture-of-Experts, MoE）Transformer，沿 Moonlight／DeepSeek-V3 的设计，以 3:1 的比例交错 Kimi Delta Attention（KDA）层与多头潜在注意力（Multi-Head Latent Attention, MLA）层，每层之后接一个 MoE 前馈层。唯一的改动是给残差连接加上 AttnRes；其余组件（模型深度、隐藏维度、专家路由、MLP 结构）都不变。AttnRes 每层只引入一个 $\operatorname{RMSNorm}$ 与一个伪查询向量 $\bm{w}_l \in \mathbb{R}^d$，在总参数量里只占可忽略的一小部分。关键一点：全部伪查询向量必须初始化为零。这保证初始注意力权重 $\alpha_{i \to l}$ 在各来源层上均匀，AttnRes 在训练开始时退化为等权平均，避免训练波动——这一点本文已用实验验证。

### 5.1 缩放律

**表 2** 基线 vs 分块 AttnRes（$N=8$）vs 完整 AttnRes vs mHC(-lite)：模型配置、超参数与验证损失。

| 激活参数† | 词元数 | $L_b$ | $H$ | $d_{\text{model}}$ | $d_{\text{ff}}$ | 学习率 | 批大小‡ | 基线 | Block AttnRes | Full AttnRes | mHC(-lite) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 194M | 38.7B | 12 | 12 | 896 | 400 | $2.99 \times 10^{-3}$ | 192 | 1.931 | 1.909 | **1.899** | 1.906 |
| 241M | 45.4B | 13 | 13 | 960 | 432 | $2.80 \times 10^{-3}$ | 256 | 1.895 | 1.875 | 1.874 | **1.869** |
| 296M | 62.1B | 14 | 14 | 1024 | 464 | $2.50 \times 10^{-3}$ | 320 | 1.829 | 1.809 | **1.804** | 1.807 |
| 436M | 87.9B | 16 | 16 | 1168 | 528 | $2.20 \times 10^{-3}$ | 384 | 1.766 | 1.746 | **1.737** | 1.747 |
| 528M | 119.0B | 17 | 17 | 1264 | 560 | $2.02 \times 10^{-3}$ | 432 | 1.719 | 1.693 | **1.692** | 1.694 |

（译注：原表脚注——† 指本文 MoE 模型中激活参数的数量，不含嵌入；‡ 全部模型都用 8192 的上下文长度训练；⋆ $L_b = L/2$ 表示 Transformer 块的数量。原表粗体为每行最优值，照录。）

**图 4** Attention Residuals 的缩放律曲线。完整与分块 AttnRes 在全部规模上都稳定优于基线。Block AttnRes 紧贴完整 AttnRes，在最大规模上收回了大部分收益。（译注：图中拟合式为基线 $\mathcal{L} = 1.891 \times C^{-0.057}$、完整 AttnRes $\mathcal{L} = 1.865 \times C^{-0.057}$、Block AttnRes $\mathcal{L} = 1.870 \times C^{-0.058}$，横轴 $C$ 为以 PFLOP/s-days 计的算力，图上标有 $1.25\times$ 的算力等效箭头。）

本文扫了 5 个模型规模（表 2），每个规模训练 3 个变体：PreNorm 基线、完整 AttnRes，以及块数约 8 的 Block AttnRes。它们用 8192 词元的上下文窗口与余弦学习率调度训练。同一个缩放律规模组内，全部变体共用在基线下选定的同一组超参数，以保证比较公平；这一设定有意偏向基线，因此比较是保守的。按通行做法，本文拟合 $\mathcal{L} = A \times C^{-\alpha}$ 形式的幂律曲线，其中 $\mathcal{L}$ 是验证损失，$C$ 是以 PFLOP/s-days 计的算力。

**缩放行为。** 图 4 给出拟合的缩放曲线。基线服从 $\mathcal{L} = 1.891 \times C^{-0.057}$，Block AttnRes 拟合为 $\mathcal{L} = 1.870 \times C^{-0.058}$，完整 AttnRes 拟合为 $\mathcal{L} = 1.865 \times C^{-0.057}$。三个变体的斜率相近，但 AttnRes 在整个算力范围内都取得更低的损失。按拟合曲线，在 5.6 PFLOP/s-days 处，Block AttnRes 达到 1.692，基线为 1.714，相当于 $1.25\times$ 的算力优势。完整与 Block AttnRes 之间的差距随规模缩小，在最大规模上只剩 0.001。表 2 中还列入 mHC(-lite) 供参照。完整 AttnRes 优于 mHC，Block AttnRes 则以更低的每层内存读写追平它：$5.5d$ 对 $m{=}4$ 条流的 mHC 的 $34d$（表 1）。

### 5.2 主要结果

**训练配方。** 本文研究的最大模型基于完整的 Kimi Linear 48B 配置：27 个 Transformer 块（54 层），256 个路由专家中激活 8 个，另加 1 个共享专家，总参数 48B、激活参数 3B。这个模型用 Block AttnRes，每块 6 层，得到 9 个块，再加上词嵌入，深度方向共有 10 个来源。

本文沿用 Kimi Linear 1.4T 词元那批运行的同一套数据与训练配方：全部模型都用 4096 词元的上下文窗口、Muon 优化器、WSD（Warmup–Stable–Decay，预热—稳定—衰减）学习率调度预训练，全局批大小为 800 万（8M）词元。最终模型的训练分两个阶段：（i）在 1T 词元上做 WSD 预训练；（ii）随后在约 400B 高质量词元上做中训练（mid-training），沿用 Moonlight 的退火配方。

中训练之后，继续用渐进加长的序列长度 32K 词元训练。本文架构用的是混合 KDA/MLA 注意力，其中 MLA 不用位置编码（NoPE），因此扩展上下文不需要 YaRN 之类改动，也不需要重缩放注意力温度。

**图 5** 基线与 Block AttnRes 的训练动态。**(a)** 训练期间的验证损失。**(b)** 训练结束时每个 Transformer 块的输出幅值。**(c)** 每个 Transformer 块的梯度幅值。

**训练动态。** 图 5 比较了最终的基线模型与 Block AttnRes 模型在 1T 词元上的训练动态：

- **验证损失：** AttnRes 在整个训练过程中验证损失都更低，差距在衰减阶段拉大，最终损失明显更低。
- **输出幅值：** 基线受 PreNorm 稀释问题困扰：隐藏状态幅值随深度单调增长，深层被迫从固定尺度的归一化输入里学出越来越大的输出，才能保持影响力。Block AttnRes 把这种增长限制在每块内部，因为块边界上的选择性聚合把累积重置了，得到一种有界的周期形态。
- **梯度幅值：** 基线的全部残差权重都固定为 1，没有手段调节深度方向的梯度流，导致最早的层梯度不成比例地大。Block AttnRes 中可学习的 softmax 权重（图 8）让各来源为概率质量相互竞争，梯度分布于是均匀得多。

**表 3** AttnRes 与基线在同一预训练配方后的性能比较。每行最优结果以粗体标出。

| 类别 | 基准 | 基线 | AttnRes |
|---|---|---|---|
| 通用（General） | MMLU | 73.5 | **74.6** |
| 通用 | MMLU-Pro | **52.2** | **52.2** |
| 通用 | GPQA-Diamond | 36.9 | **44.4** |
| 通用 | BBH | 76.3 | **78.0** |
| 通用 | ARC-Challenge | 64.6 | **65.7** |
| 通用 | HellaSwag | 83.2 | **83.4** |
| 通用 | TriviaQA | 69.9 | **71.8** |
| 数学与代码（Math & Code） | GSM8K | 81.7 | **82.4** |
| 数学与代码 | MGSM | 64.9 | **66.1** |
| 数学与代码 | Math | 53.5 | **57.1** |
| 数学与代码 | CMath | 84.7 | **85.1** |
| 数学与代码 | HumanEval | 59.1 | **62.2** |
| 数学与代码 | MBPP | 72.0 | **73.9** |
| 中文（Chinese） | CMMLU | 82.0 | **82.9** |
| 中文 | C-Eval | 79.6 | **82.5** |

（译注：原表把 MMLU-Pro 一行标为 MMLU-Pro，正文评测清单里写作 MMLU-Pro Hard，系同一基准的口径差异，照原文分别保留；官方仓库 README 的成绩表是本表的子集，数值一致。）

**下游表现。** 沿 Kimi Linear 的评测规程，本文在三个方面评测两个模型（表 3）：

- **语言理解与推理**：MMLU、MMLU-Pro Hard、GPQA-Diamond、BBH、ARC-Challenge、HellaSwag 与 TriviaQA。
- **推理（代码与数学）**：GSM8K、MGSM、Math、CMath、HumanEval 与 MBPP。
- **中文语言理解**：CMMLU 与 C-Eval。

如表 3 所示，Block AttnRes 在全部基准上追平或优于基线。提升在多步推理任务上格外明显，如 GPQA-Diamond（+7.5）与 Minerva Math（+3.6，译注：即表 3 的 Math 一行），代码生成如 HumanEval（+3.1）也一样；知识型基准如 MMLU（+1.1）与 TriviaQA（+1.9）也有扎实的收益。这一形态与下述设想吻合：深度方向信息流的改善对组合型任务有好处——在那里，后面的层能有选择地取回更早的表征，并在此之上继续构建。

### 5.3 消融研究

本文在表 2 中的 16 头模型上做消融，以验证 AttnRes 的关键设计选择（表 4）。全部模型共用同一组超参数与同一算力预算。

**表 4** AttnRes 关键组件的消融（16 层模型）。

| 变体 | 损失 |
|---|---|
| 基线（PreNorm） | 1.766 |
| DenseFormer | 1.767 |
| mHC | 1.747 |
| AttnRes：完整（Full） | 1.737 |
| 完整，配随输入而变的查询 | 1.731 |
| 完整，配与输入无关的混合 | 1.749 |
| 完整，配 sigmoid | 1.741 |
| 完整，去掉 RMSNorm | 1.743 |
| SWA（$W = 1 + 8$） | 1.764 |
| Block（$S=4$） | 1.746 |
| Block，配多头（$H=16$） | 1.752 |
| Block，去掉 RMSNorm | 1.750 |

**图 6** 块大小对验证损失的影响（16 层模型）。（译注：图中 Block AttnRes 随块大小 $S$ 取 32、16、8、4、2 时，验证损失依次为 1.757、1.753、1.748、1.746、1.746；虚线为基线 1.766 与完整 AttnRes 即 $S=1$ 的 1.737。）

**与此前方法的比较。** 本文把 AttnRes 与 PreNorm 基线（损失 1.766）和两个有代表性的残差推广方法比较。DenseFormer 让每层能访问全部先前输出，却用固定的、与输入无关的标量系数把它们组合起来，相对基线没有收益（1.767），凸显了随输入而变的加权的重要性。mHC 借 $m$ 条并行流与学到的混合矩阵引入输入依赖，改进到 1.747。AttnRes 再往前一步，用 softmax 注意力做显式的内容相关选择：完整 AttnRes 达到 1.737，Block AttnRes 达到 1.746，只用每层单个查询向量就优于这两种方法。

**跨层访问。** 本文比较三种跨层访问粒度。完整 AttnRes 直接来自时间—深度对偶（第 3 节），对全部先前层做注意力，损失最低（1.737）。一个降低其内存代价的简单办法是滑动窗口聚合（sliding-window aggregation, SWA）：只保留最近的 $W{=}8$ 个层输出加词嵌入；它优于基线（1.764），却远逊于完整与 Block AttnRes，说明有选择地访问远处的层，比对许多近处的层做注意力更重要。

Block AttnRes 给出更好的权衡：块大小 $S{=}4$ 时达到 1.746，每层内存开销还保持恒定。块大小图扫了 $S$ 的全谱，从 $S{=}1$（即完整 AttnRes）到越来越粗的分组。$S$ 变大时损失平缓退化，$S{=}2, 4, 8$ 都落在 1.746 附近，更大的块（$S{=}16, 32$）则向基线靠拢。实践中，本文出于基础设施效率把块数固定在约 8（第 4 节）。将来硬件缓解内存容量约束后，采用更细的块大小或完整 AttnRes，是进一步提升性能的自然路径。

**组件设计。** 本文还逐项消融注意力机制的各个组件：

- **随输入而变的查询。** 一个自然的推广是把查询改成随输入而变，由当前隐藏状态投影得到。这能把损失进一步降到 1.731，却给每层引入一个 $d \times d$ 投影，且解码时需要顺序的内存访问，因此本文默认仍用学到的查询。
- **与输入无关的混合。** 本文去掉查询与键，换成可学习的、与输入无关的标量来加权先前层，结果性能变差（1.749 对 1.737）。
- **softmax 与 sigmoid 之争。** 把 softmax 换成 sigmoid 会让性能退化（1.741）。本文把原因归于 softmax 的竞争性归一化：它迫使在各来源之间作出更尖锐的选择。
- **多头注意力。** 本文在 Block AttnRes 上测试逐头的深度聚合（$H{=}16$），让不同的通道组关注不同的来源层。结果性能变差（1.752 对 1.746），说明最优的深度方向混合在各通道间大体一致：一层的输出只要相关，就是整体相关。
- **键上的 RMSNorm。** 去掉 RMSNorm 会让完整 AttnRes（1.743）与 Block AttnRes（1.750）双双退化。对完整 AttnRes，它防止单个输出天然较大的层主导 softmax。对 Block AttnRes 这件事更关键，因为块级表征累积的层更多，幅值差异可能长得很大；RMSNorm 防止这些差异扭曲注意力权重。

### 5.4 分析

#### 5.4.1 最优架构

**图 7** 固定算力（约 $6.5\times10^{19}$ FLOPs、约 $2.3\times10^8$ 激活参数）下的架构扫描。每个格子报告一种 $(d_{\mathrm{model}}/L_b,\, H/L_b)$ 配置的验证损失，其中 $L_b = L/2$ 是 Transformer 块数；星号标出最优点。左为基线，右为 Attention Residuals；横轴为 $d_{\mathrm{model}}/L_b$，纵轴为 $H/L_b$。（译注：两张热图的全部读数如下。基线——$H/L_b$ 取 0.7、0.6、0.5、0.4、0.3 五行，$d_{\mathrm{model}}/L_b$ 取 15、30、45、60、75 五列：0.7 行 2.017、1.909、1.875、1.851、1.858；0.6 行 1.990、1.902、1.862、1.852、1.862；0.5 行 1.973、1.883、1.859、1.849、1.854；0.4 行 1.952、1.868、1.850、1.849、1.857；0.3 行 1.926、1.857、1.851、1.847、1.858。AttnRes——0.7 行 1.954、1.890、1.843、1.828、1.824；0.6 行 1.931、1.863、1.830、1.817、1.818；0.5 行 1.917、1.841、1.819、1.812、1.817；0.4 行 1.893、1.823、1.815、1.813、1.813；0.3 行 1.877、1.816、1.802、1.806、1.820。）

为理解 AttnRes 怎样重塑最优的架构缩放，本文在固定算力与参数预算下做了一项受控的容量再分配研究。核心问题是：AttnRes 是否改变了人们偏好的深度—宽度—注意力权衡；特别是，鉴于它在深度维度上可能的强项，它是否比常规 Transformer 设计经验更偏爱更深的模型。为隔离与深度直接耦合的结构因素，本文按内部经验观察固定每个专家的 MLP 扩张比（$d_{\mathrm{ff}}/d_{\mathrm{model}} \approx 0.45$）。再固定总训练算力（FLOPs $\approx 6.5\times10^{19}$）与激活参数（$\approx 2.3\times10^8$），确保任何性能差异纯粹来自架构再分配，而非整体容量差异。在这一受约束的预算下，本文在一个 $5\times5$ 网格上枚举 25 种配置：$d_{\mathrm{model}}/L_b \in \{15, 30, 45, 60, 75\}$，$H/L_b \in \{0.3, 0.4, 0.5, 0.6, 0.7\}$，其中 $L_b = L/2$ 是 Transformer 块数，$H$ 是注意力头数。结果见图 7。

两张热图呈现共同形态：损失随 $d_{\mathrm{model}}/L_b$ 增大而下降、随 $H/L_b$ 减小而下降，两种方法的最优点都落在 $H/L_b \approx 0.3$。在共同趋势之外，AttnRes 在 25 种配置的每一种上损失都低于基线，低 0.019–0.063。最显眼的差别在最优点的位置：基线在 $d_{\mathrm{model}}/L_b \approx 60$ 处取得最低损失（1.847），AttnRes 把它移到 $d_{\mathrm{model}}/L_b \approx 45$（1.802）。在固定参数预算下，更低的 $d_{\mathrm{model}}/L_b$ 对应更深、更窄的网络，说明 AttnRes 能更有效地利用额外深度。本文提醒，这种对深度的偏好并不直接等于部署建议，因为更深的模型由于顺序计算，推理延迟一般更高。这项扫描的定位是诊断工具，用来揭示 AttnRes 在哪里收益最大；这份深度偏好可以与推理成本一并纳入架构选择。

#### 5.4.2 分析学到的 AttnRes 形态

**图 8** 一个 16 头模型在完整（上）与分块（下）Attention Residuals 下的深度方向注意力权重分布，对词元取平均。该模型有 16 个注意力层与 16 个 MLP 层。每一行显示第 $l$ 个注意力层（左）或 MLP 层（右）怎样把权重分给先前来源。对角占优表明局部性仍是主要的信息通路；来源 0（嵌入）上持续存在的权重与偶尔出现的非对角集中，则揭示出学到的跳跃连接。分块注意力（$N=8$）以更尖锐、更果断的权重分布收回了基本结构。

本文在图 8 中可视化 16 头模型（表 2）学到的权重 $\alpha_{i \to l}$，同时给出完整与分块（$N{=}8$）AttnRes。每张热图显示第 $l$ 个注意力层或 MLP 层（行）怎样把注意力分给先前来源（列），注意力前与 MLP 前的层分开呈现。本文强调三点观察：

- **局部性得到保留。** 每层最强地关注它的直接前一层，但有选择的非对角集中也会出现（例如第 4 层关注早期来源，分块设定下第 15–16 层回望更早来源），说明学到的跳跃连接超出了标准残差通路。
- **层间分工。** 嵌入 $\bm{h}_1$ 全程保有不可忽略的权重，在注意力前的层里尤其如此。MLP 前的输入对近期表征呈现更尖锐的对角依赖，注意力前的输入则保持更宽的感受野，这与注意力跨层路由信息、MLP 局部运算的图景一致。
- **Block AttnRes 保住结构。** 对角占优、嵌入持续存在、层间分工，这三样都从完整变体传到了分块变体，说明分块压缩起着隐式正则化的作用，同时保住了关键的信息通路。

## 6 讨论

### 6.1 序列—深度对偶

残差连接经固定递推 $\bm{h}_{l} = \bm{h}_{l-1} + f_{l-1}(\bm{h}_{l-1})$ 沿深度传播信息，与 RNN 沿时间传播信息很像。测试时训练（Test-Time Training, TTT）把这一类比的序列一侧形式化了（参看快速权重编程器 Fast Weight Programmers）：把每个递推步写成对一个自监督损失的梯度下降：

$$\mathbf{W}_t = \mathbf{W}_{t-1} - \eta\,\nabla\ell(\mathbf{W}_{t-1};\, \bm{x}_t),$$

（公式说明：$\mathbf{W}_t$ 是随词元更新的快速权重状态，$\eta$ 是步长，$\ell$ 由一个慢网络参数化，每个词元更新一次。）

当 $f$ 是线性的，这就退化为朴素线性注意力 $\mathbf{S}_{t} = \mathbf{S}_{t-1} + \bm{k}_t\bm{v}_t^\top$。标准残差沿深度呈现同样的加法形式：$\bm{h}_l$ 充当状态，每层 $f_l$ 充当一次“梯度步”。

如已有工作所指出的，这一对偶可以推广到更丰富的变体（表 5）。序列一侧随数据而变的门控，对应深度一侧的 Highway 网络；delta 规则（delta rule）对应 DDL；MRLA 则镜像 GLA 的门控线性注意力。这些方法都在递推范式内部改良递推更新。AttnRes 再往前一步，用直接的跨层注意力取代深度方向的递推，正如 Transformer 用自注意力取代了时间方向的递推。当前架构的层数还稳稳落在 softmax 注意力的实用范围内，因此本文采用朴素的深度方向注意力。把表达力更强、内存又更省的替代方案（例如线性复杂度的方案）纳进来，是将来工作的自然方向。

**表 5** 残差更新机制比较。*权重（Weight）*：混合系数是架构固定的（Fixed）、学到后固定的（Static，训练后不再变），还是随输入而变的（Dynamic）。*来源（Source）*：第 $l$ 层能访问哪些更早的表征。为表述清楚，多数公式省略了归一化。

| 方法 | 更新规则 | 权重 | 来源 |
|---|---|---|---|
| **单状态递推：第 $l$ 层只收到 $\bm{h}_{l-1}$** | | | |
| Residual | $\bm{h}_{l} = \bm{h}_{l-1} + f_{l-1}(\bm{h}_{l-1})$ | Fixed | $\bm{h}_{l-1}$ |
| ReZero | $\bm{h}_{l} = \bm{h}_{l-1} + \alpha_{l} \cdot f_{l-1}(\bm{h}_{l-1})$ | Static | $\bm{h}_{l-1}$ |
| LayerScale | $\bm{h}_{l} = \bm{h}_{l-1} + \mathrm{diag}(\bm{\lambda}_{l}) \cdot f_{l-1}(\bm{h}_{l-1})$ | Static | $\bm{h}_{l-1}$ |
| Highway | $\bm{h}_{l} = (1{-}\bm{g}_{l}) \odot \bm{h}_{l-1} + \bm{g}_{l} \odot f_{l-1}(\bm{h}_{l-1})$ | Dynamic | $\bm{h}_{l-1}$ |
| DeepNorm | $\bm{h}_{l} = \mathrm{Norm}(\alpha \, \bm{h}_{l-1} + f_{l-1}(\bm{h}_{l-1}))$ | Fixed | $\bm{h}_{l-1}$ |
| KEEL | $\bm{h}_{l} = \mathrm{Norm}(\alpha \, \bm{h}_{l-1} + f_{l-1}(\mathrm{Norm}(\bm{h}_{l-1})))$ | Fixed | $\bm{h}_{l-1}$ |
| **多状态递推：第 $l$ 层收到 $m$ 条流** | | | |
| SiameseNorm | $\bm{h}^{1}_{l} = \mathrm{Norm}(\bm{h}^{1}_{l-1} + \bm{y}_{l-1})$；$\bm{h}^{2}_{l} = \bm{h}^{2}_{l-1} + \bm{y}_{l-1}$ | Fixed | 2 条流 |
| HC/mHC | $\mathbf{H}_{l} = \mathbf{H}_{l-1}\mathbf{A}_{l} + f_{l-1}(\mathbf{H}_{l-1}\bm{\alpha}_{l-1})\,\bm{\beta}_{l-1}^\top$ | Dynamic | $m$ 条流 |
| DDL | $\mathbf{H}_{l} = (\mathbf{I} - \beta_{l} \bm{k}_{l}\bm{k}_{l}^\top)\mathbf{H}_{l-1} + \beta_{l} \bm{k}_{l}\bm{v}_{l}^\top$ | Dynamic | $d_v$ 条流 |
| **跨层访问：第 $l$ 层能访问单个的更早层输出** | | | |
| DenseNet | $\bm{h}_{l} = \mathrm{ConvPool}([\bm{h}_1;\, f_1(\bm{h}_1);\, \ldots;\, f_{l-1}(\bm{h}_{l-1})])$ | Static | $[\bm{h}_1,\ldots,\bm{h}_{l-1}]$ |
| DenseFormer | $\bm{h}_{l} = \alpha_{0 \to l}\,\bm{h}_1 + \sum_{i=1}^{l-1} \alpha_{i \to l}\,f_i(\bm{h}_i)$ | Static | $[\bm{h}_1,\ldots,\bm{h}_{l-1}]$ |
| MRLA¹ | $\bm{h}_{l} = \sum_{i=1}^{l-1} \sigma\!\bigl(\mathrm{ConvPool}(f_{l-1}(\bm{h}_{l-1}))\bigr)^{\!\top} \sigma\!\bigl(\mathrm{ConvPool}(f_{i}(\bm{h}_i))\bigr) \, \mathrm{Conv}(f_i(\bm{h}_i))$ | Dynamic | $[\bm{h}_1,\ldots,\bm{h}_{l-1}]$ |
| AttnRes（本文）完整（Full）² | $\bm{h}_{l} \propto \sum_{i=0}^{l-1} \phi(\bm{w}_{l}, \bm{k}_i) \, \bm{v}_i$ | Dynamic | $[\bm{h}_1,\ldots,\bm{h}_{l-1}]$ |
| AttnRes（本文）分块（Block）³ | $\bm{h}_{l} \propto \sum_{i=0}^{n-1} \phi(\bm{w}_{l}, \bm{k}_i) \, \bm{v}_i + \phi(\bm{w}_{l}, \bm{k}_n^j) \, \bm{v}_n^j$ | Dynamic | $[\bm{b}_0,\ldots,\bm{b}_{n{-}1},\bm{b}_n^j]$ |

（译注：原表脚注——¹ ConvPool：先池化再卷积（通道投影）。² $\phi(\bm{q},\bm{k}) = \exp\left(\bm{q}^\top \operatorname{RMSNorm}(\bm{k})\right)$；$\bm{k}_i = \bm{v}_i$；$\bm{v}_0 = \bm{h}_1$，$\bm{v}_{i\geq1} = f_i(\bm{h}_i)$；softmax 在全部来源上联合归一化。³ $\phi$ 与归一化同完整版；$\bm{v}_i = \bm{b}_i$，$\bm{v}_n^j = \bm{b}_n^j$。）

### 6.2 残差连接作为结构化矩阵

上文讨论的残差变体，都可以看成对先前层输出的加权聚合。本文用一个*深度混合矩阵*（depth mixing matrix）$\mathbf{M} \in \mathbb{R}^{L \times L}$ 把这件事形式化，其中 $\mathbf{M}_{i \to l}$ 是第 $l$ 层分给第 $i$ 层输出的权重。各变体的差别在于这些权重怎样产生（固定、学到，还是随输入而变），以及 $\mathbf{M}$ 是被约束为低秩，还是可以稠密。$\mathbf{M}$ 的半可分秩（semiseparable rank）为比较它们提供了一副统一的透镜。

具体地，第 $l$ 层的输入是 $\bm{h}_l = \sum_{i=0}^{l-1} \mathbf{M}_{i \to l}\, \bm{v}_i$，其中 $\bm{v}_0 = \bm{h}_1$（嵌入），$i \geq 1$ 时 $\bm{v}_i = f_i(\bm{h}_i)$。图 9 可视化了有代表性方法的 $\mathbf{M}$；下面逐个推导。

- **标准残差**，$\bm{h}_{l} = \bm{h}_{l-1} + f_{l-1}(\bm{h}_{l-1})$。展开得 $\bm{h}_l = \sum_{i=0}^{l-1} \bm{v}_i$，因此对全部 $i < l$ 都有 $\mathbf{M}_{i \to l} = 1$，$\mathbf{M}$ 是一个全 1 的下三角矩阵：

$$\begin{bmatrix} \bm{h}_1 \\ \bm{h}_2 \\ \vdots \\ \bm{h}_L \end{bmatrix} = \begin{bmatrix} 1 & & & \\ 1 & 1 & & \\ \vdots & \vdots & \ddots & \\ 1 & 1 & \cdots & 1 \end{bmatrix} \begin{bmatrix} \bm{v}_0 \\ \bm{v}_1 \\ \vdots \\ \bm{v}_{L-1} \end{bmatrix}$$

- **Highway**，$\bm{h}_{l} = (1{-}g_{l})\,\bm{h}_{l-1} + g_{l}\,f_{l-1}(\bm{h}_{l-1})$（此处为表述清楚写成标量门控，逐元素推广是直接的）。定义承载连乘（carry product）$\gamma_{i \to l}^{\times} \coloneqq \prod_{j=i+1}^{l}(1-g_j)$，则权重为：嵌入对应 $\mathbf{M}_{0 \to l} = \gamma_{1 \to l}^{\times}$，$i \geq 1$ 时 $\mathbf{M}_{i \to l} = g_{i+1}\,\gamma_{i+1 \to l}^{\times}$。累积连乘经标量门控分解，因此 $\mathbf{M}$ 是 1-半可分的，与标准残差同秩，只是权重随输入而变。这些权重按构造求和为 1，使 Highway 成为一个不用 softmax 的深度方向折棍注意力（stick-breaking attention）实例。
- **(m)HC** 维护 $m$ 条并行流 $\mathbf{H}_{l} \in \mathbb{R}^{d \times m}$，按下面的式子更新：

$$\mathbf{H}_{l} = \mathbf{H}_{l-1} \mathbf{A}_{l} + f_{l-1}(\mathbf{H}_{l-1} \bm{\alpha}_{l-1})\, \bm{\beta}_{l-1}^\top,$$

（公式说明：$\mathbf{A}_{l} \in \mathbb{R}^{m \times m}$ 是学到的转移矩阵，$\bm{\alpha}_{l-1} \in \mathbb{R}^{m}$ 把各条流混成给 $f_{l-1}$ 的单个输入，$\bm{\beta}_{l-1} \in \mathbb{R}^{m}$ 把输出再分发回各条流。）

展开递推得到有效权重：

$$\mathbf{M}_{i \to l} = \bm{\beta}_{i}^\top \, \mathbf{A}_{i+1 \to l}^{\times} \, \bm{\alpha}_{l}, \tag{9}$$

（公式说明：$\mathbf{A}_{i \to j}^{\times} \coloneqq \prod_{k=i+1}^{j} \mathbf{A}_k$，是中间各层转移矩阵的连乘。）

$m \times m$ 的转移使 $\mathbf{M}$ 成为 $m$-半可分的。mHC 还进一步约束每个 $\mathbf{A}_l$ 为双随机矩阵（doubly stochastic），以稳定深度方向的累积连乘。

- **完整 AttnRes** 经 $\phi(\bm{w}_l, \bm{k}_i) = \exp\left(\bm{w}_l^\top \operatorname{RMSNorm}(\bm{k}_i)\right)$ 加归一化算出 $\mathbf{M}_{i \to l} = \alpha_{i \to l}$，其中 $\bm{k}_i = \bm{v}_i$ 是随输入而变的层输出，得到稠密的、秩为 $L$ 的 $\mathbf{M}$。
- **Block AttnRes** 把各层划分为 $N$ 个块 $\mathcal{B}_1, \ldots, \mathcal{B}_N$。对已完成的更早块 $\mathcal{B}_n$ 中的来源 $i$，它们共用同一个块级键／值 $\bm{b}_n$，因此对每个 $i \in \mathcal{B}_n$ 都有 $\mathbf{M}_{i \to l} = \alpha_{n \to l}$。在当前块内，每层还额外对不断演进的部分和 $\bm{b}_n^{i-1}$ 做注意力，为每个块内位置引入一个额外的不同来源。因此 $\mathbf{M}$ 的有效秩落在 $N$ 与 $N + S$ 之间（$S$ 是块大小），在标准残差（$N{=}1$）与完整 AttnRes（$N{=}L$）之间插值。

**图 9** 四种残差变体的深度混合矩阵 $\mathbf{M}$（$L{=}4$；Block AttnRes 用块大小 $S{=}2$）。Highway 为表述清楚用标量门控呈现。AttnRes 的面板显示未归一化的 $\phi$ 分数；背景色把共享同一来源（完整 AttnRes）或同一来源块（Block AttnRes）的条目分组。

**实用性。** 结构化矩阵视角有两个用处。其一，它能给出单从递推形式看不出的分析洞见。例如 AttnRes 随输入而变的 $\mathbf{M}$ 揭示出深度方向的注意力汇（attention sinks，见第 5.4.2 节）：某些层不论输入如何都稳定地吸引高权重，与序列方向注意力里的同一现象互为镜像。其二，它通过暴露核函数 $\phi$ 的哪些性质要紧，为新设计提供线索。例如当 $\phi$ 可以分解为 $\phi(\bm{q}, \bm{k}) = \varphi(\bm{q})^\top \varphi(\bm{k})$（$\varphi$ 是某个特征映射）时，深度方向注意力就塌缩成一个递推——正是上文 MRLA–GLA 与 DDL–DeltaNet 对应关系背后的结构。

**此前的残差作为深度方向线性注意力。** 结构化矩阵视角还进一步连到序列—深度对偶：它表明已有的残差变体实际上都是深度轴上的*线性*注意力实例。例如展开的 (m)HC 权重 $\mathbf{M}_{i \to l} = \boldsymbol{\beta}_i^\top \mathbf{A}_{i+1 \to l}^{\times} \boldsymbol{\alpha}_l$（公式 9）有一种自然的注意力解读：$\boldsymbol{\alpha}_l$ 扮演第 $l$ 层发出的查询，$\boldsymbol{\beta}_i$ 充当概括第 $i$ 层贡献的键，累积转移 $\mathbf{A}_{i+1 \to l}^{\times}$ 充当深度相对位置算子，支配查询—键跨中间各层的相互作用。值得注意的是，$m$ 条并行流对应沿深度轴的状态扩张（state expansion），把递推状态从 $d$ 扩到 $d \times m$，从而提高 $\mathbf{M}$ 的半可分秩。已有工作表明，把 $\mathbf{A}_{i+1 \to l}^{\times}$ 换成单位矩阵仍有有竞争力的性能，凸显了状态扩张的作用。在这副透镜下，(m)HC 这类方法充当带矩阵值状态的深度方向*线性*注意力，AttnRes 则充当深度方向 *softmax* 注意力。

## 7 相关工作

**归一化、缩放与深度稳定性。** 标准残差更新 $\bm{h}_{l+1} = \bm{h}_l + f_l(\bm{h}_l)$ 呈现一重根本张力，发生在*归一化位置*与*梯度传播*之间。后归一化（PostNorm）保持幅值有界，却扭曲梯度，因为残差路径上反复归一化会复合成深度上的梯度消失。前归一化（PreNorm）恢复了一条干净的恒等路径，却引入无界的幅值增长：$\|\bm{h}_l\|$ 按 $O(L)$ 增长，每层的相对贡献随之缩小，迫使深层产生越来越大的输出，并限制了有效深度。后来的工作经缩放残差路径、混合归一化、放大的跳跃连接，或学到的逐元素门控，来调和这两个诉求（见表 5）。AttnRes 绕开这重张力：它用对单个更早层输出的选择性聚合取代加法递推，既避开 PreNorm 的累积幅值增长，也避开 PostNorm 反复的尺度收缩。

**多状态递推。** 上面全部单状态方法都只让第 $l$ 层以 $\bm{h}_{l-1}$ 为条件，从中无法有选择地取回单个更早层的贡献。若干方法把递推拓宽为多条并行流来应对：Hyper-Connections 及其稳定化变体 mHC 维护 $m$ 条带学到混合矩阵的流；DDL 维护一个经 delta 规则擦除—写入机制更新的矩阵状态；SiameseNorm 维护两条参数共享的流——一条 PreNorm、一条 PostNorm——以同时保住恒等梯度与有界表征。这些方法缓解了信息压缩，却仍以直接前驱的状态为条件；AttnRes 与它们正交：它提供对单个更早层输出的选择性访问，同时与任何归一化或门控方案都兼容。本文在第 6.2 节讨论了它与 Hyper-Connections 的形式联系。

**跨层连接。** 另一条工作线绕开单状态瓶颈，让每层直接访问单个的更早层输出。最简单的办法用静态权重：DenseNet 拼接全部先前特征图；ELMo 用学到的标量权重对各层表征算 softmax 加权和；DenseFormer 与 ANCRe 给出学到的逐层标量系数，训练后固定。对随输入而变的聚合，MUDDFormer 经一个小 MLP 在四条解耦的流上生成随位置而变的权重；MRLA 对全部先前层施加逐元素 sigmoid 门控，不过它可分离的查询—键乘积更接近线性注意力，而非基于 softmax 的检索。另一些方法用完整的跨层访问换更有针对性的设计：Value Residual Learning 只访问单个更早层；LAuReL 用对前 $k$ 个激活的低秩投影扩充残差；Dreamer 把序列注意力与深度注意力及稀疏专家结合起来。AttnRes 把经 softmax 归一化的、随输入而变的权重，与经每层单个 $d$ 维伪查询对全部先前层的选择性访问结合起来，并引入把代价从 $O(L^2)$ 降到 $O(LN)$ 的块结构。基于缓存的流水线通信与两阶段计算策略（第 4 节）使 Block AttnRes 在大规模下实用，开销可忽略。

## 结论

（译注：原文结论一节不带编号。）

受序列与深度之间对偶的启发，本文提出 AttnRes，用学到的、随输入而变的深度方向注意力，取代固定的、均匀的残差累积。本文经消融研究与缩放律实验验证了这一方法，表明其收益跨规模成立。完整 AttnRes 必须在每层访问全部先前层输出，跨层聚合的内存占用随之长成 $O(Ld)$，这在当前硬件上对大规模模型是难以承受的。因此本文提出 Block AttnRes：把各层划分为 $N$ 个块，对块级表征做注意力。经验上，用约 8 个块就能收回完整 AttnRes 的大部分收益；将来硬件约束放松后，更细粒度的分块仍是一个有前景的方向。再加上跨阶段缓存与两阶段计算策略，Block AttnRes 在大规模下实用，只带来边际的训练开销与极小的推理开销。

## 参考文献（原文照录）

[1] Jacob Austin et al. Program Synthesis with Large Language Models. 2021. arXiv: 2108.07732 [cs.PL]. URL: https://arxiv.org/abs/2108.07732.

[2] Thomas Bachlechner et al. ReZero is All You Need: Fast Convergence at Large Depth. 2020. arXiv: 2003.04887 [cs.LG]. URL: https://arxiv.org/abs/2003.04887.

[3] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural Machine Translation by Jointly Learning to Align and Translate. 2016. arXiv: 1409.0473 [cs.CL]. URL: https://arxiv.org/abs/1409.0473.

[4] Chen Chen and Lai Wei. Post-LayerNorm Is Back: Stable, ExpressivE, and Deep. 2026. arXiv: 2601.19895 [cs.LG]. URL: https://arxiv.org/abs/2601.19895.

[5] Mark Chen et al. Evaluating Large Language Models Trained on Code. 2021. arXiv: 2107.03374 [cs.LG]. URL: https://arxiv.org/abs/2107.03374.

[6] Peter Clark et al. “Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge”. In: arXiv:1803.05457v1 (2018).

[7] Karl Cobbe et al. Training Verifiers to Solve Math Word Problems. 2021. arXiv: 2110.14168 [cs.LG]. URL: https://arxiv.org/abs/2110.14168.

[8] Tri Dao and Albert Gu. “Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality”. In: CoRR abs/2405.21060 (2024). DOI: 10.48550/ARXIV.2405.21060. arXiv: 2405.21060. URL: https://doi.org/10.48550/arXiv.2405.21060.

[9] DeepSeek-AI et al. DeepSeek-V3 Technical Report. 2025. arXiv: 2412.19437 [cs.CL]. URL: https://arxiv.org/abs/2412.19437.

[10] Yanwen Fang et al. Cross-Layer Retrospective Retrieving via Layer Attention. 2023. arXiv: 2302.03985 [cs.CV]. URL: https://arxiv.org/abs/2302.03985.

[11] Andrey Gromov et al. The Unreasonable Ineffectiveness of the Deeper Layers. 2025. arXiv: 2403.17887 [cs.CL]. URL: https://arxiv.org/abs/2403.17887.

[12] Kaiming He et al. Deep Residual Learning for Image Recognition. 2015. arXiv: 1512.03385 [cs.CV]. URL: https://arxiv.org/abs/1512.03385.

[13] Dan Hendrycks et al. Measuring Massive Multitask Language Understanding. 2021. arXiv: 2009.03300 [cs.CY]. URL: https://arxiv.org/abs/2009.03300.

[14] Dan Hendrycks et al. Measuring Mathematical Problem Solving With the MATH Dataset. 2021. arXiv: 2103.03874 [cs.LG]. URL: https://arxiv.org/abs/2103.03874.

[15] Jordan Hoffmann et al. Training Compute-Optimal Large Language Models. 2022. arXiv: 2203.15556 [cs.CL]. URL: https://arxiv.org/abs/2203.15556.

[16] Shengding Hu et al. MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies. 2024. arXiv: 2404.06395 [cs.CL]. URL: https://arxiv.org/abs/2404.06395.

[17] Gao Huang et al. Densely Connected Convolutional Networks. 2018. arXiv: 1608.06993 [cs.CV]. URL: https://arxiv.org/abs/1608.06993.

[18] Yanping Huang et al. “GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism”. In: Advances in NeurIPS. 2019.

[19] Yuzhen Huang et al. “C-eval: A multi-level multi-discipline chinese evaluation suite for foundation models”. In: Advances in NeurIPS 36 (2023), pp. 62991–63010.

[20] Robert A. Jacobs et al. “Adaptive Mixtures of Local Experts”. In: Neural Computation 3.1 (1991), pp. 79–87. DOI: 10.1162/neco.1991.3.1.79.

[21] Mandar Joshi et al. “Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension”. In: arXiv preprint arXiv:1705.03551 (2017).

[22] Jared Kaplan et al. Scaling Laws for Neural Language Models. 2020. arXiv: 2001.08361 [cs.LG]. URL: https://arxiv.org/abs/2001.08361.

[23] Angelos Katharopoulos et al. “Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention”. In: Proceedings of ICML. Ed. by Hal Daumé III and Aarti Singh. PMLR, 2020, pp. 5156–5165. URL: https://proceedings.mlr.press/v119/katharopoulos20a.html.

[24] Jonas Knupp et al. Depth-Recurrent Attention Mixtures: Giving Latent Reasoning the Attention it Deserves. 2026. arXiv: 2601.21582 [cs.AI]. URL: https://arxiv.org/abs/2601.21582.

[25] Aitor Lewkowycz et al. Solving Quantitative Reasoning Problems with Language Models. 2022. arXiv: 2206.14858 [cs.CL]. URL: https://arxiv.org/abs/2206.14858.

[26] Haonan Li et al. “CMMLU: Measuring massive multitask language understanding in Chinese”. In: Findings of the Association for Computational Linguistics: ACL 2024. Ed. by Lun-Wei Ku, Andre Martins, and Vivek Srikumar. Bangkok, Thailand: Association for Computational Linguistics, Aug. 2024, pp. 11260–11285. DOI: 10.18653/v1/2024.findings-acl.671. URL: https://aclanthology.org/2024.findings-acl.671/.

[27] Tianyu Li et al. SiameseNorm: Breaking the Barrier to Reconciling Pre/Post-Norm. 2026. arXiv: 2602.08064 [cs.LG]. URL: https://arxiv.org/abs/2602.08064.

[28] Jingyuan Liu et al. Muon is Scalable for LLM Training. 2025. arXiv: 2502.16982 [cs.LG]. URL: https://arxiv.org/abs/2502.16982.

[29] Brian Mak and Jeffrey Flanigan. Residual Matrix Transformers: Scaling the Size of the Residual Stream. 2025. arXiv: 2506.22696 [cs.LG]. URL: https://arxiv.org/abs/2506.22696.

[30] Gaurav Menghani, Ravi Kumar, and Sanjiv Kumar. LAuReL: Learned Augmented Residual Layer. 2025. arXiv: 2411.07501 [cs.LG]. URL: https://arxiv.org/abs/2411.07501.

[31] Maxim Milakov and Natalia Gimelshein. Online normalizer calculation for softmax. 2018. arXiv: 1805.02867 [cs.PF]. URL: https://arxiv.org/abs/1805.02867.

[32] Tsendsuren Munkhdalai et al. “Metalearned Neural Memory”. In: ArXiv abs/1907.09720 (2019). URL: https://api.semanticscholar.org/CorpusID:198179407.

[33] Deepak Narayanan et al. Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM. 2021. arXiv: 2104.04473 [cs.CL]. URL: https://arxiv.org/abs/2104.04473.

[34] Toan Q. Nguyen and Julian Salazar. “Transformers without Tears: Improving the Normalization of Self-Attention”. In: Proceedings of IWSLT. Ed. by Jan Niehues et al. 2019. URL: https://aclanthology.org/2019.iwslt-1.17/.

[35] OpenAI et al. GPT-4 Technical Report. 2024. arXiv: 2303.08774 [cs.CL]. URL: https://arxiv.org/abs/2303.08774.

[36] Matteo Pagliardini et al. DenseFormer: Enhancing Information Flow in Transformers via Depth Weighted Averaging. 2024. arXiv: 2402.02622 [cs.CL]. URL: https://arxiv.org/abs/2402.02622.

[37] Bowen Peng et al. “Yarn: Efficient context window extension of large language models”. In: arXiv preprint arXiv:2309.00071 (2023).

[38] Matthew E. Peters et al. “Deep Contextualized Word Representations”. In: Proceedings of NAACL. 2018, pp. 2227–2237. URL: https://aclanthology.org/N18-1202/.

[39] Reiner Pope et al. Efficiently Scaling Transformer Inference. 2022. arXiv: 2211.05102 [cs.LG].

[40] Zhen Qin et al. HGRN2: Gated Linear RNNs with State Expansion. 2024. arXiv: 2404.07904 [cs.CL].

[41] David Rein et al. “Gpqa: A graduate-level google-proof q&a benchmark”. In: First Conference on Language Modeling. 2024.

[42] Imanol Schlag, Kazuki Irie, and Jürgen Schmidhuber. “Linear Transformers Are Secretly Fast Weight Programmers”. In: Proceedings of ICML. Ed. by Marina Meila and Tong Zhang. PMLR, 2021, pp. 9355–9366. URL: https://proceedings.mlr.press/v139/schlag21a.html.

[43] Jürgen Schmidhuber. “Learning to control fast-weight memories: An alternative to dynamic recurrent networks”. In: Neural Computation 4.1 (1992), pp. 131–139.

[44] Freda Shi et al. Language Models are Multilingual Chain-of-Thought Reasoners. 2022. arXiv: 2210.03057 [cs.CL]. URL: https://arxiv.org/abs/2210.03057.

[45] Rupesh Kumar Srivastava, Klaus Greff, and Jürgen Schmidhuber. Highway Networks. 2015. arXiv: 1505.00387 [cs.LG]. URL: https://arxiv.org/abs/1505.00387.

[46] Yu Sun et al. “Learning to (Learn at Test Time): RNNs with Expressive Hidden States”. In: ArXiv abs/2407.04620 (2024). URL: https://api.semanticscholar.org/CorpusID:271039606.

[47] Yutao Sun et al. Retentive Network: A Successor to Transformer for Large Language Models. 2023. arXiv: 2307.08621 [cs.CL].

[48] Mirac Suzgun et al. “Challenging big-bench tasks and whether chain-of-thought can solve them”. In: arXiv preprint arXiv:2210.09261 (2022).

[49] Shawn Tan et al. “Scaling Stick-Breaking Attention: An Efficient Implementation and In-depth Study”. In: Proceedings of ICLR. 2025.

[50] Hugo Touvron et al. Going deeper with Image Transformers. 2021. arXiv: 2103.17239 [cs.CV]. URL: https://arxiv.org/abs/2103.17239.

[51] Hugo Touvron et al. LLaMA: Open and Efficient Foundation Language Models. 2023. arXiv: 2302.13971 [cs.CL].

[52] Ashish Vaswani et al. “Attention is All you Need”. In: Advances in NeurIPS. Ed. by I. Guyon et al. Curran Associates, Inc., 2017. URL: https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf.

[53] Ashish Vaswani et al. “Attention is All you Need”. In: Advances in NeurIPS. Ed. by I. Guyon et al. Vol. 30. Curran Associates, Inc., 2017. URL: https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf.

[54] Hongyu Wang et al. DeepNet: Scaling Transformers to 1,000 Layers. 2022. arXiv: 2203.00555 [cs.CL]. URL: https://arxiv.org/abs/2203.00555.

[55] Yubo Wang et al. “Mmlu-pro: A more robust and challenging multi-task language understanding benchmark”. In: Advances in NeurIPS 37 (2024), pp. 95266–95290.

[56] Da Xiao et al. “MUDDFormer: Breaking Residual Bottlenecks in Transformers via Multiway Dynamic Dense Connections”. In: Proceedings of ICML. 2025.

[57] Guangxuan Xiao et al. “Efficient streaming language models with attention sinks”. In: arXiv preprint arXiv:2309.17453 (2023).

[58] Tian Xie. Your DeepSeek mHC Might Not Need the “m”. Zhihu blog post. 2026. URL: https://zhuanlan.zhihu.com/p/2010852389670908320.

[59] Zhenda Xie et al. mHC: Manifold-Constrained Hyper-Connections. 2026. arXiv: 2512.24880 [cs.CL]. URL: https://arxiv.org/abs/2512.24880.

[60] Ruibin Xiong et al. On Layer Normalization in the Transformer Architecture. 2020. arXiv: 2002.04745 [cs.LG]. URL: https://arxiv.org/abs/2002.04745.

[61] Bowen Yang et al. Rope to Nope and Back Again: A New Hybrid Attention Strategy. 2025. arXiv: 2501.18795 [cs.CL]. URL: https://arxiv.org/abs/2501.18795.

[62] Songlin Yang, Jan Kautz, and Ali Hatamizadeh. “Gated Delta Networks: Improving Mamba2 with Delta Rule”. In: Proceedings of ICLR. 2025. URL: https://openreview.net/forum?id=r8H7xhYPwz.

[63] Songlin Yang et al. “Gated Linear Attention Transformers with Hardware-Efficient Training”. In: Proceedings of ICML. PMLR, 2024.

[64] Yongyi Yang and Jianyang Gao. mHC-lite: You Don’t Need 20 Sinkhorn-Knopp Iterations. 2026. arXiv: 2601.05732 [cs.LG]. URL: https://arxiv.org/abs/2601.05732.

[65] Rowan Zellers et al. “HellaSwag: Can a Machine Really Finish Your Sentence?” In: Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics. 2019.

[66] Biao Zhang and Rico Sennrich. “Root mean square layer normalization”. In: Advances in NeurIPS 32 (2019).

[67] Yifan Zhang et al. Deep Delta Learning. 2026. arXiv: 2601.00417 [cs.LG]. URL: https://arxiv.org/abs/2601.00417.

[68] Yilang Zhang et al. ANCRe: Adaptive Neural Connection Reassignment for Efficient Depth Scaling. 2026. arXiv: 2602.09009 [cs.LG]. URL: https://arxiv.org/abs/2602.09009.

[69] Yu Zhang et al. Kimi Linear: An Expressive, Efficient Attention Architecture. 2025. arXiv: 2510.26692 [cs.CL].

[70] Shu Zhong et al. Understanding Transformer from the Perspective of Associative Memory. 2025. arXiv: 2505.19488 [cs.LG]. URL: https://arxiv.org/abs/2505.19488.

[71] Zhanchao Zhou et al. “Value Residual Learning”. In: Proceedings of ACL. Ed. by Wanxiang Che et al. Vienna, Austria, 2025, pp. 28341–28356. URL: https://aclanthology.org/2025.acl-long.1375/.

[72] Defa Zhu et al. Hyper-Connections. 2025. arXiv: 2409.19606 [cs.LG]. URL: https://arxiv.org/abs/2409.19606.

[73] Zhijian Zhuo et al. HybridNorm: Towards Stable and Efficient Transformer Training via Hybrid Normalization. 2025. arXiv: 2503.04598 [cs.CL]. URL: https://arxiv.org/abs/2503.04598.

## 附录 A 贡献

原文说明：作者按其贡献的重要性排序，担任项目领导角色的作者排在最后；带星号者为同等贡献（Equal contribution）。

（译注：原附录在此逐一列出全部作者姓名。按本谱系统一的团队署名口径，本译文条目只署 Kimi Team，不逐一译录个人作者名单；附录的排序与同等贡献说明已在上文译出，名单本身的省略在此如实注明，并非原文缺失。）

## 附录 B 完整注意力残差的优化推理读写

完整 AttnRes 的朴素实现会在每层扫描全部先前层输出，内存流量随深度线性增长。但如第 4.2 节所说，伪查询 $\bm{w}_l$ 是学到的参数，与输入和隐藏状态都无关。因此可以按两阶段调度把跨块访问在各层之间批处理起来，把总读写压到朴素上界以下许多。

注意，下文引入的块划分纯粹是一种推理调度手段。与 Block AttnRes 不同，它不改变模型架构，也不把逐层来源换成块摘要；它只是把摊薄论证写具体。

**设定。** 设模型有 $L$ 层、隐藏维度 $d$，划分为 $N$ 个连续块，每块大小 $S = L/N$。推理逐块进行：阶段 1 对一个块内全部 $S$ 层、针对全部先前块联合计算块间注意力；阶段 2 顺序走完块内依赖。

### 阶段 1：批量的块间注意力

考虑第 $n$ 块及其 $S$ 层。查询 $\{\bm{w}_l\}_{l \in \mathcal{B}_n}$ 在执行开始前全部已知，因此 $(n{-}1)S$ 个先前键—值对只需从 HBM（高带宽内存）读一次，就能被全部 $S$ 个查询复用。第 $n$ 块的读取代价于是是：

$$\mathrm{Read}_{\text{inter}}^{(n)} = 2(n-1)Sd, \tag{11}$$

（公式说明：系数 2 同时计入键与值。）

对全部 $N$ 块求和，并用 $SN=L$：

$$\mathrm{Read}_{\text{inter}} = \sum_{n=1}^{N} 2(n-1)Sd = 2Sd \cdot \frac{N(N-1)}{2} = dL(N-1). \tag{12}$$

阶段 1 还为每层写一个 $d$ 维输出，每块 $\mathrm{Write}_{\text{inter}}^{(n)} = Sd$，总计：

$$\mathrm{Write}_{\text{inter}} = Ld \tag{13}$$

### 阶段 2：顺序的块内注意力

阶段 1 覆盖当前块之前的全部来源。块内每层却依赖它之前的层，这些必须按顺序处理。第 $t$ 层（$1 \le t \le S$）读 $t{-}1$ 个块内键—值对，代价为 $2(t{-}1)d$。对一个块求和：

$$\mathrm{Read}_{\text{intra}}^{(n)} = \sum_{t=1}^{S} 2(t-1)d = S(S-1)d. \tag{14}$$

阶段 2 也为每层写一个输出，因此 $\mathrm{Write}_{\text{intra}}^{(n)} = Sd$。

### 每层总摊薄读写

把两个阶段对全部 $N$ 块求和：

$$\mathrm{Read}_{\text{total}} = dL(N-1) + N \cdot S(S-1)d, \qquad \mathrm{Write}_{\text{total}} = 2Ld. \tag{15}$$

除以 $L$ 并用 $SN=L$：

$$\text{每层读取} = (N-1)d + (S-1)d = (S+N-2)d, \qquad \text{每层写入} = 2d, \tag{16}$$

$$\boxed{\;\text{每层总读写} = (S+N)\,d.\;} \tag{17}$$

（公式说明：附录式号照原文为 (11)–(17)。）

把块间读取批处理起来，于是每层读写从 $\mathcal{O}(L)$ 降到 $\mathcal{O}(S{+}N)$。这一调度的两阶段划分与 Block AttnRes 相同：块间注意力占流量大头，顺序计算则留在每个块内局部进行。
