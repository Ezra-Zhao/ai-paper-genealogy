# 用深度神经网络与树搜索精通围棋（全文精译）

## 书目信息

- **原题**：Mastering the Game of Go with Deep Neural Networks and Tree Search
- **作者**：David Silver、Aja Huang、Chris J. Maddison、Arthur Guez、Laurent Sifre、George van den Driessche、Julian Schrittwieser、Ioannis Antonoglou、Veda Panneershelvam、Marc Lanctot、Sander Dieleman、Dominik Grewe、John Nham、Nal Kalchbrenner、Ilya Sutskever、Timothy Lillicrap、Madeleine Leach、Koray Kavukcuoglu、Thore Graepel、Demis Hassabis（其中 David Silver 与 Aja Huang 并列第一作者，原文注 "*These authors contributed equally to this work"）
- **单位**：1. Google DeepMind，5 New Street Square, London EC4A 3TW, UK；2. Google，1600 Amphitheatre Parkway, Mountain View, California 94043, USA
- **发表**：《自然》（Nature）第 529 卷，484–489 页，2016 年 1 月 27 日在线发表（刊期 1 月 28 日），Research Article 栏目
- **DOI**：10.1038/nature16961
- **收稿与接受**：2015 年 11 月 11 日收到；2016 年 1 月 5 日接受
- **原文链接（核实过的真实来源链）**：
  - Nature 官方页面：https://www.nature.com/articles/nature16961 （官方页面含摘要、图 1–图 6 目录与参考文献 1–62 全文条目；正文全文为订阅内容预览）
  - 公开全文镜像（本译文实际取全文所据版本）：https://gwern.net/doc/reinforcement-learning/model/alphago/2016-silver.pdf （出版方排版 PDF，共 20 页：正文 484–489 页 + Methods + 扩展数据表 1–11；PDF 元数据题名、作者名单与 DOI 均与 Nature 官方页面一致，已交叉核对）
  - DOI 解析：https://doi.org/10.1038/nature16961
- **对应导读**：见《AI 顶尖论文谱系》第二卷《深度学习革命》第 8 篇（★）
- **译本说明**：本译文据上述公开镜像的出版方排版 PDF 逐节译出。正文、图 1–图 6 图注、Methods（方法）全节、致谢、作者贡献、作者信息全部译出；扩展数据表（Extended Data）1–11 的表题、表注全部译出，表 1、2、3、5、6、7、8 的数据逐格转录，表 9 为胜率交叉大表，胜率与置信区间逐格转录（置信区间以方括号附后），表 10、11 同为胜率交叉大表，只转录胜率点估计，置信区间见原 PDF（各表处已注明）；表 4 在原 PDF 中以棋盘图案图形排版、无可转录的文字行，本文件译其表题与表注并以译注说明其构成，不静默删节。公式以 LaTeX 呈现并附中文说明；正文与 Methods 的参考文献条目（1–62）保留英文原样，集中列于文末。原文图 1–图 6 与扩展数据图版的图像本身未嵌入本文件，图注已全部译出。译者注写作"（译注：……）"，公式的译者说明写作"（公式说明：……）"，均不与原文混排。本文件为 Ezra 个人学习收藏，仅本地存档。

---

## 用深度神经网络与树搜索精通围棋

**David Silver¹\*、Aja Huang¹\*、Chris J. Maddison¹、Arthur Guez¹、Laurent Sifre¹、George van den Driessche¹、Julian Schrittwieser¹、Ioannis Antonoglou¹、Veda Panneershelvam¹、Marc Lanctot¹、Sander Dieleman¹、Dominik Grewe¹、John Nham²、Nal Kalchbrenner¹、Ilya Sutskever²、Timothy Lillicrap¹、Madeleine Leach¹、Koray Kavukcuoglu¹、Thore Graepel¹ 与 Demis Hassabis¹**

¹ Google DeepMind，5 New Street Square, London EC4A 3TW, UK。² Google，1600 Amphitheatre Parkway, Mountain View, California 94043, USA。
\* 这两位作者对本工作贡献相同。

**围棋长期被视为经典棋类游戏中对人工智能最具挑战性的一种，原因在于它巨大的搜索空间，以及评估棋局局面与着法的难度。本文介绍一种计算机围棋的新方法：用"价值网络"（value networks）评估棋局局面，用"策略网络"（policy networks）选择着法。这些深度神经网络用一种新颖的组合方式训练：先从人类高手棋谱做有监督学习（supervised learning），再从自我对弈（self-play）的棋局做强化学习（reinforcement learning）。不做任何前瞻搜索（lookahead search）时，这些神经网络下围棋的水平已与最先进的蒙特卡洛树搜索（Monte Carlo tree search, MCTS）程序相当——后者要模拟数千局随机自我对弈。我们还提出一种新搜索算法，把蒙特卡洛模拟与价值网络、策略网络结合起来。用这一搜索算法，我们的程序 AlphaGo 对其他围棋程序取得了 99.8% 的胜率，并以 5 比 0 战胜了人类欧洲围棋冠军。这是计算机程序第一次在全尺寸围棋对局中击败人类职业棋手——这一成就此前被认为至少还要十年。**

所有完全信息（perfect information）博弈都有一个最优价值函数（optimal value function）$v^*(s)$：在所有棋手都完美对弈的前提下，它决定每一个棋局局面（即状态（state）$s$）的对局结果。这类博弈可以这样求解：在一棵搜索树上递归计算最优价值函数；搜索树包含约 $b^d$ 种可能的着法序列，其中 $b$ 是博弈的宽度（breadth，每个局面下合法着法的数目），$d$ 是其深度（depth，对局长度）。在大博弈中，例如国际象棋（$b \approx 35$，$d \approx 80$）¹，尤其是围棋（$b \approx 250$，$d \approx 150$）¹，穷举搜索不可行²,³，但有效搜索空间可以用两条一般原则来缩减。其一，用局面评估（position evaluation）缩减搜索深度：在状态 $s$ 处截断搜索树，把 $s$ 以下的子树替换为一个近似价值函数 $v(s) \approx v^*(s)$，由它预测从状态 $s$ 出发的结果。这一做法已在国际象棋⁴、跳棋⁵ 和奥赛罗（othello）⁶ 中达到超人类水平，但由于围棋的复杂性，人们曾认为它在围棋中无从下手⁷。其二，用从策略（policy）$p(a|s)$ 中采样动作来缩减搜索宽度：$p(a|s)$ 是局面 $s$ 下可能着法 $a$ 上的概率分布。例如，蒙特卡洛 rollout（Monte Carlo rollouts）⁸ 完全不分叉地搜索到最大深度——从策略 $p$ 为双方采样长串动作序列。对许多次 rollout 取平均可以给出有效的局面评估，在双陆棋（backgammon）⁸ 和拼字游戏 Scrabble⁹ 中达到超人类水平，在围棋中达到弱业余水平¹⁰。

蒙特卡洛树搜索（Monte Carlo tree search, MCTS）¹¹,¹² 用蒙特卡洛 rollout 估计搜索树中每个状态的价值。执行的模拟越多，搜索树长得越大，相关价值也越准确。搜索中选择动作所用的策略也随时间改进：挑选价值更高的子节点。渐近地，这一策略收敛到最优对弈，评估收敛到最优价值函数¹²。当前最强的围棋程序都以 MCTS 为基础，并辅以训练来预测人类高手着法的策略¹³。这些策略用来把搜索收窄到一束高概率动作上，并在 rollout 中采样动作。这一做法已达到很强的业余水平¹³–¹⁵。但此前的工作局限于基于输入特征线性组合的浅层策略¹³–¹⁵ 或价值函数¹⁶。

近来，深度卷积神经网络（deep convolutional neural networks）在视觉领域取得了前所未有的成绩，例如图像分类¹⁷、人脸识别¹⁸ 和玩 Atari 游戏¹⁹。它们使用许多层神经元，每层排布成相互重叠的图块（tiles），构造出一幅图像越来越抽象、越来越局部化的表征²⁰。我们对围棋采用类似的架构：把棋局局面作为一幅 $19 \times 19$ 的图像输入，用卷积层构造该局面的表征。我们用这些神经网络缩减搜索树的有效深度与宽度：用价值网络评估局面，用策略网络采样动作。

我们用一条由若干个机器学习阶段组成的流水线（pipeline）训练这些神经网络（图 1）。起点是直接从人类高手着法训练一个有监督学习（supervised learning, SL）策略网络 $p_\sigma$。这提供快速、高效的学习更新，带即时反馈和高质量梯度。与此前工作¹³,¹⁵ 类似，我们还训练一个快速策略（fast policy）$p_\pi$，能在 rollout 中迅速采样动作。接下来训练一个强化学习（reinforcement learning, RL）策略网络 $p_\rho$：它以优化自我对弈对局的最终结果来改进 SL 策略网络。这一步把策略调整向正确目标——赢棋，而不是最大化预测准确率。最后训练一个价值网络 $v_\theta$，预测 RL 策略网络与自身对弈时的胜者。我们的程序 AlphaGo 把策略网络与价值网络同 MCTS 高效地结合在一起。

### 策略网络的有监督学习

训练流水线的第一阶段，建立在用有监督学习预测围棋高手着法的此前工作¹³,²¹–²⁴ 之上。SL 策略网络 $p_\sigma(a|s)$ 在带权重 $\sigma$ 的卷积层与修正线性（rectifier）非线性之间交替；最后一个 softmax 层输出全部合法着法 $a$ 上的概率分布。策略网络的输入 $s$ 是棋盘状态的一个简单表征（见扩展数据表 2）。策略网络在随机采样的状态—动作对 $(s, a)$ 上训练，用随机梯度上升（stochastic gradient ascent）最大化人类在状态 $s$ 所选着法 $a$ 的似然：

$$\Delta \sigma \propto \frac{\partial \log p_\sigma(a|s)}{\partial \sigma}$$

（公式说明：$\sigma$ 是 SL 策略网络的权重；$p_\sigma(a|s)$ 是网络给人类实际所下着法 $a$ 的概率；更新方向是让这一概率的对数似然上升。）

我们用 KGS 围棋服务器（KGS Go Server）的 3,000 万个局面训练了一个 13 层策略网络，称之为 SL 策略网络。该网络在一个留出测试集上预测高手着法的准确率：使用全部输入特征时为 57.0%，只用原始棋盘局面与着法历史作输入时为 55.7%；相比之下，投稿之日其他研究组的最先进水平为 44.4%²⁴（完整结果见扩展数据表 3）。准确率上的小幅提升带来了棋力的大幅提升（图 2a）；更大的网络准确率更高，但在搜索中评估更慢。我们还训练了一个更快但准确率更低的 rollout 策略 $p_\pi(a|s)$：它对小型图案特征（pattern features）做线性 softmax（见扩展数据表 4），权重为 $\pi$；其准确率为 24.2%，选一个动作只需 2 μs，而策略网络需要 3 ms。

### 策略网络的强化学习

训练流水线的第二阶段，目标是用策略梯度（policy gradient）强化学习²⁵,²⁶ 改进策略网络。RL 策略网络 $p_\rho$ 与 SL 策略网络结构相同，权重 $\rho$ 初始化为相同的值，$\rho = \sigma$。我们在当前策略网络 $p_\rho$ 与一个随机选出的先前迭代版本策略网络之间进行对局。从一个对手池中这样随机抽取对手，能防止过拟合到当前策略，从而稳定训练。我们使用一个奖励函数（reward function）$r(s)$：对一切非终局时间步 $t < T$ 取零。结果 $z_t = \pm r(s_T)$ 是对局结束时的终局奖励（terminal reward），从时间步 $t$ 的当前棋手视角看：赢为 +1，输为 −1。随后在每个时间步 $t$，沿使期望结果最大化的方向²⁵ 用随机梯度上升更新权重：

$$\Delta \rho \propto \frac{\partial \log p_\rho(a_t|s_t)}{\partial \rho} z_t$$

（公式说明：$a_t \sim p_\rho(\cdot|s_t)$ 是从 RL 策略网络输出分布采样的着法；$z_t$ 是该局最终胜负（+1/−1）；赢了的着法被强化，输了的着法被削弱。）

我们评估了 RL 策略网络的实战表现：每一步着法都从其输出概率分布采样，$a_t \sim p_\rho(\cdot|s_t)$。直接对局时，RL 策略网络对 SL 策略网络的胜率超过 80%。我们还测试了最强的开源围棋程序 Pachi¹⁴——一个精巧的蒙特卡洛搜索程序，在 KGS 上为业余 2 段，每步执行 100,000 次模拟。完全不用搜索，RL 策略网络对 Pachi 的胜率为 85%。相比之下，此前仅基于有监督学习的卷积网络最先进工作，对 Pachi 的胜率为 11%²³，对稍弱一点的程序 Fuego 为 12%²⁴。

### 价值网络的强化学习

训练流水线的最后阶段聚焦局面评估：估计一个价值函数 $v^p(s)$，预测在双方都使用策略 $p$ 对弈时从局面 $s$ 出发的结果²⁸–³⁰：

$$v^p(s) = \mathbb{E}\left[z_t \mid s_t = s,\ a_{t\ldots T} \sim p\right]$$

（公式说明：$\mathbb{E}$ 为期望；$z_t$ 是从局面 $s$ 继续按策略 $p$ 下到终局时的胜负结果。）

理想情况下我们想知道完美对弈下的最优价值函数 $v^*(s)$；实践中我们改为估计我们最强策略的价值函数 $v^{p_\rho}$，所用的策略是 RL 策略网络 $p_\rho$。我们用一个权重为 $\theta$ 的价值网络 $v_\theta(s)$ 近似该价值函数，$v_\theta(s) \approx v^{p_\rho}(s) \approx v^*(s)$。这个神经网络与策略网络架构相似，但输出的是单个预测值而非概率分布。我们在状态—结果对 $(s, z)$ 上做回归来训练价值网络的权重，用随机梯度下降（stochastic gradient descent）最小化预测值 $v_\theta(s)$ 与对应结果 $z$ 之间的均方误差（mean squared error, MSE）：

$$\Delta \theta \propto \frac{\partial v_\theta(s)}{\partial \theta} \left(z - v_\theta(s)\right)$$

（公式说明：更新量正比于预测误差 $(z - v_\theta(s))$ 与价值输出对权重 $\theta$ 的梯度之积，即朝着减小均方误差的方向调整。）

用由完整棋局构成的数据直接预测对局结果这一朴素做法会导致过拟合。问题在于：连续局面之间强相关——只差一颗子，但整局共用同一个回归目标。用 KGS 数据集这样训练时，价值网络记住的是对局结果而非泛化到新局面：测试集上 MSE 最低 0.37，训练集上则为 0.19。为缓解这一问题，我们生成了一个新的自我对弈数据集，由 3,000 万个互不相同的局面组成，每个局面采自一局单独的对局。每局对局由 RL 策略网络与自身对弈直至终局。在这个数据集上训练得到训练集 MSE 0.226、测试集 MSE 0.234，说明过拟合极小。图 2b 给出价值网络的局面评估准确率，并与使用快速 rollout 策略 $p_\pi$ 的蒙特卡洛 rollout 作比较：价值函数始终更准确。单次 $v_\theta(s)$ 评估的准确率还接近使用 RL 策略网络 $p_\rho$ 的蒙特卡洛 rollout，但计算量少 15,000 倍。

### 用策略网络与价值网络进行搜索

AlphaGo 在一个 MCTS 算法中结合策略网络与价值网络（图 3），用前瞻搜索（lookahead search）选择动作。搜索树的每条边 $(s, a)$ 存储一个动作价值（action value）$Q(s, a)$、访问计数（visit count）$N(s, a)$ 和先验概率（prior probability）$P(s, a)$。搜索树由模拟（simulation，即在树内下降走完完整对局、不做回传（backup））遍历，从根状态开始。在每次模拟的每个时间步 $t$，从状态 $s_t$ 选择动作：

$$a_t = \operatorname*{argmax}_{a} \left(Q(s_t, a) + u(s_t, a)\right)$$

（公式说明：$\operatorname*{argmax}_a$ 表示取使括号内数值最大的那个动作 $a$。）

即最大化动作价值加一个奖励项：

$$u(s, a) \propto \frac{P(s, a)}{1 + N(s, a)}$$

（公式说明：奖励项 $u(s, a)$ 正比于该边的先验概率 $P(s, a)$，并随访问计数 $N(s, a)$ 增大而衰减，以鼓励探索（exploration）较少访问的着法。）

当遍历到达第 $L$ 步的叶节点（leaf node）$s_L$ 时，该叶节点可以被展开（expanded）。叶局面 $s_L$ 只由 SL 策略网络 $p_\sigma$ 处理一次，其输出概率被存为每个合法动作 $a$ 的先验概率，$P(s, a) = p_\sigma(a|s)$。叶节点用两种很不同的方式评估：第一种用价值网络 $v_\theta(s_L)$；第二种用快速 rollout 策略 $p_\pi$ 把一局随机 rollout 下到终局步 $T$，得结果 $z_L$；两种评估用一个混合参数（mixing parameter）$\lambda$ 合成叶评估 $V(s_L)$：

$$V(s_L) = (1 - \lambda) v_\theta(s_L) + \lambda z_L$$

（公式说明：$\lambda = 0$ 表示只用价值网络，$\lambda = 1$ 表示只用 rollout，中间值是两者的加权混合。）

模拟结束时，更新所有被遍历边的动作价值与访问计数。每条边累积经过它的全部模拟的访问计数与评估均值：

$$N(s, a) = \sum_{i=1}^{n} 1(s, a, i)$$

$$Q(s, a) = \frac{1}{N(s, a)} \sum_{i=1}^{n} 1(s, a, i) V\left(s_L^i\right)$$

（公式说明：$n$ 是模拟总次数；$s_L^i$ 是第 $i$ 次模拟的叶节点；$1(s, a, i)$ 是指示函数，第 $i$ 次模拟遍历了边 $(s, a)$ 时取 1，否则取 0。）

搜索完成后，算法从根局面选择访问次数最多的着法。

值得注意的是：SL 策略网络 $p_\sigma$ 在 AlphaGo 中的表现优于更强的 RL 策略网络 $p_\rho$，推测是因为人类选择的是一束多样的有希望着法，而 RL 只优化单一最佳着法。不过，由更强的 RL 策略网络导出的价值函数 $v_\theta(s) \approx v^{p_\rho}(s)$，在 AlphaGo 中的表现优于由 SL 策略网络导出的价值函数 $v_\theta(s) \approx v^{p_\sigma}(s)$。

评估策略网络与价值网络所需的计算量，比传统搜索启发式高几个数量级。为把 MCTS 与深度神经网络高效结合，AlphaGo 使用异步多线程搜索：在 CPU 上执行模拟，在 GPU 上并行计算策略网络与价值网络。AlphaGo 的最终版本使用 40 个搜索线程、48 个 CPU 和 8 个 GPU。我们还实现了一个分布式版本的 AlphaGo，使用多台机器、40 个搜索线程、1,202 个 CPU 和 176 个 GPU。Methods 部分给出异步与分布式 MCTS 的完整细节。

### 评估 AlphaGo 的棋力

为评估 AlphaGo，我们在 AlphaGo 的若干变体与其他几个围棋程序之间举行了一场内部锦标赛（internal tournament），其他程序包括最强的商业程序 Crazy Stone¹³ 与 Zen，以及最强的开源程序 Pachi¹⁴ 与 Fuego¹⁵。这些程序全部基于高性能 MCTS 算法。我们还加入了开源程序 GnuGo——一个使用 MCTS 之前最先进搜索方法的围棋程序。所有程序每步有 5 秒计算时间。

锦标赛结果（见图 4a）显示：单机版 AlphaGo 比以往任何围棋程序强出许多个段位，在与其他围棋程序的对局中赢下 495 局中的 494 局（99.8%）。为给 AlphaGo 更大的挑战，我们还进行了让四子（four handicap stones，即对手在每局开始时获得免费着法）的对局；AlphaGo 对 Crazy Stone、Zen 和 Pachi 的让子局胜率分别为 77%、86% 和 99%。分布式版 AlphaGo 明显更强：对单机版 AlphaGo 胜率 77%，对其他程序胜率 100%。

我们还评估了 AlphaGo 的一些变体：只用价值网络评估局面（$\lambda = 0$），或只用 rollout（$\lambda = 1$）（见图 4b）。即使不用 rollout，AlphaGo 也超过了所有其他围棋程序的表现，说明价值网络为围棋中的蒙特卡洛评估提供了一个可行的替代方案。不过混合评估（$\lambda = 0.5$）表现最好，对其他变体的胜率 ≥95%。这说明两种局面评估机制互补：价值网络近似的是由强但慢得不切实际的 $p_\rho$ 所下对局的结果，而 rollout 能精确地数出（score）并评估由较弱但更快的 rollout 策略 $p_\pi$ 所下对局的结果。图 5 可视化了 AlphaGo 对一个真实对局局面的评估。

最后，我们评估了分布式版 AlphaGo 对樊麾（Fan Hui）的对局。樊麾是职业二段（professional 2 dan），2013、2014、2015 年欧洲围棋锦标赛冠军。2015 年 10 月 5 日至 9 日，AlphaGo 与樊麾进行了一场正式的五番棋比赛。AlphaGo 以 5 比 0 赢得比赛（图 6 与扩展数据表 1）。这是计算机围棋程序第一次在不让子的完整围棋对局中击败人类职业棋手——这一成就此前被认为至少还要十年³,⁷,³¹。

### 讨论

在这项工作中，我们开发出一个基于深度神经网络与树搜索组合的围棋程序，其水平达到最强人类棋手之列，从而完成了人工智能的一大"宏大挑战"（grand challenges）³¹–³³。我们第一次为围棋开发出有效的着法选择与局面评估函数，它们基于用有监督学习与强化学习的新颖组合训练出的深度神经网络。我们提出了一种新搜索算法，成功地把神经网络评估与蒙特卡洛 rollout 结合起来。我们的程序 AlphaGo 把这些组件大规模地整合进一个高性能树搜索引擎。

在与樊麾的比赛中，AlphaGo 评估的局面数比深蓝（Deep Blue）在其对卡斯帕罗夫（Kasparov）的国际象棋比赛中评估的少几千倍⁴；补偿办法是：用策略网络更聪明地选择要评估的局面，用价值网络更精确地评估它们——这种方式也许更接近人类下棋的方式。此外，深蓝依赖手工制作的评估函数，而 AlphaGo 的神经网络完全通过通用的有监督学习与强化学习方法、直接从对弈中训练得到。

围棋在许多方面集中体现了人工智能所面对的困难³³,³⁴：一项有挑战性的决策任务、一个无从下手的搜索空间，以及一个复杂到似乎无法用策略或价值函数直接近似的最优解。计算机围棋此前的重大突破——MCTS 的引入——曾带动许多其他领域的相应进展，例如通用博弈对弈（general game-playing）、经典规划（classical planning）、部分可观察规划（partially observed planning）、调度（scheduling）与约束满足（constraint satisfaction）³⁵,³⁶。把树搜索与策略网络、价值网络结合起来之后，AlphaGo 终于在围棋中达到职业水平，这让人看到希望：在其他看似无从下手的人工智能领域，人类水平的性能如今也可能实现。

**在线内容**（Online Content）：Methods 以及任何额外的扩展数据显示项（Extended Data display items）与源数据（Source Data），均可在本文的在线版本中获得；这些部分独有的参考文献只出现在在线论文中。

（2015 年 11 月 11 日收到；2016 年 1 月 5 日接受。）

补充信息（Supplementary Information）可在本文的在线版本中获得。

**致谢**（Acknowledgements）：我们感谢樊麾同意与 AlphaGo 对弈；感谢 T. Manning 担任比赛裁判；感谢 R. Munos 与 T. Schaul 有益的讨论与建议；感谢 A. Cain 与 M. Cant 在视觉呈现上的工作；感谢 P. Dayan、G. Wayne、D. Kumaran、D. Purves、H. van Hasselt、A. Barreto 与 G. Ostrovski 审阅本文；并感谢 DeepMind 团队其余成员的支持、想法与鼓励。

**作者贡献**（Author Contributions）：A.H.、G.v.d.D.、J.S.、I.A.、M.La.、A.G.、T.G. 与 D.S. 设计并实现了 AlphaGo 的搜索。C.J.M.、A.G.、L.S.、A.H.、I.A.、V.P.、S.D.、D.G.、N.K.、I.S.、K.K. 与 D.S. 设计并训练了 AlphaGo 的神经网络。J.S.、J.N.、A.H. 与 D.S. 设计并实现了 AlphaGo 的评估框架。D.S.、M.Le.、T.L.、T.G.、K.K. 与 D.H. 管理项目并提供建议。D.S.、T.G.、A.G. 与 D.H. 撰写了本文。（译注：缩写按原文，即作者姓名首字母。）

**作者信息**（Author Information）：重印与许可信息见 www.nature.com/reprints。作者声明无竞争性财务利益。欢迎读者对本文的在线版本发表评论。通信与材料请求请寄 D.S.（davidsilver@google.com）或 D.H.（demishassabis@google.com）。

---

## 图注

**图 1｜神经网络训练流水线与架构。** a，一个快速 rollout 策略 $p_\pi$ 与有监督学习（SL）策略网络 $p_\sigma$，在一个局面数据集上被训练来预测人类高手着法。一个强化学习（RL）策略网络 $p_\rho$ 初始化为 SL 策略网络，然后用策略梯度学习改进，以（在与策略网络先前版本的对局中）最大化结果（即赢更多棋）。用 RL 策略网络进行自我对弈生成一个新的数据集。最后，一个价值网络 $v_\theta$ 用回归训练，来预测自我对弈数据集各局面的期望结果（即当前棋手是否获胜）。b，AlphaGo 所用神经网络架构的示意表示。策略网络以棋局局面的一个表征 $s$ 为输入，穿过许多带参数 $\sigma$（SL 策略网络）或 $\rho$（RL 策略网络）的卷积层，输出合法着法 $a$ 上的一个概率分布 $p_\sigma(a|s)$ 或 $p_\rho(a|s)$，以棋盘上的概率图（probability map）表示。价值网络同样使用许多带参数 $\theta$ 的卷积层，但输出一个标量值 $v_\theta(s')$，预测局面 $s'$ 的期望结果。

**图 2｜策略网络与价值网络的强度与准确率。** a，图示策略网络的棋力与其训练准确率的关系。每层有 128、192、256、384 个卷积滤波器（convolutional filters）的策略网络在训练中被定期评估；图中给出 AlphaGo 使用该策略网络时对比赛版 AlphaGo 的胜率。b，价值网络与使用不同策略的 rollout 之间评估准确率的比较。局面与结果采自人类高手棋局。每个局面或由价值网络 $v_\theta$ 的单次前向传递评估，或由 100 次 rollout 的平均结果评估——rollout 分别用均匀随机 rollout、快速 rollout 策略 $p_\pi$、SL 策略网络 $p_\sigma$ 或 RL 策略网络 $p_\rho$ 下出。图中绘制预测值与实际对局结果之间的均方误差随对局阶段（该局面已下着法数）的变化。（译注：图 2a 中准确率从约 50% 升到 59% 一段，胜率大体从 0 升到约 70%；图 2b 中价值网络的均方误差全程低于各 rollout 曲线，RL 策略网络 rollout 次之。）

**图 3｜AlphaGo 中的蒙特卡洛树搜索。** a，每次模拟遍历搜索树，选择动作价值 $Q$ 最大、再加一个取决于该边所存先验概率 $P$ 的奖励项 $u(P)$ 的那条边。b，叶节点可以被展开；新节点由策略网络 $p_\sigma$ 处理一次，其输出概率被存为每个动作的先验概率 $P$。c，每次模拟结束时，叶节点用两种方式评估：用价值网络 $v_\theta$；以及用快速 rollout 策略 $p_\pi$ 把一局 rollout 下到对局结束，再用函数 $r$ 计算胜者。d，动作价值 $Q$ 被更新，以跟踪该动作以下子树中全部评估 $r(\cdot)$ 与 $v_\theta(\cdot)$ 的均值。

**图 4｜AlphaGo 的锦标赛评估。** a，不同围棋程序之间一场比赛的结果（见扩展数据表 6–11）。每个程序每步约用 5 秒计算时间。为给 AlphaGo 更大的挑战，一些程序（上方浅色条）对所有对手都被让四子（即每局开始时的免费着法）。程序在 Elo 量表³⁷ 上评估：230 分的差距对应 79% 的获胜概率，大致相当于 KGS 上一个业余段位的优势³⁸；图中还给出与人类等级的大致对应，水平线表示该程序在线上达到的 KGS 等级。与欧洲人类冠军樊麾的对局也被包括在内；这些对局使用更长的用时控制。图中给出 95% 置信区间。b，单机版 AlphaGo 在不同组件组合下的表现。只使用策略网络的版本不做任何搜索。c，AlphaGo 中 MCTS 随搜索线程与 GPU 数量的可扩展性（scalability）研究，分别使用异步搜索（浅蓝）或分布式搜索（深蓝），每步 2 秒。（译注：图 4b 中组件从左到右大体为 rollout、策略网络、价值网络的组合增减对 Elo 的影响；图 4c 中 Elo 随线程数从 1 增至 40、GPU 从 1 增至 280 而上升，分布式版最高。）

**图 5｜AlphaGo（执黑、待下）在一场与樊麾的非正式对局中如何选择着法。** 以下每项统计，最大值所在位置都以橙色圆圈标出。a，用价值网络 $v_\theta(s')$ 对根局面 $s$ 的全部后继局面 $s'$ 的评估；图中给出最高几项评估的估计获胜百分比。b，树中每条边 $(s, a)$ 的动作价值 $Q(s, a)$，从根局面 $s$ 出发；只对价值网络评估取平均（$\lambda = 0$）。c，动作价值 $Q(s, a)$，只对 rollout 评估取平均（$\lambda = 1$）。d，直接来自 SL 策略网络 $p_\sigma(a|s)$ 的着法概率；以百分比报告（若高于 0.1%）。e，模拟中从根节点选择各动作的百分比频率。f，AlphaGo 搜索树的主变化（principal variation，访问计数最大的路径）。着法以编号序列呈现。AlphaGo 选择了红圈标出的着法；樊麾以白色方块标出的着法回应；在他的赛后解说中，他更偏好 AlphaGo 预测的（标号 1 的）着法。

**图 6｜AlphaGo 与欧洲冠军樊麾比赛的棋局。** 着法按落子顺序以编号序列显示。落在同一交叉点上的重复着法，在棋盘下方成对显示。每对中的第一个着法编号表示该重复着法是何时下的，交叉点由第二个着法编号确定（见补充信息）。五局分别为：第 1 局，樊麾（黑）对 AlphaGo（白），AlphaGo 以 2.5 子胜；第 2 局，AlphaGo（黑）对樊麾（白），AlphaGo 中盘胜（wins by resignation，指对方认输；下同）；第 3 局，樊麾（黑）对 AlphaGo（白），AlphaGo 中盘胜；第 4 局，AlphaGo（黑）对樊麾（白），AlphaGo 中盘胜；第 5 局，樊麾（黑）对 AlphaGo（白），AlphaGo 中盘胜。（译注：各局棋谱的逐手编号图在本文件中未嵌入，棋谱数据见原 PDF 第 488 页图 6 与补充信息。）

---

## 方法（Methods）

### 问题设定

许多完全信息博弈，如国际象棋、跳棋、奥赛罗、双陆棋与围棋，都可以定义为交替马尔可夫博弈（alternating Markov games）³⁹。在这类博弈中，有一个状态空间（state space）$\mathcal{S}$（状态包含当前轮到哪位棋手落子的指示）；一个动作空间（action space）$\mathcal{A}(s)$，定义任一给定状态 $s \in \mathcal{S}$ 下的合法动作；一个状态转移函数（state transition function）$f(s, a, \xi)$，定义在状态 $s$ 选择动作 $a$ 并给定随机输入 $\xi$（例如骰子）之后的后继状态；最后有一个奖励函数 $r_i(s)$，描述棋手 $i$ 在状态 $s$ 收到的奖励。我们只关注双人零和博弈（two-player zero-sum games）：$r_1(s) = -r_2(s) = r(s)$，状态转移是确定性的，$f(s, a, \xi) = f(s, a)$，且除终局时间步 $T$ 外奖励为零。对局结果 $z_t = \pm r(s_T)$ 是对局结束时的终局奖励，从时间步 $t$ 的当前棋手视角看。

策略 $p(a|s)$ 是合法动作 $a \in \mathcal{A}(s)$ 上的概率分布。价值函数是：若双方全部动作都按策略 $p$ 选择时的期望结果，即 $v^p(s) = \mathbb{E}[z_t \mid s_t = s, a_{t\ldots T} \sim p]$。零和博弈有一个唯一的最优价值函数 $v^*(s)$，它决定在双方完美对弈下从状态 $s$ 出发的结果：

$$v^*(s) = \begin{cases} z_T & \text{若 } s = s_T, \\ \max_a \left(-v^*(f(s, a))\right) & \text{其他情形} \end{cases}$$

（公式说明：终局状态的价值就是终局结果 $z_T$；其余状态的价值取对当前棋手最有利的动作，并对后继状态的价值取负——因为后继状态是从对手视角评估的，这正是零和博弈的负极大（negamax）形式。）

### 此前工作

最优价值函数可以用极小极大（minimax，或等价的 negamax）搜索⁴⁰ 递归计算。大多数博弈都太大，无法做穷举极小极大树搜索；取而代之的做法是截断博弈：用近似价值函数 $v(s) \approx v^*(s)$ 代替终局奖励。带 alpha–beta 剪枝（alpha–beta pruning）⁴⁰ 的深度优先极小极大搜索在国际象棋⁴、跳棋⁵ 和奥赛罗⁶ 中达到超人类水平，但在围棋中并不有效⁷。

强化学习可以直接从自我对弈的对局中学习近似最优价值函数³⁹。此前大多数工作聚焦于特征 $\phi(s)$ 与权重 $\theta$ 的线性组合 $v_\theta(s) = \phi(s) \cdot \theta$。权重用时序差分学习（temporal-difference learning）⁴¹ 训练，应用于国际象棋⁴²,⁴³、跳棋⁴⁴,⁴⁵ 和围棋³⁰；或用线性回归训练，应用于奥赛罗⁶ 与 Scrabble⁹。时序差分学习也被用来训练一个神经网络近似最优价值函数，在双陆棋中达到超人类水平⁴⁶；在小棋盘围棋中用卷积网络达到弱级位（kyu，级）水平²⁸,²⁹,⁴⁷。

极小极大搜索的一个替代方案是蒙特卡洛树搜索（MCTS）¹¹,¹²：它用一个双重近似估计内部节点的最优价值，$V^n(s) \approx v^{P^n}(s) \approx v^*(s)$。第一个近似 $V^n(s) \approx v^{P^n}(s)$ 用 $n$ 次蒙特卡洛模拟估计一个模拟策略（simulation policy）$P^n$ 的价值函数。第二个近似 $v^{P^n}(s) \approx v^*(s)$ 用模拟策略 $P^n$ 代替极小极大最优动作。模拟策略按一个搜索控制函数 $\operatorname*{argmax}_a (Q^n(s, a) + u(s, a))$（如 UCT¹²）选择动作：该函数选择动作价值更高的子节点，$Q^n(s, a) = -V^n(f(s, a))$，再加一个鼓励探索的奖励项 $u(s, a)$；或者，在状态 $s$ 处没有搜索树时，从一个快速 rollout 策略 $p_\pi(a|s)$ 采样动作。随着执行的模拟增多、搜索树长得更深，模拟策略得到越来越准确的统计量支撑。在极限下，两个近似都变为精确，MCTS（例如带 UCT）收敛¹² 到最优价值函数：

$$\lim_{n \to \infty} V^n(s) = \lim_{n \to \infty} v^{P^n}(s) = v^*(s)$$

（公式说明：$n$ 是模拟次数；当模拟次数趋于无穷时，蒙特卡洛估计与其模拟策略的价值都收敛到真正的最优价值 $v^*(s)$。）

当前最强的围棋程序都基于 MCTS¹³–¹⁵,³⁶。

MCTS 此前已与策略结合使用：用策略把搜索树收窄到高概率着法¹³，或使奖励项偏向高概率着法⁴⁸。MCTS 也与价值函数结合使用过：用价值函数初始化新展开节点的动作价值¹⁶，或把蒙特卡洛评估与极小极大评估混合⁴⁹。相比之下，AlphaGo 对价值函数的使用基于截断蒙特卡洛搜索算法⁸,⁹：这类算法在对局结束前终止 rollout，用价值函数代替终局奖励。AlphaGo 的局面评估把完整 rollout 与截断 rollout 混合起来，在某些方面类似著名的时序差分学习算法 TD($\lambda$)。AlphaGo 与此前工作的另一点不同是：它对策略与价值函数使用更慢但更强的表征；评估深度神经网络比线性表征慢几个数量级，因此必须异步进行。

MCTS 的性能在很大程度上由 rollout 策略的质量决定。此前工作聚焦于手工图案⁵⁰，或用有监督学习¹³、强化学习¹⁶、模拟均衡（simulation balancing）⁵¹,⁵² 或在线适应³⁰,⁵³ 学习 rollout 策略；但众所周知，基于 rollout 的局面评估经常不准确⁵⁴。AlphaGo 使用相对简单的 rollout，转而用价值网络更直接地处理局面评估这一难题。

### 搜索算法

为把大型神经网络高效地整合进 AlphaGo，我们实现了一个异步策略与价值 MCTS 算法（asynchronous policy and value MCTS, APV-MCTS）。搜索树中每个节点 $s$ 包含针对全部合法动作 $a \in \mathcal{A}(s)$ 的边 $(s, a)$。每条边存储一组统计量：

$$\{P(s, a),\ N_v(s, a),\ N_r(s, a),\ W_v(s, a),\ W_r(s, a),\ Q(s, a)\}$$

（公式说明：$P(s, a)$ 是先验概率；$W_v(s, a)$ 与 $W_r(s, a)$ 分别是动作总价值的蒙特卡洛估计，累积自 $N_v(s, a)$ 次叶评估与 $N_r(s, a)$ 次 rollout 奖励；$Q(s, a)$ 是该边的合并平均动作价值。）

多次模拟在各自的搜索线程上并行执行。APV-MCTS 算法按图 3 所示的四个阶段进行。

**选择（Selection，图 3a）。** 每次模拟的第一个树内阶段从搜索树根开始，到模拟到达时间步 $L$ 的叶节点时结束。在这些时间步中的每一步，$t < L$，按搜索树中的统计量选择动作，$a_t = \operatorname*{argmax}_a (Q(s_t, a) + u(s_t, a))$，使用 PUCT 算法⁴⁸ 的一个变体：

$$u(s, a) = c_{\mathrm{puct}} P(s, a) \frac{\sqrt{\sum_b N_r(s, b)}}{1 + N_r(s, a)}$$

（公式说明：$c_{\mathrm{puct}}$ 是决定探索程度的常数；$\sum_b N_r(s, b)$ 是状态 $s$ 下全部动作的 rollout 访问总数。这一搜索控制策略起初偏好先验概率高、访问计数低的动作，渐近地偏好动作价值高的动作。）

**评估（Evaluation，图 3c）。** 叶局面 $s_L$ 被加入一个队列，由价值网络评估 $v_\theta(s_L)$，除非它此前已被评估过。每次模拟的第二个 rollout 阶段从叶节点 $s_L$ 开始，一直持续到对局结束。在这些时间步中的每一步，$t \ge L$，双方的动作都按 rollout 策略选择，$a_t \sim p_\pi(\cdot|s_t)$。对局到达终局状态时，由最终比分计算结果 $z_t = \pm r(s_T)$。

**回传（Backup，图 3d）。** 在模拟的每个树内步 $t \le L$，rollout 统计量按"好像输了 $n_{\mathrm{vl}}$ 局"那样更新：$N_r(s_t, a_t) \leftarrow N_r(s_t, a_t) + n_{\mathrm{vl}}$；$W_r(s_t, a_t) \leftarrow W_r(s_t, a_t) - n_{\mathrm{vl}}$；这一虚拟损失（virtual loss）⁵⁵ 使其他线程不致同时探索完全相同的变化。模拟结束时，在一次反向传递中遍历每一步 $t \le L$ 更新 rollout 统计量，用实际结果替换虚拟损失：$N_r(s_t, a_t) \leftarrow N_r(s_t, a_t) - n_{\mathrm{vl}} + 1$；$W_r(s_t, a_t) \leftarrow W_r(s_t, a_t) + n_{\mathrm{vl}} + z_t$。异步地，当叶局面 $s_L$ 的评估完成时，会另行发起一次反向传递。价值网络的输出 $v_\theta(s_L)$ 被用来在第二次反向传递中、遍历每一步 $t \le L$ 更新价值统计量：$N_v(s_t, a_t) \leftarrow N_v(s_t, a_t) + 1$，$W_v(s_t, a_t) \leftarrow W_v(s_t, a_t) + v_\theta(s_L)$。每个状态—动作的总体评估是两个蒙特卡洛估计的加权平均：

$$Q(s, a) = (1 - \lambda) \frac{W_v(s, a)}{N_v(s, a)} + \lambda \frac{W_r(s, a)}{N_r(s, a)}$$

（公式说明：该式以权重参数 $\lambda$ 把价值网络评估与 rollout 评估混合在一起。全部更新都以无锁（lock-free）方式⁵⁶ 执行。）

**展开（Expansion，图 3b）。** 当访问计数超过一个阈值，$N_r(s, a) > n_{\mathrm{thr}}$ 时，后继状态 $s' = f(s, a)$ 被加入搜索树。新节点被初始化为 $\{N(s', a) = N_r(s', a) = 0,\ W(s', a) = W_r(s', a) = 0,\ P(s', a) = p_\sigma(a|s')\}$，并使用一个树策略（tree policy）$p_\tau(a|s')$（与 rollout 策略类似但特征更多，见扩展数据表 4）提供占位先验概率，供动作选择使用。局面 $s'$ 同时被插入一个队列，等待策略网络的异步 GPU 评估。先验概率由 SL 策略网络 $p_\sigma^\beta(\cdot|s')$ 计算，softmax 温度（softmax temperature）设为 $\beta$；算出的先验概率以一次原子更新（atomic update）替换占位先验概率，$P(s', a) \leftarrow p_\sigma^\beta(a|s')$。阈值 $n_{\mathrm{thr}}$ 被动态调整，以使局面加入策略队列的速率与 GPU 评估策略网络的速率相匹配。局面由策略网络与价值网络评估时都使用小批量（mini-batch）大小 1，以最小化端到端评估时间。

我们还实现了一个分布式 APV-MCTS 算法。该架构由一台执行主搜索的主机（master）、许多执行异步 rollout 的远程工作 CPU、以及许多执行异步策略与价值网络评估的远程工作 GPU 组成。整棵搜索树存储在主机上，主机只执行每次模拟的树内阶段。叶局面被传送给工作 CPU 执行模拟的 rollout 阶段，并传送给工作 GPU 计算网络特征、评估策略网络与价值网络。策略网络的先验概率被送回主机，在新展开的节点处替换占位先验概率。rollout 的奖励与价值网络的输出各自被送回主机，并沿原搜索路径回传。

搜索结束时，AlphaGo 选择访问计数最大的动作；这比最大化动作价值对异常值（outliers）不那么敏感¹⁵。搜索树在后续时间步被复用：与所下动作对应的子节点成为新的根节点；该子节点以下的子树连同其全部统计量被保留，树的其余部分被丢弃。比赛版 AlphaGo 在对手思考时继续搜索。如果访问计数最大的动作与动作价值最大的动作不一致，它会延长搜索。用时控制（time controls）其余方面被塑形为把大部分时间用在中盘⁵⁷。当 AlphaGo 的总体评估跌破估计 10% 的获胜概率时它认输，即 $\max_a Q(s, a) < -0.8$。

AlphaGo 不使用大多数蒙特卡洛围棋程序所用的"全着法当首着"（all-moves-as-first）¹⁰ 或快速动作价值估计（rapid action value estimation）⁵⁸ 启发式；在使用策略网络作先验知识时，这些有偏的启发式似乎不带来额外收益。此外 AlphaGo 不使用渐进加宽（progressive widening）¹³、动态贴目（dynamic komi）⁵⁹ 或开局库（opening book）⁶⁰。AlphaGo 在对樊麾比赛时所用的参数列于扩展数据表 5。

### Rollout 策略

Rollout 策略 $p_\pi(a|s)$ 是一个线性 softmax 策略，基于快速的、增量计算的、基于局部图案的特征。这些特征既包括导致状态 $s$ 的上一着法周围的"响应"（response）图案，也包括状态 $s$ 下候选着法 $a$ 周围的"非响应"（non-response）图案。每个非响应图案是一个二值特征，匹配一个以 $a$ 为中心的特定 $3 \times 3$ 图案，图案由每个相邻交叉点的颜色（黑、白、空）与气数（liberty count，1、2、≥3）定义。每个响应图案是一个二值特征，匹配以上一着法为中心的 12 点菱形图案²¹ 中各点的颜色与气数。此外，少量手工局部特征编码常识性的围棋规则（见扩展数据表 4）。与策略网络类似，rollout 策略的权重 $\pi$ 用随机梯度下降、从 Tygem 服务器上人类对局的 800 万个局面训练，以最大化对数似然。Rollout 在空棋盘上每个 CPU 线程每秒执行约 1,000 次模拟。

我们的 rollout 策略 $p_\pi(a|s)$ 所含的手工知识少于最先进的围棋程序¹³。取而代之，我们利用 MCTS 内部更高质量的动作选择——它同时受搜索树与策略网络的信息支持。我们还引入一项新技术：缓存搜索树的全部着法，随后在 rollout 中下出类似着法；这是对"上次好回复"（last good reply）启发式⁵³ 的推广。在树遍历的每一步，概率最大的动作连同其图案上下文（上一着法与当前着法周围的颜色、气数与子数构成的 $3 \times 3$ 图案上下文）一起被插入一张哈希表。在 rollout 的每一步，图案上下文与哈希表匹配；若找到匹配，就以高概率下出所存的着法。

### 对称性

此前工作利用围棋的对称性，在卷积层中使用旋转与反射不变的滤波器²⁴,²⁸,²⁹。这对小型神经网络可能有效，但在更大的网络中反而损害性能，因为它妨碍中间滤波器识别特定的不对称图案²³。我们改为在运行时利用对称性：用八种反射与旋转构成的二面体群（dihedral group）动态变换每个局面 $s$，得到 $d_1(s), \ldots, d_8(s)$。在一个显式对称集成（explicit symmetry ensemble）中，一个包含全部 8 个局面的小批量被送入策略网络或价值网络并行计算。对价值网络，输出值直接取平均：

$$v_\theta(s) = \frac{1}{8} \sum_{j=1}^{8} v_\theta(d_j(s))$$

（公式说明：$d_j(s)$ 是局面 $s$ 的第 $j$ 种旋转/反射变换；8 个朝向的价值估计取平均，得到对朝向不敏感的评估。）

对策略网络，输出概率的各平面被旋转/反射回原始朝向，再平均到一起给出集成预测：

$$p_\sigma(\cdot|s) = \frac{1}{8} \sum_{j=1}^{8} d_j^{-1}\left(p_\sigma(\cdot|d_j(s))\right)$$

（公式说明：$d_j^{-1}$ 是把概率图变换回原朝向的逆变换。这一做法用于我们的原始网络评估，见扩展数据表 3。）

与之不同，APV-MCTS 使用隐式对称集成：为每次评估随机选择单个旋转/反射 $j \in [1, 8]$。我们只计算该朝向的一次评估；在每次模拟中用 $v_\theta(d_j(s_L))$ 计算叶节点 $s_L$ 的价值，让搜索程序对这些评估取平均。类似地，我们为单个随机选定的旋转/反射计算策略网络，$d_j^{-1}(p_\sigma(\cdot|d_j(s)))$。

### 策略网络：分类

我们训练策略网络 $p_\sigma$，按 KGS 数据集中高手所下的着法对局面分类。该数据集包含 KGS 6 段到 9 段人类棋手所下 160,000 局对局的 2,940 万个局面；其中 35.4% 的对局是让子局。数据集被拆分为测试集（前 100 万个局面）与训练集（其余 2,840 万个局面）。停一手着法（pass moves）被排除在数据集之外。每个局面由一个原始棋盘描述 $s$ 与人类选择的着法 $a$ 组成。我们对数据集做了增广，纳入每个局面的全部八种反射与旋转。对称增广与输入特征都为每个局面预先计算好。每个训练步，我们从增广后的 KGS 数据集采样一个随机选定的、含 $m$ 个样本的小批量 $\{s^k, a^k\}_{k=1}^m$，施加一次异步随机梯度下降更新，以最大化动作的对数似然：

$$\Delta \sigma = \frac{\alpha}{m} \sum_{k=1}^{m} \frac{\partial \log p_\sigma(a^k|s^k)}{\partial \sigma}$$

（公式说明：$\alpha$ 是步长（step size）；梯度是对一个小批量内各样本对数似然梯度的平均。）

步长 $\alpha$ 初始化为 0.003，每 8,000 万个训练步减半，不带动量项（momentum terms），小批量大小 $m = 16$。更新在 50 个 GPU 上用 DistBelief⁶¹ 异步施加；早于 100 步的梯度被丢弃。训练共进行约 3.4 亿个训练步，耗时约 3 周。

### 策略网络：强化学习

我们用策略梯度强化学习²⁵,²⁶ 进一步训练策略网络。每次迭代由一个含 $n$ 局对局的小批量并行进行：对局在正在训练的当前策略网络 $p_\rho$ 与一个对手 $p_{\rho^-}$ 之间进行，后者使用先前某次迭代的参数 $\rho^-$，从一个对手池中随机采样，以增加训练的稳定性。权重初始化为 $\rho = \rho^- = \sigma$。每 500 次迭代，我们把当前参数 $\rho$ 加入对手池。小批量中的每局对局 $i$ 都下到第 $T^i$ 步终局，然后计分，从每位棋手的视角确定结果 $z_t^i = \pm r(s_{T^i})$。随后重放这些对局，确定策略梯度更新：

$$\Delta \rho = \frac{\alpha}{n} \sum_{i=1}^{n} \sum_{t=1}^{T^i} \frac{\partial \log p_\rho(a_t^i|s_t^i)}{\partial \rho} \left(z_t^i - v(s_t^i)\right)$$

（公式说明：这使用带基线（baseline）$v(s_t^i)$ 的 REINFORCE 算法²⁵ 来减小方差；$v(s_t^i)$ 是对该局面价值的估计，实际胜负超出基线的部分才驱动更新。）

在训练流水线的第一遍，基线设为零；在第二遍，我们用价值网络 $v_\theta(s)$ 作基线，这带来了小幅性能提升。策略网络以这种方式训练了 10,000 个每批 128 局的小批量，使用 50 个 GPU，耗时一天。

### 价值网络：回归

我们训练价值网络 $v_\theta(s) \approx v^{p_\rho}(s)$，近似 RL 策略网络 $p_\rho$ 的价值函数。为避免过拟合到一局对局内部强相关的局面，我们构造了一个由互不相关自我对弈局面组成的新数据集。该数据集由 3,000 多万个局面组成，每个局面取自一局独一无二的对局。每局对局分三个阶段生成：先随机采样一个时间步 $U \sim \mathrm{unif}\{1, 450\}$，前 $t = 1, \ldots, U-1$ 步着法从 SL 策略网络采样，$a_t \sim p_\sigma(\cdot|s_t)$；然后从可用着法中均匀随机采样一步，$a_U \sim \mathrm{unif}\{1, 361\}$（反复采样直至 $a_U$ 合法）；然后其余的着法序列直至对局终局，$t = U+1, \ldots, T$，从 RL 策略网络采样，$a_t \sim p_\rho(\cdot|s_t)$。最后对对局计分，确定结果 $z_t = \pm r(s_T)$。每局对局只向数据集加入单个训练样本 $(s_{U+1}, z_{U+1})$。这些数据提供价值函数 $v^{p_\rho}(s_{U+1}) = \mathbb{E}[z_{U+1} \mid s_{U+1}, a_{U+1, \ldots, T} \sim p_\rho]$ 的无偏样本。在生成的前两个阶段，我们从噪声更大的分布采样，以增加数据集的多样性。训练方法与 SL 策略网络训练相同，只是参数更新基于预测值与观察奖励之间的均方误差：

$$\Delta \theta = \frac{\alpha}{m} \sum_{k=1}^{m} \left(z^k - v_\theta(s^k)\right) \frac{\partial v_\theta(s^k)}{\partial \theta}$$

（公式说明：$m$ 是小批量大小；$z^k$ 是样本 $k$ 的实际对局结果；更新朝着减小预测误差的方向调整 $\theta$。）

价值网络训练了 5,000 万个每批 32 个局面的小批量，使用 50 个 GPU，耗时一周。

### 策略/价值网络的特征

每个局面 $s$ 都被预处理成一组 $19 \times 19$ 的特征平面（feature planes）。我们使用的特征直接来自棋规的原始表征，指示围棋棋盘每个交叉点的状态：子的颜色、气（与子链相邻的空点）、提子（captures）、合法性、子落下后经过的回合数，以及（仅价值网络）当前待下颜色。此外，我们使用一个简单的战术特征，计算吃子梯子（ladder）搜索⁷ 的结果。全部特征都相对于当前待下颜色计算；例如，每个交叉点的子的颜色被表征为"我方"或"对方"，而非黑或白。每个整数特征值被拆分成多个 $19 \times 19$ 的二值平面（one-hot 编码，即独热编码）。例如，用分开的二值特征平面表示一个交叉点有 1 个气、2 个气、……、≥8 个气。特征平面的完整集合列于扩展数据表 2。

### 神经网络架构

策略网络的输入是一个由 48 个特征平面组成的 $19 \times 19 \times 48$ 图像栈（image stack）。第一个隐藏层先把输入零填充（zero pads）成一幅 $23 \times 23$ 图像，然后用 $k$ 个核大小 $5 \times 5$、步幅（stride）1 的滤波器与输入图像做卷积，并施加修正线性非线性。其后的每个隐藏层（第 2 至 12 层）先把各自的前一隐藏层零填充成一幅 $21 \times 21$ 图像，然后用 $k$ 个核大小 $3 \times 3$、步幅 1 的滤波器做卷积，同样跟随修正线性非线性。最后一层用 1 个核大小 $1 \times 1$、步幅 1 的滤波器做卷积，每个位置有不同的偏置，并施加 softmax 函数。比赛版 AlphaGo 使用 $k = 192$ 个滤波器；图 2b 与扩展数据表 3 还给出 $k = 128$、256、384 个滤波器的训练结果。

价值网络的输入同样是一个 $19 \times 19 \times 48$ 图像栈，另加一个描述当前待下颜色的二值特征平面。第 2 至 11 隐藏层与策略网络相同，第 12 隐藏层是一个额外的卷积层，第 13 隐藏层用 1 个核大小 $1 \times 1$、步幅 1 的滤波器做卷积，第 14 隐藏层是一个有 256 个修正线性单元的全连接线性层。输出层是一个只有一个 tanh 单元的全连接线性层。

### 评估

我们通过举行一场内部锦标赛、测量每个程序的 Elo 等级分（Elo rating），评估计算机围棋程序的相对强度。我们用一个逻辑函数（logistic function）估计程序 $a$ 击败程序 $b$ 的概率：

$$p(a\ \text{beats}\ b) = \frac{1}{1 + \exp\left(c_{\mathrm{elo}}(e(b) - e(a))\right)}$$

（公式说明：$e(\cdot)$ 是程序的 Elo 等级分；等级分用 BayesElo 程序³⁷ 以贝叶斯逻辑回归（Bayesian logistic regression）估计，使用标准常数 $c_{\mathrm{elo}} = 1/400$。量表锚定在职业棋手樊麾的 BayesElo 等级分上（投稿之日为 2,908）⁶²。）

全部程序每步最多有 5 秒计算时间；对局用中国规则（Chinese rules）计分，贴目（komi）7.5 点（补偿白方后下的额外分数）。我们还进行了让子局：AlphaGo 执白对既有围棋程序；这些对局使用一种非标准让子系统：保留贴目，但黑方在通常的让子点上获得额外的子。用这些规则，让 $K$ 子相当于给黑方 $K - 1$ 步免费着法，而不是用标准无贴目让子规则时的 $K - 1/2$ 步。我们使用这套让子规则，是因为 AlphaGo 的价值网络是专门按 7.5 贴目训练的。

除分布式 AlphaGo 外，每个计算机围棋程序都在它自己的一台单机上执行，规格相同，使用最新可用版本与该程序支持的最佳硬件配置（见扩展数据表 6）。图 4 中计算机程序的大致等级，基于该程序在 KGS 上达到过的最高等级；不过 KGS 版本可能与公开可用版本不同。

对樊麾的比赛由一位公正裁判仲裁。五局正式对局与五局非正式对局以 7.5 贴目、无让子、中国规则进行。AlphaGo 在这两组对局中的比分分别为 5–0 与 3–2（图 6 与扩展数据表 1）。正式对局的用时控制为 1 小时基本时间加三段 30 秒读秒（byoyomi）；非正式对局的用时控制为三段 30 秒读秒。用时控制与比赛条件由樊麾在赛前选定；双方还约定，比赛总结果只由正式对局决定。为大致评估樊麾相对于计算机围棋程序的等级分，我们把全部十局对局的结果附加到内部锦标赛的结果中，忽略用时控制的差异。

---

## 扩展数据（Extended Data）

**扩展数据表 1｜AlphaGo 与樊麾比赛的细节**

| 日期 | 黑 | 白 | 类别 | 结果 |
|---|---|---|---|---|
| 5/10/15 | Fan Hui | AlphaGo | Formal（正式） | AlphaGo wins by 2.5 points（AlphaGo 以 2.5 子胜） |
| 5/10/15 | Fan Hui | AlphaGo | Informal（非正式） | Fan Hui wins by resignation（樊麾中盘胜） |
| 6/10/15 | AlphaGo | Fan Hui | Formal | AlphaGo wins by resignation（AlphaGo 中盘胜） |
| 6/10/15 | AlphaGo | Fan Hui | Informal | AlphaGo wins by resignation |
| 7/10/15 | Fan Hui | AlphaGo | Formal | AlphaGo wins by resignation |
| 7/10/15 | Fan Hui | AlphaGo | Informal | AlphaGo wins by resignation |
| 8/10/15 | AlphaGo | Fan Hui | Formal | AlphaGo wins by resignation |
| 8/10/15 | AlphaGo | Fan Hui | Informal | AlphaGo wins by resignation |
| 9/10/15 | Fan Hui | AlphaGo | Formal | AlphaGo wins by resignation |
| 9/10/15 | AlphaGo | Fan Hui | Informal | Fan Hui wins by resignation |

该比赛由五局用时较长的正式对局与五局用时较短的非正式对局组成。用时控制与比赛条件由樊麾在赛前选定。

**扩展数据表 2｜神经网络的输入特征**

| 特征 | 平面数 | 说明 |
|---|---|---|
| Stone colour（子的颜色） | 3 | 我方子 / 对方子 / 空 |
| Ones（全 1） | 1 | 一个填满 1 的常数平面 |
| Turns since（经过回合） | 8 | 一步着法落下后经过了多少回合 |
| Liberties（气） | 8 | 气的数目（相邻空点） |
| Capture size（提子数） | 8 | 将提掉多少对方子 |
| Self-atari size（自陷打吃数） | 8 | 将有多少己方子被提掉 |
| Liberties after move（落子后气数） | 8 | 下出这一步后的气数 |
| Ladder capture（梯子提子） | 1 | 在此点落子是否为一次成功的梯子提子 |
| Ladder escape（梯子逃脱） | 1 | 在此点落子是否为一次成功的梯子逃脱 |
| Sensibleness（合理性） | 1 | 该着法是否合法且不填自己的眼 |
| Zeros（全 0） | 1 | 一个填满 0 的常数平面 |
| Player color（玩家颜色） | 1 | 当前棋手是否为黑方 |

策略网络使用的特征平面（除最后一项特征外的全部）与价值网络使用的特征平面（全部特征）。

**扩展数据表 3｜策略网络的有监督学习结果**

| 滤波器 | 对称数 | 特征 | 测试准确率 % | 训练准确率 % | 原始网络胜率 % | AlphaGo 胜率 % | 前向时间（ms） |
|---|---|---|---|---|---|---|---|
| 128 | 1 | 48 | 54.6 | 57.0 | 36 | 53 | 2.8 |
| 192 | 1 | 48 | 55.4 | 58.0 | 50 | 50 | 4.8 |
| 256 | 1 | 48 | 55.9 | 59.1 | 67 | 55 | 7.1 |
| 256 | 2 | 48 | 56.5 | 59.8 | 67 | 38 | 13.9 |
| 256 | 4 | 48 | 56.9 | 60.2 | 69 | 14 | 27.6 |
| 256 | 8 | 48 | 57.0 | 60.4 | 69 | 5 | 55.3 |
| 192 | 1 | 4 | 47.6 | 51.4 | 25 | 15 | 4.8 |
| 192 | 1 | 12 | 54.7 | 57.1 | 30 | 34 | 4.8 |
| 192 | 1 | 20 | 54.7 | 57.2 | 38 | 40 | 4.8 |
| 192 | 8 | 4 | 49.2 | 53.2 | 24 | 2 | 36.8 |
| 192 | 8 | 12 | 55.7 | 58.3 | 32 | 3 | 36.8 |
| 192 | 8 | 20 | 55.8 | 58.4 | 42 | 3 | 36.8 |

策略网络架构的卷积层含 128、192 或 256 个滤波器；显式对称集成取 2、4 或 8 种对称；输入特征平面只用扩展数据表 1（译注：原 PDF 此处印作 "Extended Data Table 1"，按上下文应指列出特征平面的扩展数据表 2，疑为原刊笔误，译文照录并在此注明）所列的前 4、12 或 20 个。结果包括 KGS 数据集上的测试与训练准确率；以及给定策略网络对 AlphaGo 的策略网络（表中高亮的第 2 行）所赢对局的百分比：直接用策略网络选着法（原始胜率，raw wins）；或用 AlphaGo 的搜索选着法（AlphaGo 胜率，AlphaGo wins）；最后是策略网络单次评估的计算时间。

**扩展数据表 4｜Rollout 策略与树策略的输入特征**

Rollout 策略使用的特征（第一组）与树策略使用的特征（第一组与第二组）。（译注：本表在原 PDF 中以棋盘图案图版排版，正文仅有以下表注，无可转录的文字行；图案基于每个图案交叉点处的子的颜色（黑/白/空）与气（1、2、≥3）。据 Methods 的 "Rollout 策略" 一节，第一组为响应图案与非响应图案特征，树策略在此之外使用更多特征；具体图案构成见原 PDF 扩展数据表 4 图版。）

表注原文：Rollout 策略（第一组）与树策略（第一组与第二组）使用的特征。图案基于图案每个交叉点处子的颜色（黑/白/空）与气（1、2、≥3）。

**扩展数据表 5｜AlphaGo 使用的参数**

| 符号 | 参数 | 值 |
|---|---|---|
| $\beta$ | Softmax 温度（softmax temperature） | 0.67 |
| $\lambda$ | 混合参数（mixing parameter） | 0.5 |
| $n_{\mathrm{vl}}$ | 虚拟损失（virtual loss） | 3 |
| $n_{\mathrm{thr}}$ | 展开阈值（expansion threshold） | 40 |
| $c_{\mathrm{puct}}$ | 探索常数（exploration constant） | 5 |

**扩展数据表 6｜不同围棋程序之间锦标赛的结果**

| 简称 | 计算机棋手 | 版本 | 用时设置 | CPU | GPU | KGS 等级 | Elo |
|---|---|---|---|---|---|---|---|
| $\alpha_{rvp}^d$ | Distributed AlphaGo（分布式） | 见 Methods | 5 seconds | 1202 | 176 | – | 3140 |
| $\alpha_{rvp}$ | AlphaGo | 见 Methods | 5 seconds | 48 | 8 | – | 2890 |
| CS | CrazyStone | 2015 | 5 seconds | 32 | – | 6d | 1929 |
| ZN | Zen | 5 | 5 seconds | 8 | – | 6d | 1888 |
| PC | Pachi | 10.99 | 400,000 sims | 16 | – | 2d | 1298 |
| FG | Fuego | svn1989 | 100,000 sims | 16 | – | – | 1148 |
| GG | GnuGo | 3.8 | level 10 | 1 | – | 5k | 431 |
| CS4 | CrazyStone | 4 handicap stones（让四子） | 5 seconds | 32 | – | – | 2526 |
| ZN4 | Zen | 4 handicap stones | 5 seconds | 8 | – | – | 2413 |
| PC4 | Pachi | 4 handicap stones | 400,000 sims | 16 | – | – | 1756 |

每个程序每步最多思考 5 秒；与樊麾的对局使用更长的用时控制，见 Methods 所述。CS4、ZN4 与 PC4 被让四子（译注：原表注印作 "CN4, ZN4 and PC4"，CS4 在注中误印为 CN4，译文按表格简称照录并在此注明）；全部对局贴目 7.5。Elo 等级分由 BayesElo 计算。

**扩展数据表 7｜AlphaGo 不同变体之间锦标赛的结果**

| 简称 | 策略网络 | 价值网络 | Rollout | 混合常数 | 策略 GPU | 价值 GPU | Elo 等级分 |
|---|---|---|---|---|---|---|---|
| $\alpha_{rvp}$ | $p_\sigma$ | $v_\theta$ | $p_\pi$ | $\lambda = 0.5$ | 2 | 6 | 2890 |
| $\alpha_{vp}$ | $p_\sigma$ | $v_\theta$ | – | $\lambda = 0$ | 2 | 6 | 2177 |
| $\alpha_{rp}$ | $p_\sigma$ | – | $p_\pi$ | $\lambda = 1$ | 8 | 0 | 2416 |
| $\alpha_{rv}$ | [$p_\tau$] | $v_\theta$ | $p_\pi$ | $\lambda = 0.5$ | 0 | 8 | 2077 |
| $\alpha_{v}$ | [$p_\tau$] | $v_\theta$ | – | $\lambda = 0$ | 0 | 8 | 1655 |
| $\alpha_{r}$ | [$p_\tau$] | – | $p_\pi$ | $\lambda = 1$ | 0 | 0 | 1457 |
| $\alpha_{p}$ | $p_\sigma$ | – | – | – | 0 | 0 | 1517 |

只用 rollout（$\alpha_{rp}$、$\alpha_r$）、只用价值网络（$\alpha_{vp}$、$\alpha_v$）或两者混合（$\alpha_{rvp}$、$\alpha_{rv}$）评估局面；或使用策略网络 $p_\sigma$（$\alpha_{rvp}$、$\alpha_{vp}$、$\alpha_{rp}$），或不用策略网络（译注：原表注此处重复印作 "($\alpha_{rvp}$, $\alpha_{vp}$, $\alpha_{rp}$)"，按上下文应为其余变体 $\alpha_{rv}$、$\alpha_v$、$\alpha_r$、$\alpha_p$，即全程改用树策略 $p_\tau$ 的占位概率，译文照录原注并在此注明），即全程改用树策略 $p_\tau$ 的占位概率。每个程序在有 48 个 CPU 与 8 个 GPU 的单机上每步用 5 秒。Elo 等级分由 BayesElo 计算。

**扩展数据表 8｜AlphaGo 与分布式 AlphaGo 之间锦标赛的结果：用硬件测试可扩展性**

| AlphaGo | 搜索线程 | CPU | GPU | Elo |
|---|---|---|---|---|
| Asynchronous（异步） | 1 | 48 | 8 | 2203 |
| Asynchronous | 2 | 48 | 8 | 2393 |
| Asynchronous | 4 | 48 | 8 | 2564 |
| Asynchronous | 8 | 48 | 8 | 2665 |
| Asynchronous | 16 | 48 | 8 | 2778 |
| Asynchronous | 32 | 48 | 8 | 2867 |
| Asynchronous | 40 | 48 | 8 | 2890 |
| Asynchronous | 40 | 48 | 1 | 2181 |
| Asynchronous | 40 | 48 | 2 | 2738 |
| Asynchronous | 40 | 48 | 4 | 2850 |
| Distributed（分布式） | 12 | 428 | 64 | 2937 |
| Distributed | 24 | 764 | 112 | 3079 |
| Distributed | 40 | 1202 | 176 | 3140 |
| Distributed | 64 | 1920 | 280 | 3168 |

每个程序每步最多思考 2 秒。Elo 等级分由 BayesElo 计算。

**扩展数据表 9｜各程序之间胜率（%）交叉表**

列首程序对行首程序的胜率（%）；方括号内为 95% Agresti–Coull 置信区间。每个程序每步最多思考 5 秒。CS4、ZN4 与 PC4 被让四子（原注印作 CN4，见表 6 译注）；全部对局贴目 7.5。分布式 AlphaGo 对 $\alpha_{rvp}$ 的胜率为 77% [70; 82]，对全部其他程序为 100%（未进行让子对局）。（译注：原表为大矩阵，排版上分两块印出；以下按原 PDF 可读出的数值转录，行列含义同上。第一块为七个 AlphaGo 变体之间的互相对照，第二块为其他程序对七个变体的数值。）

|  | $\alpha_{rvp}$ | $\alpha_{vp}$ | $\alpha_{rp}$ | $\alpha_{rv}$ | $\alpha_{r}$ | $\alpha_{v}$ | $\alpha_{p}$ |
|---|---|---|---|---|---|---|---|
| $\alpha_{rvp}$ | - | 1 [0; 5] | 5 [4; 7] | 0 [0; 4] | 0 [0; 8] | 0 [0; 19] | 0 [0; 19] |
| $\alpha_{vp}$ | 99 [95; 100] | - | 61 [52; 69] | 35 [25; 48] | 6 [1; 27] | 0 [0; 22] | 1 [0; 6] |
| $\alpha_{rp}$ | 95 [93; 96] | 39 [31; 48] | - | 13 [7; 23] | 0 [0; 9] | 0 [0; 22] | 4 [1; 21] |
| $\alpha_{rv}$ | 100 [96; 100] | 65 [52; 75] | 87 [77; 93] | - | 0 [0; 18] | 29 [8; 64] | 48 [33; 65] |
| $\alpha_{r}$ | 100 [92; 100] | 94 [73; 99] | 100 [91; 100] | 100 [82; 100] | - | 78 [45; 94] | 78 [71; 84] |
| $\alpha_{v}$ | 100 [81; 100] | 100 [78; 100] | 100 [78; 100] | 71 [36; 92] | 22 [6; 55] | - | 30 [16; 48] |
| $\alpha_{p}$ | 100 [81; 100] | 99 [94; 100] | 96 [79; 99] | 52 [35; 67] | 22 [16; 29] | 70 [52; 84] | - |
| CS | 100 [97; 100] | 74 [66; 81] | 98 [94; 99] | 80 [70; 87] | 5 [3; 7] | 36 [16; 61] | 8 [5; 14] |
| ZN | 99 [93; 100] | 84 [67; 93] | 98 [93; 99] | 92 [67; 99] | 6 [2; 19] | 40 [12; 77] | 100 [65; 100] |
| PC | 100 [98; 100] | 99 [95; 100] | 100 [98; 100] | 98 [89; 100] | 78 [73; 81] | 87 [68; 95] | 55 [47; 62] |
| FG | 100 [97; 100] | 99 [93; 100] | 100 [96; 100] | 100 [91; 100] | 78 [73; 83] | 100 [65; 100] | 65 [55; 73] |
| GG | 100 [44; 100] | 100 [34; 100] | 100 [68; 100] | 100 [57; 100] | 99 [97; 100] | 67 [21; 94] | 99 [95; 100] |
| CS4 | 77 [69; 84] | 12 [8; 18] | 53 [44; 61] | 15 [8; 24] | 0 [0; 3] | 0 [0; 30] | 0 [0; 8] |
| ZN4 | 86 [77; 92] | 25 [16; 38] | 67 [56; 76] | 14 [7; 27] | 0 [0; 12] | 0 [0; 43] | - |
| PC4 | 99 [97; 100] | 82 [75; 88] | 98 [95; 99] | 89 [79; 95] | 32 [26; 39] | 13 [3; 36] | 35 [25; 46] |

（译注：原表数值方向为"列首程序对行首程序的胜率"与"行首对列首"在排版转录中易混；上表按原 PDF 行列位置照录，读时以对 $\alpha_{rvp}$ 行为例：其对 $\alpha_{vp}$ 的对应格为 1 [0; 5]，与 $\alpha_{vp}$ 行对 $\alpha_{rvp}$ 格的 99 [95; 100] 互补，即表中数值表示列首程序的胜率。引用具体数字时请对照原 PDF。）

**扩展数据表 10｜单机可扩展性研究中各程序之间胜率（%）交叉表**

方括号内为 95% Agresti–Coull 置信区间。每个程序每步 2 秒；全部对局贴目 7.5。行列均为（线程数，GPU 数）配置：（1, 8）、（2, 8）、（4, 8）、（8, 8）、（16, 8）、（32, 8）、（40, 8）、（40, 4）、（40, 2）、（40, 1）。（译注：以下按原 PDF 行列位置转录对角线以外的数值；原表每个单元格为"胜率 [区间下界; 区间上界]"，为版面计此处只转录胜率点估计，完整置信区间见原 PDF。）

| 线程\GPU | (1,8) | (2,8) | (4,8) | (8,8) | (16,8) | (32,8) | (40,8) | (40,4) | (40,2) | (40,1) |
|---|---|---|---|---|---|---|---|---|---|---|
| (1, 8) | - | 70 | 90 | 94 | 86 | 98 | 98 | 100 | 96 | 38 |
| (2, 8) | 30 | - | 72 | 81 | 86 | 92 | 93 | 83 | 84 | 26 |
| (4, 8) | 10 | 28 | - | 62 | 71 | 82 | 84 | 81 | 78 | 18 |
| (8, 8) | 6 | 19 | 38 | - | 61 | 65 | 73 | 74 | 64 | 12 |
| (16, 8) | 14 | 14 | 29 | 39 | - | 52 | 61 | 52 | 41 | 5 |
| (32, 8) | 2 | 8 | 18 | 35 | 48 | - | 52 | 44 | 26 | 0 |
| (40, 8) | 2 | 8 | 16 | 27 | 39 | 48 | - | 43 | 41 | 4 |
| (40, 4) | 0 | 17 | 19 | 26 | 48 | 56 | 57 | - | 29 | 2 |
| (40, 2) | 4 | 16 | 22 | 36 | 59 | 74 | 59 | 71 | - | 5 |
| (40, 1) | 62 | 74 | 82 | 88 | 95 | 100 | 96 | 98 | 95 | - |

**扩展数据表 11｜分布式可扩展性研究中各程序之间胜率（%）交叉表**

方括号内为 95% Agresti–Coull 置信区间。每个程序每步 2 秒；全部对局贴目 7.5。行列均为（线程数，GPU 数，CPU 数）配置：（40, 8, 48）、（12, 64, 428）、（24, 112, 764）、（40, 176, 1202）、（64, 280, 1920）。数值为胜率点估计，置信区间见原 PDF。

| 线程/GPU/CPU | (40, 8, 48) | (12, 64, 428) | (24, 112, 764) | (40, 176, 1202) | (64, 280, 1920) |
|---|---|---|---|---|---|
| (40, 8, 48) | - | 52 | 68 | 77 | 81 |
| (12, 64, 428) | 48 | - | 64 | 62 | 83 |
| (24, 112, 764) | 32 | 36 | - | 36 | 60 |
| (40, 176, 1202) | 23 | 38 | 64 | - | 53 |
| (64, 280, 1920) | 19 | 17 | 40 | 47 | - |

（扩展数据表 10、11 的置信区间处理方式说明：两表的每个单元格在原 PDF 中均为"胜率 [95% 区间]"形式，区间数值未能从排版文本中逐格可靠分离，本译文转录胜率点估计并在此如实注明，引用时请核对原 PDF；扩展数据表 9 的区间已逐格转录。）

---

## 参考文献（原文照录）

1. Allis, L. V. *Searching for Solutions in Games and Artificial Intelligence.* PhD thesis, Univ. Limburg, Maastricht, The Netherlands (1994).
2. van den Herik, H., Uiterwijk, J. W. & van Rijswijck, J. Games solved: now and in the future. *Artif. Intell.* **134**, 277–311 (2002).
3. Schaeffer, J. The games computers (and people) play. *Advances in Computers* **52**, 189–266 (2000).
4. Campbell, M., Hoane, A. & Hsu, F. Deep Blue. *Artif. Intell.* **134**, 57–83 (2002).
5. Schaeffer, J. et al. A world championship caliber checkers program. *Artif. Intell.* **53**, 273–289 (1992).
6. Buro, M. From simple features to sophisticated evaluation functions. In *1st International Conference on Computers and Games*, 126–145 (1999).
7. Müller, M. Computer Go. *Artif. Intell.* **134**, 145–179 (2002).
8. Tesauro, G. & Galperin, G. On-line policy improvement using Monte-Carlo search. In *Advances in Neural Information Processing*, 1068–1074 (1996).
9. Sheppard, B. World-championship-caliber Scrabble. *Artif. Intell.* **134**, 241–275 (2002).
10. Bouzy, B. & Helmstetter, B. Monte-Carlo Go developments. In *10th International Conference on Advances in Computer Games*, 159–174 (2003).
11. Coulom, R. Efficient selectivity and backup operators in Monte-Carlo tree search. In *5th International Conference on Computers and Games*, 72–83 (2006).
12. Kocsis, L. & Szepesvári, C. Bandit based Monte-Carlo planning. In *15th European Conference on Machine Learning*, 282–293 (2006).
13. Coulom, R. Computing Elo ratings of move patterns in the game of Go. *ICGA J.* **30**, 198–208 (2007).
14. Baudiš, P. & Gailly, J.-L. Pachi: State of the art open source Go program. In *Advances in Computer Games*, 24–38 (Springer, 2012).
15. Müller, M., Enzenberger, M., Arneson, B. & Segal, R. Fuego – an open-source framework for board games and Go engine based on Monte-Carlo tree search. *IEEE Trans. Comput. Intell. AI in Games* **2**, 259–270 (2010).
16. Gelly, S. & Silver, D. Combining online and offline learning in UCT. In *17th International Conference on Machine Learning*, 273–280 (2007).
17. Krizhevsky, A., Sutskever, I. & Hinton, G. ImageNet classification with deep convolutional neural networks. In *Advances in Neural Information Processing Systems*, 1097–1105 (2012).
18. Lawrence, S., Giles, C. L., Tsoi, A. C. & Back, A. D. Face recognition: a convolutional neural-network approach. *IEEE Trans. Neural Netw.* **8**, 98–113 (1997).
19. Mnih, V. et al. Human-level control through deep reinforcement learning. *Nature* **518**, 529–533 (2015).
20. LeCun, Y., Bengio, Y. & Hinton, G. Deep learning. *Nature* **521**, 436–444 (2015).
21. Stern, D., Herbrich, R. & Graepel, T. Bayesian pattern ranking for move prediction in the game of Go. In *International Conference of Machine Learning*, 873–880 (2006).
22. Sutskever, I. & Nair, V. Mimicking Go experts with convolutional neural networks. In *International Conference on Artificial Neural Networks*, 101–110 (2008).
23. Maddison, C. J., Huang, A., Sutskever, I. & Silver, D. Move evaluation in Go using deep convolutional neural networks. *3rd International Conference on Learning Representations* (2015).
24. Clark, C. & Storkey, A. J. Training deep convolutional neural networks to play go. In *32nd International Conference on Machine Learning*, 1766–1774 (2015).
25. Williams, R. J. Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Mach. Learn.* **8**, 229–256 (1992).
26. Sutton, R., McAllester, D., Singh, S. & Mansour, Y. Policy gradient methods for reinforcement learning with function approximation. In *Advances in Neural Information Processing Systems*, 1057–1063 (2000).
27. Sutton, R. & Barto, A. *Reinforcement Learning: an Introduction* (MIT Press, 1998).
28. Schraudolph, N. N., Dayan, P. & Sejnowski, T. J. Temporal difference learning of position evaluation in the game of Go. *Adv. Neural Inf. Process. Syst.* **6**, 817–824 (1994).
29. Enzenberger, M. Evaluation in Go by a neural network using soft segmentation. In *10th Advances in Computer Games Conference*, 97–108 (2003). 267.
30. Silver, D., Sutton, R. & Müller, M. Temporal-difference search in computer Go. *Mach. Learn.* **87**, 183–219 (2012).
31. Levinovitz, A. The mystery of Go, the ancient game that computers still can't win. *Wired Magazine* (2014).
32. Mechner, D. All Systems Go. *The Sciences* **38**, 32–37 (1998).
33. Mandziuk, J. Computational intelligence in mind games. In *Challenges for Computational Intelligence*, 407–442 (2007).
34. Berliner, H. A chronology of computer chess and its literature. *Artif. Intell.* **10**, 201–214 (1978).
35. Browne, C. et al. A survey of Monte-Carlo tree search methods. *IEEE Trans. Comput. Intell. AI in Games* **4**, 1–43 (2012).
36. Gelly, S. et al. The grand challenge of computer Go: Monte Carlo tree search and extensions. *Commun. ACM* **55**, 106–113 (2012).
37. Coulom, R. Whole-history rating: A Bayesian rating system for players of time-varying strength. In *International Conference on Computers and Games*, 113–124 (2008).
38. KGS. Rating system math. http://www.gokgs.com/help/rmath.html.
39. Littman, M. L. Markov games as a framework for multi-agent reinforcement learning. In *11th International Conference on Machine Learning*, 157–163 (1994).
40. Knuth, D. E. & Moore, R. W. An analysis of alpha-beta pruning. *Artif. Intell.* **6**, 293–326 (1975).
41. Sutton, R. Learning to predict by the method of temporal differences. *Mach. Learn.* **3**, 9–44 (1988).
42. Baxter, J., Tridgell, A. & Weaver, L. Learning to play chess using temporal differences. *Mach. Learn.* **40**, 243–263 (2000).
43. Veness, J., Silver, D., Blair, A. & Uther, W. Bootstrapping from game tree search. In *Advances in Neural Information Processing Systems* (2009).
44. Samuel, A. L. Some studies in machine learning using the game of checkers II - recent progress. *IBM J. Res. Develop.* **11**, 601–617 (1967).
45. Schaeffer, J., Hlynka, M. & Jussila, V. Temporal difference learning applied to a high-performance game-playing program. In *17th International Joint Conference on Artificial Intelligence*, 529–534 (2001).
46. Tesauro, G. TD-gammon, a self-teaching backgammon program, achieves master-level play. *Neural Comput.* **6**, 215–219 (1994).
47. Dahl, F. Honte, a Go-playing program using neural nets. In *Machines that learn to play games*, 205–223 (Nova Science, 1999).
48. Rosin, C. D. Multi-armed bandits with episode context. *Ann. Math. Artif. Intell.* **61**, 203–230 (2011).
49. Lanctot, M., Winands, M. H. M., Pepels, T. & Sturtevant, N. R. Monte Carlo tree search with heuristic evaluations using implicit minimax backups. In *IEEE Conference on Computational Intelligence and Games*, 1–8 (2014).
50. Gelly, S., Wang, Y., Munos, R. & Teytaud, O. Modification of UCT with patterns in Monte-Carlo Go. *Tech. Rep.* **6062**, INRIA (2006).
51. Silver, D. & Tesauro, G. Monte-Carlo simulation balancing. In *26th International Conference on Machine Learning*, 119 (2009).
52. Huang, S.-C., Coulom, R. & Lin, S.-S. Monte-Carlo simulation balancing in practice. In *7th International Conference on Computers and Games*, 81–92 (Springer-Verlag, 2011).
53. Baier, H. & Drake, P. D. The power of forgetting: improving the last-good-reply policy in Monte Carlo Go. *IEEE Trans. Comput. Intell. AI in Games* **2**, 303–309 (2010).
54. Huang, S. & Müller, M. Investigating the limits of Monte-Carlo tree search methods in computer Go. In *8th International Conference on Computers and Games*, 39–48 (2013).
55. Segal, R. B. On the scalability of parallel UCT. *Computers and Games* **6515**, 36–47 (2011).
56. Enzenberger, M. & Müller, M. A lock-free multithreaded Monte-Carlo tree search algorithm. In *12th Advances in Computer Games Conference*, 14–20 (2009).
57. Huang, S.-C., Coulom, R. & Lin, S.-S. Time management for Monte-Carlo tree search applied to the game of Go. In *International Conference on Technologies and Applications of Artificial Intelligence*, 462–466 (2010).
58. Gelly, S. & Silver, D. Monte-Carlo tree search and rapid action value estimation in computer Go. *Artif. Intell.* **175**, 1856–1875 (2011).
59. Baudiš, P. Balancing MCTS by dynamically adjusting the komi value. *ICGA J.* **34**, 131 (2011).
60. Baier, H. & Winands, M. H. Active opening book application for Monte-Carlo tree search in 19×19 Go. In *Benelux Conference on Artificial Intelligence*, 3–10 (2011).
61. Dean, J. et al. Large scale distributed deep networks. In *Advances in Neural Information Processing Systems*, 1223–1231 (2012).
62. Go ratings. http://www.goratings.org.
