# AI Landmark Papers: From Foundations to the Frontier — Vol. 1 Foundations (1943–2006)

English edition; abstracts are quoted verbatim from the original papers where available.

This volume collects eight papers, from the first mathematical model of an artificial neuron to the deep belief network that reignited deep learning. Each entry gives the bibliographic record, the abstract, the key ideas, and the paper's place in the lineage. Where an early paper has no abstract in the modern sense, this is stated plainly and no substitute is passed off as one.

---

### 1. A Logical Calculus of the Ideas Immanent in Nervous Activity

- **Authors & year**: Warren S. McCulloch and Walter Pitts, 1943, *Bulletin of Mathematical Biophysics*, Vol. 5, No. 4, pp. 115–133
- **One-line significance**: The first mathematical model of an artificial neuron in history, and the source of the entire lineage.

**Abstract**

The original has no separate abstract section. Its first paragraph, however, is an outline written by the authors themselves and serves the same purpose; it is quoted verbatim below (source: original paper, first paragraph; reprinted in McCulloch's collected works).

> Because of the "all-or-none" character of nervous activity, neural events and the relations among them can be treated by means of propositional logic. It is found that the behavior of every net can be described in these terms, with the addition of more complicated logical means for nets containing circles; and that for any logical expression satisfying certain conditions, one can find a net behaving in the fashion it describes. It is shown that many particular choices among possible neurophysiological assumptions are equivalent, in the sense that for every net behaving under one assumption, there exists another net which behaves under the other and gives the same results, although perhaps not in the same time. Various applications of the calculus are discussed.

**Key ideas**

- A neuron is reduced to a threshold logic unit: it fires when the total excitatory input reaches its threshold, and stays silent otherwise; an inhibitory input has an absolute veto. A neuron thereby becomes a logic gate performing AND, OR, and NOT.
- Because a single neuron is a logic gate, the behaviour of a whole network can be written as a formula of propositional logic; conversely, given any logical expression, a network can be built that realises it. Neural activity acquired a precise mathematical language for the first time.
- Networks with circles (feedback), supplemented by an external storage device, are equivalent in computing power to a Turing machine — the activity of the brain can, in principle, be formalised as computation.

**Relation to preceding and following work**

This is where the lineage begins: it supplies the basic component, the "neuron," but the component's parameters must be set by hand — the network cannot learn by itself. Fifteen years later, Rosenblatt's perceptron (No. 3 in this volume) supplies exactly the missing lesson: learning.

**Original link**: https://doi.org/10.1007/BF02478259

---

### 2. Computing Machinery and Intelligence

- **Authors & year**: Alan M. Turing, 1950, *Mind*, Vol. 59, No. 236, pp. 433–460
- **One-line significance**: It does not discuss an algorithm but a question — it gives "Can machines think?" an operational test, and is the founding philosophical text of artificial intelligence.

**Abstract**

The original has no abstract (philosophical journals in 1950 had no such section). What follows is a reading guide, with the opening sentence quoted. Turing begins head-on: "I propose to consider the question, 'Can machines think?'" He immediately observes that both "machine" and "think" are hard to define, and replaces the question with a testable game: an interrogator converses in writing with a machine and a human, and if the interrogator cannot tell which is the machine, the machine passes. This is what later came to be called the Turing test; in the paper it is "the imitation game." Turing then rehearses and answers nine classes of objection, among them the theological objection, the "heads in the sand" objection, the mathematical objection (by way of Gödel's theorem), the argument from consciousness, and the objection that a machine can only do what it is told to do. He closes with a prediction: in about fifty years, programmers will be able to build machines with enough storage that, after five minutes of questioning, an average interrogator will have no more than a 70 per cent chance of making the right identification — that is, the machine will mislead the interrogator at least 30 per cent of the time. And rather than simulating the adult mind, he proposes simulating a child's mind and then educating it — this "child machine" proposal is the earliest statement of the machine-learning programme.

**Key ideas**

- It sidesteps the definitional dispute over "thinking" and substitutes a behavioural criterion: indistinguishability counts. This operationalist move became the source of the whole tradition of AI evaluation.
- The "child machine" route: build a simple, nearly blank machine first, then let it acquire competence through training and reward and punishment — in essence, replacing programming with education, and foreshadowing the learning paradigm.
- Its reply to the mathematical objection is measured but unyielding: the limitations of any one machine do not preclude us from expecting another machine free of those limitations.

**Relation to preceding and following work**

McCulloch and Pitts showed that neural networks can, in principle, compute; Turing asks the prior, philosophical question: if a machine computes, is it thereby "thinking"? Every subsequent wave of AI enthusiasm has returned to this paper to re-argue the point. Its links to the other papers in this volume are ones of intellectual resonance rather than technical descent; the Turing test will be brought back on stage in Volume 2's sequence-to-sequence work and in Volume 3's era of large models.

**Original link**: https://doi.org/10.1093/mind/LIX.236.433

---

### 3. The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain

- **Authors & year**: Frank Rosenblatt, 1958, *Psychological Review*, Vol. 65, No. 6, pp. 386–408
- **One-line significance**: The first neural network model that could learn to classify from examples by itself — the moment "learning" entered neural networks.

**Abstract**

The original has no abstract. The paper opens with three questions and closes with its conclusions; the following are verified excerpts from the original, given here as a reading guide rather than as an abstract.

The three opening questions:

> (1) How is information about the physical world sensed, or detected, by the biological system? (2) In what form is information stored, or remembered? (3) How does information contained in storage, or in memory, influence recognition and behavior?

A central claim of the paper:

> Whatever information is retained must somehow be stored as a preference for a particular response; i.e., the information is contained in connections or associations rather than topographic representation.

From the conclusion: in an environment of random stimuli, a system composed of randomly connected units, satisfying the parameter constraints discussed in the paper, can learn to associate specific responses with specific stimuli.

The paper's final sentence expresses the hope that, by studying systems such as the perceptron, we may eventually understand the basic laws of organization common to all information-processing systems, machines and men included.

**Key ideas**

- The structure has three layers: sensory units (S-units) receive the stimulus, association units (A-units) sum it up over random connections, and response units (R-units) produce the output. Crucially, the connection weights are no longer constants but quantities that can change — the most fundamental difference from the McCulloch–Pitts model.
- The learning rule is error-driven: when the answer is wrong, adjust the weights in the direction that reduces the error; when it is right, leave them alone. Rosenblatt later proved the perceptron convergence theorem: if the data are linearly separable, this rule is guaranteed to find a correct solution in finitely many steps.
- The stance is thoroughgoing connectionism and empiricism: knowledge resides in no single unit but is distributed across the overall pattern of connection strengths; and the system must be analysed in the language of probability, because the exact structure of biological nervous systems is not, and need not be, known.

**Relation to preceding and following work**

It turned McCulloch and Pitts's fixed logic gates into a plastic learning machine. But a single layer can only draw one straight line (a hyperplane); it cannot solve even so simple a nonlinear problem as XOR. In their 1969 book *Perceptrons*, Minsky and Papert spelled this limitation out, and neural networks fell silent for more than a decade. Rescuing them required making the intermediate layers learnable as well — and that had to wait for No. 4 in this volume.

**Original link**: https://doi.org/10.1037/h0042519

---

### 4. Learning Representations by Back-Propagating Errors ★

- **Authors & year**: David E. Rumelhart, Geoffrey E. Hinton, and Ronald J. Williams, 1986, *Nature*, Vol. 323, pp. 533–536
- **One-line significance**: It brought backpropagation before the whole field; multilayer networks became genuinely trainable, and connectionism was revived.

**Abstract** (source: Nature, publisher's page for the article)

> We describe a new learning procedure, back-propagation, for networks of neurone-like units. The procedure repeatedly adjusts the weights of the connections in the network so as to minimize a measure of the difference between the actual output vector of the net and the desired output vector. As a result of the weight adjustments, internal 'hidden' units which are not part of the input or output come to represent important features of the task domain, and the regularities in the task are captured by the interactions of these units. The ability to create useful new features distinguishes back-propagation from earlier, simpler methods such as the perceptron-convergence procedure.

**Key ideas**

- It solves the credit assignment problem: when the output is wrong, how is the blame to be apportioned to every weight buried inside the network? The answer is the chain rule — propagate the error signal back layer by layer, give each weight a gradient equal to its share of responsibility for the error, and nudge it in the direction of gradient descent.
- Hidden units are not decoration. In experiments on XOR, encoding, and family relationships, the hidden units spontaneously formed internal representations that were meaningful for the task — features no longer had to be designed by hand. That is the weight carried by "learning representations" in the title.
- The method itself was not the trio's invention ex nihilo: reverse-mode automatic differentiation already had a lineage in control theory, Werbos's 1974 dissertation had applied it to multilayer networks, and LeCun had independently given a similar derivation in 1985. The achievement of this short *Nature* paper was to prove, with a set of elegant experiments, that the method really could learn — and to convince the entire field.

**Relation to preceding and following work**

It directly untied the knot Rosenblatt left behind: what a single-layer perceptron cannot learn, a network with one hidden layer trained by backpropagation can. It also became the common tool of nearly everything that follows — LeCun's convolutional networks are trained with it, the LSTM analysis starts from where it fails on recurrent networks, and Hinton's 2006 work was devised to get around its inability to train deep networks from scratch.

**Original link**: https://www.nature.com/articles/323533a0

---

### 5. Gradient-Based Learning Applied to Document Recognition

- **Authors & year**: Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner, 1998, *Proceedings of the IEEE*, Vol. 86, No. 11, pp. 2278–2324
- **One-line significance**: The consummation of the convolutional neural network (CNN): it sets out LeNet-5 and presents the first commercial deep learning system, reading several million cheques a day.

**Abstract** (source: Proceedings of the IEEE / IEEE Xplore)

> Multilayer neural networks trained with the back-propagation algorithm constitute the best example of a successful gradient based learning technique. Given an appropriate network architecture, gradient-based learning algorithms can be used to synthesize a complex decision surface that can classify high-dimensional patterns, such as handwritten characters, with minimal preprocessing. This paper reviews various methods applied to handwritten character recognition and compares them on a standard handwritten digit recognition task. Convolutional neural networks, which are specifically designed to deal with the variability of 2D shapes, are shown to outperform all other techniques. Real-life document recognition systems are composed of multiple modules including field extraction, segmentation recognition, and language modeling. A new learning paradigm, called graph transformer networks (GTN), allows such multimodule systems to be trained globally using gradient-based methods so as to minimize an overall performance measure. Two systems for online handwriting recognition are described. Experiments demonstrate the advantage of global training, and the flexibility of graph transformer networks. A graph transformer network for reading a bank cheque is also described. It uses convolutional neural network character recognizers combined with global training techniques to provide record accuracy on business and personal cheques. It is deployed commercially and reads several million cheques per day.

**Key ideas**

- Convolution with weight sharing: the same set of feature detectors slides across the whole image, which drastically cuts the number of parameters and naturally tolerates shifts and deformations of the character; subsampling layers then compress the representation step by step, yielding high-level features that are insensitive to deformation. This architecture later went by the name LeNet-5.
- One methodological claim runs through the whole paper: hand-designed feature extractors are a dead end; a system should consume raw pixels directly and learn its features from data as one trainable whole. Saying this in 1998 took confidence, and the confidence came from the experimental numbers.
- Graph transformer networks widen the view from a single recogniser to the entire system: segmentation, recognition, and language modelling are no longer separate fiefdoms but are assembled into one differentiable graph and trained end-to-end against a global objective. This is an early, complete specimen of the later credo that everything should be differentiable and trained together.

**Relation to preceding and following work**

It put the backpropagation of No. 4 to work on its first large-scale real task, proving that gradient-based learning was no laboratory toy. Fourteen years later, AlexNet in Volume 2 uses exactly this convolutional architecture — only with scale, data, and computing power all magnified by orders of magnitude.

**Original link**: https://doi.org/10.1109/5.726791

---

### 6. Long Short-Term Memory

- **Authors & year**: Sepp Hochreiter and Jürgen Schmidhuber, 1997, *Neural Computation*, Vol. 9, No. 8, pp. 1735–1780
- **One-line significance**: It fitted recurrent neural networks with a controllable memory channel, solved the long-term dependency problem, and remained the default choice for sequence modelling for nearly twenty years.

**Abstract** (source: PubMed PMID 9377276)

> Learning to store information over extended time intervals by recurrent backpropagation takes a very long time, mostly because of insufficient, decaying error backflow. We briefly review Hochreiter's (1991) analysis of this problem, then address it by introducing a novel, efficient, gradient-based method called long short-term memory (LSTM). Truncating the gradient where this does not do harm, LSTM can learn to bridge minimal time lags in excess of 1000 discrete-time steps by enforcing constant error flow through constant error carousels within special units. Multiplicative gate units learn to open and close access to the constant error flow. LSTM is local in space and time; its computational complexity per time step and weight is O(1). Our experiments with artificial data involve local, distributed, real-valued, and noisy pattern representations. In comparisons with real-time recurrent learning, back propagation through time, recurrent cascade correlation, Elman nets, and neural sequence chunking, LSTM leads to many more successful runs, and learns much faster. LSTM also solves complex, artificial long-time-lag tasks that have never been solved by previous recurrent network algorithms.

**Key ideas**

- Diagnose before prescribing: ordinary recurrent neural networks (RNNs) fail to learn long dependencies because the error signal, multiplied repeatedly as it travels back through time, must either decay or explode exponentially — later known universally as the vanishing and exploding gradient problems.
- The constant error carousel is the heart of the remedy: the memory cell's self-connection is fixed at weight 1, so error flowing through it is neither amplified nor diminished and can be preserved intact for over a thousand steps. Memory is no longer "repeatedly rewritten" but "kept as it is."
- Gates make the decisions: an input gate controls what may be written in, an output gate what may be read out, and the multiplicative structure lets the network learn the switching itself. Note that the original 1997 version has no forget gate — Gers and colleagues added it in 1999; the LSTM in universal use today is the improved version.

**Relation to preceding and following work**

Backpropagation solved "deep"; LSTM solved "long." It gave neural networks a firm foothold in sequence tasks such as speech, handwriting, and machine translation, until the Transformer of 2017 (Volume 3) took over its throne by a different route, the attention mechanism.

**Original link**: https://doi.org/10.1162/neco.1997.9.8.1735

---

### 7. Support-Vector Networks

- **Authors & year**: Corinna Cortes and Vladimir Vapnik, 1995, *Machine Learning*, Vol. 20, No. 3, pp. 273–297
- **One-line significance**: The formal debut of the support vector machine (SVM); for more than a decade afterwards it outperformed neural networks on most tasks and was the mainstream of machine learning before the deep learning revival.

**Abstract** (source: original paper, Springer)

> The support-vector network is a new learning machine for two-group classification problems. The machine conceptually implements the following idea: input vectors are non-linearly mapped to a very high-dimension feature space. In this feature space a linear decision surface is constructed. Special properties of the decision surface ensures high generalization ability of the learning machine. The idea behind the support-vector network was previously implemented for the restricted case where the training data can be separated without errors. We here extend this result to non-separable training data. High generalization ability of support-vector networks utilizing polynomial input transformations is demonstrated. We also compare the performance of the support-vector network to various classical learning algorithms that all took part in a benchmark study of Optical Character Recognition.

**Key ideas**

- Maximum margin: infinitely many planes can separate two classes; the SVM chooses the one farthest from the samples on both sides. The larger the margin, the greater the confidence on new samples — generalisation acquired a clear geometric interpretation for the first time, backed by Vapnik's statistical learning theory (VC dimension and related machinery).
- The kernel trick: if the data cannot be separated in the original space, map it into a high-dimensional space and separate it there. The elegant part is that the algorithm in fact only ever needs inner products between pairs of samples, so a kernel function can stand in for the explicit mapping — however high the dimension, the computation remains affordable.
- The soft margin is this paper's new contribution: real data cannot be separated cleanly, so a few samples are allowed to cross the boundary, priced by slack variables and fined in the objective — an explicit, quantified trade-off between margin and misclassification. The final decision depends only on the handful of samples lying on the margin boundary: the support vectors.

**Relation to preceding and following work**

Also a linear classifier, it takes a completely different road from the perceptron: the perceptron learns by correcting mistakes, the SVM by globally optimising an objective with theoretical guarantees. Within this volume it is the outsider — it does not belong to the neural network lineage, yet it held the advantage from 1995 to about 2010, throwing into relief how hard neural networks then were to train. After the deep learning revival, SVMs retreated to small-sample settings, but the ideas of kernels and margins seeped into much later work.

**Original link**: https://doi.org/10.1007/BF00994018

---

### 8. A Fast Learning Algorithm for Deep Belief Nets

- **Authors & year**: Geoffrey E. Hinton, Simon Osindero, and Yee-Whye Teh, 2006, *Neural Computation*, Vol. 18, No. 7, pp. 1527–1554
- **One-line significance**: Unsupervised layer-wise pretraining got deep networks to train; both the name "deep learning" and the revival itself are reckoned from here.

**Abstract** (source: PubMed PMID 16764513)

> We show how to use "complementary priors" to eliminate the explaining-away effects that make inference difficult in densely connected belief nets that have many hidden layers. Using complementary priors, we derive a fast, greedy algorithm that can learn deep, directed belief networks one layer at a time, provided the top two layers form an undirected associative memory. The fast, greedy algorithm is used to initialize a slower learning procedure that fine-tunes the weights using a contrastive version of the wake-sleep algorithm. After fine-tuning, a network with three hidden layers forms a very good generative model of the joint distribution of handwritten digit images and their labels. This generative model gives better digit classification than the best discriminative learning algorithms. The low-dimensional manifolds on which the digits lie are modeled by long ravines in the free-energy landscape of the top-level associative memory, and it is easy to explore these ravines by using the directed connections to display what the associative memory has in mind.

**Key ideas**

- Deep networks failed to train at the time not for lack of expressive power, but because, starting from a random initialisation and fine-tuning by backpropagation, the signal could not reach the depths. The remedy was to change the starting point: ignore the labels at first, and pretrain unsupervised, layer by layer, letting each layer first learn to represent the activity of the one below.
- The building block of the layer-wise greedy procedure is the restricted Boltzmann machine (RBM), trained quickly with contrastive divergence; stacked to the top, the uppermost two layers join into an undirected associative memory, and the whole stack is called a deep belief network (DBN).
- Once pretraining has set the rough position, a contrastive version of the wake-sleep algorithm (the up-down procedure) fine-tunes the generative model — and for a discriminative task, supervised fine-tuning by backpropagation can follow. On MNIST, the generative model trained this way classified digits better than the best discriminative methods of the day: a deep network had won a public comparison for the first time.

**Relation to preceding and following work**

It picks up the thread of No. 4: twenty years on, Hinton returned to multilayer networks and gave backpropagation a good starting point. Together with the contemporaneous work of Bengio and others, this paper launched "deep learning" as a banner. Later, better initialisation methods, normalisation, and labelled data at far larger scale made pretraining itself unnecessary — but "deep" has been the default ambition ever since, leading directly to AlexNet in Volume 2.

**Original link**: https://doi.org/10.1162/neco.2006.18.7.1527

---

*Verification note for this volume: the bibliographic details and abstracts of all eight papers were checked against publisher pages (Nature, IEEE, PubMed) or the original PDFs. No. 1 uses the paper's first paragraph (the authors' own outline) in place of an abstract; Nos. 2 and 3 have no abstract in the modern sense, as noted in each entry; the abstract section of No. 3 consists of verified excerpts from the original (the three opening questions, a central claim, the conclusion, and the final sentence).*
