# AI Landmark Papers: From Foundations to the Frontier — Vol. 2 The Deep Learning Revolution (2012–2016)

English edition; abstracts are quoted verbatim from the original papers where available.

This volume collects eight papers, from AlexNet's stunning result on ImageNet to AlphaGo's conquest of Go. In these five years, deep learning advanced from "deep networks can be trained" all the way to systems that can see, translate, create, and play Go. Each entry gives the bibliographic record, the abstract, the key ideas, and the paper's place in the lineage.

---

### 1. ImageNet Classification with Deep Convolutional Neural Networks ★

- **Authors & year**: Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton, 2012, *Advances in Neural Information Processing Systems* (NeurIPS), Vol. 25
- **One-line significance**: The starting gun of the deep learning revolution — a convolutional network won on a million images by a crushing margin, and the era of hand-crafted features came to an end.

**Abstract** (source: original paper, NeurIPS 2012 proceedings; the paper was never posted to arXiv)

> We trained a large, deep convolutional neural network to classify the 1.2 million high-resolution images in the ImageNet LSVRC-2010 contest into the 1000 different classes. On the test data, we achieved top-1 and top-5 error rates of 37.5% and 17.0% which is considerably better than the previous state-of-the-art. The neural network, which has 60 million parameters and 650,000 neurons, consists of five convolutional layers, some of which are followed by max-pooling layers, and three fully-connected layers with a final 1000-way softmax. To make training faster, we used non-saturating neurons and a very efficient GPU implementation of the convolution operation. To reduce overfitting in the fully-connected layers we employed a recently-developed regularization method called "dropout" that proved to be very effective. We also entered a variant of this model in the ILSVRC-2012 competition and achieved a winning top-5 test error rate of 15.3%, compared to 26.2% achieved by the second-best entry.

**Key ideas**

- There are no new components: everything here is Volume 1 material, scaled up. The convolutional architecture descends from LeCun's LeNet, and training is by backpropagation. The real variable is scale — the data grow from tens of thousands of images to 1.2 million, the parameters to 60 million, and two GPUs make the whole thing trainable. The result showed that the architecture not only holds up on big data, but wins without contest.
- The activation function becomes the rectified linear unit (ReLU): non-saturating, with gradients that do not decay, it trains several times faster than tanh. Much of what made deep networks slow to train sat in the activation function, and this choice later became the default.
- Dropout is the main weapon against overfitting: during training, half of the fully-connected neurons are randomly made to "sit out" each step, which amounts to training a large ensemble of parameter-sharing sub-networks and averaging them. Combined with data augmentation (random crops, flips, and changes to lighting and colour), it kept 60 million parameters from simply memorising the training set.
- The paper established an engineering route: feed in raw pixels end-to-end, and stop hand-designing feature extractors. From then on, the centre of gravity in vision research moved wholesale from "designing features" to "designing architectures and training methods."

**Relation to preceding and following work**

It makes good, directly, on the promise of LeCun et al. 1998 (No. 5 in Volume 1) — only with the scale magnified by several orders of magnitude. Its success also exposed a new problem at once: try to go deeper, and training turns unstable and optimisation fails. Batch normalisation (No. 6 in this volume) and ResNet (No. 7) repair exactly that wall.

**Original link**: https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

---

### 2. Sequence to Sequence Learning with Neural Networks

- **Authors & year**: Ilya Sutskever, Oriol Vinyals, and Quoc V. Le, 2014, *Advances in Neural Information Processing Systems* (NeurIPS), Vol. 27
- **One-line significance**: It proposed the encoder–decoder framework — one network reads a sentence, another writes a sentence — and end-to-end machine translation starts here.

**Abstract** (source: arXiv:1409.3215, current version, v3)

> Deep Neural Networks (DNNs) are powerful models that have achieved excellent performance on difficult learning tasks. Although DNNs work well whenever large labeled training sets are available, they cannot be used to map sequences to sequences. In this paper, we present a general end-to-end approach to sequence learning that makes minimal assumptions on the sequence structure. Our method uses a multilayered Long Short-Term Memory (LSTM) to map the input sequence to a vector of a fixed dimensionality, and then another deep LSTM to decode the target sequence from the vector. Our main result is that on an English to French translation task from the WMT'14 dataset, the translations produced by the LSTM achieve a BLEU score of 34.8 on the entire test set, where the LSTM's BLEU score was penalized on out-of-vocabulary words. Additionally, the LSTM did not have difficulty on long sentences. For comparison, a phrase-based SMT system achieves a BLEU score of 33.3 on the same dataset. When we used the LSTM to rerank the 1000 hypotheses produced by the aforementioned SMT system, its BLEU score increases to 36.5, which is close to the previous best result on this task. The LSTM also learned sensible phrase and sentence representations that are sensitive to word order and are relatively invariant to the active and the passive voice. Finally, we found that reversing the order of the words in all source sentences (but not target sentences) improved the LSTM's performance markedly, because doing so introduced many short term dependencies between the source and the target sentence which made the optimization problem easier.

**Key ideas**

- The framework fits in one sentence: an encoder LSTM reads the source sentence word by word and compresses the whole sentence into a fixed-length vector; a decoder LSTM starts from that vector and generates the target sentence word by word. The two networks join head to tail with no hand-built linguistics in between — no phrase tables, alignment tables, or reordering rules.
- That fixed-length vector (later often called the "thought vector") is both the essence of the method and its weak point: everything a sentence contains has to be squeezed into it, and the longer the sentence, the harder the squeeze. The next paper in this volume, by Bahdanau and colleagues, aims straight at this bottleneck.
- Reversing the source sentence looks like a gimmick but is not: feeding the input in backwards brings the start of the source close in time to the start of the translation, multiplying short-term dependencies and making gradients easier to propagate. The experimental numbers genuinely improved — a reminder that optimisation difficulty often hides in details of this kind.
- The learned sentence representations are sensitive to word order yet relatively invariant to active–passive conversion, which shows the vector holds more than surface word information. This was early evidence for the later idea that every sequence should first be encoded into a vector.

**Relation to preceding and following work**

It took the LSTM of Volume 1 (No. 6) — a structure for remembering sequences — and turned it into a general skeleton for sequence-to-sequence conversion. Its fixed-length bottleneck was promptly broken by the attention mechanism of the next paper; the encoder–decoder framework itself survived, and is the direct ancestor of the Transformer in Volume 3.

**Original link**: https://arxiv.org/abs/1409.3215v3

---

### 3. Neural Machine Translation by Jointly Learning to Align and Translate ★

- **Authors & year**: Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio, 2015, 3rd International Conference on Learning Representations (ICLR 2015)
- **One-line significance**: The birth certificate of the attention mechanism — the model learns, as it generates each word, to look back at the relevant positions of the source sentence, and Seq2Seq's fixed-length bottleneck was dismantled.

**Abstract** (source: arXiv:1409.0473)

> Neural machine translation is a recently proposed approach to machine translation. Unlike the traditional statistical machine translation, the neural machine translation aims at building a single neural network that can be jointly tuned to maximize the translation performance. The models proposed recently for neural machine translation often belong to a family of encoder-decoders and consists of an encoder that encodes a source sentence into a fixed-length vector from which a decoder generates a translation. In this paper, we conjecture that the use of a fixed-length vector is a bottleneck in improving the performance of this basic encoder-decoder architecture, and propose to extend this by allowing a model to automatically (soft-)search for parts of a source sentence that are relevant to predicting a target word, without having to form these parts as a hard segment explicitly. With this new approach, we achieve a translation performance comparable to the existing state-of-the-art phrase-based system on the task of English-to-French translation. Furthermore, qualitative analysis reveals that the (soft-)alignments found by the model agree well with our intuition.

**Key ideas**

- The diagnosis is precise: Seq2Seq compresses all of a sentence's information into one fixed-length vector, so long sentences inevitably lose information. The remedy is not a bigger vector but no compression at all — the encoder keeps a vector for every position of the source sentence, and the decoder picks what it needs, as it needs it.
- The mechanism is called soft alignment: before generating the *i*-th target word, the model computes a relevance weight for every source position and forms a context vector for this step alone as the weighted sum of the position vectors. The weights come from a small alignment model that is trained jointly with the translation model as a single whole — "alignment" is no longer a preprocessing step but a behaviour that grows out of training.
- The encoder becomes a bidirectional RNN: the annotation vector at each source position carries the context from both its left and its right, so when the decoder looks back it sees not an isolated word but the word's situation within the sentence.
- The gains are largest on long sentences: where the basic encoder–decoder's quality falls off rapidly with sentence length, this model's barely falls at all. A side benefit is interpretability — draw the weights as an alignment matrix and one can see at a glance where the model is "looking," and it largely matches human alignment intuition.

**Relation to preceding and following work**

It is a pinpoint repair of the preceding paper, but it opened up far more than translation: looking back over the input, weighted by relevance, is exactly the operation later called the attention mechanism. In 2017 the Transformer (Volume 3) discarded the recurrent structure altogether and kept only attention — on the strength of the seed planted here.

**Original link**: https://arxiv.org/pdf/1409.0473

---

### 4. Generative Adversarial Nets

- **Authors & year**: Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio, 2014, *Advances in Neural Information Processing Systems* (NeurIPS), Vol. 27
- **One-line significance**: Two networks trained against each other learn to generate — generative models no longer needed an explicit probability formula, and the route of implicit generative modelling was opened.

**Abstract** (source: arXiv:1406.2661)

> We propose a new framework for estimating generative models via an adversarial process, in which we simultaneously train two models: a generative model G that captures the data distribution, and a discriminative model D that estimates the probability that a sample came from the training data rather than G. The training procedure for G is to maximize the probability of D making a mistake. This framework corresponds to a minimax two-player game. In the space of arbitrary functions G and D, a unique solution exists, with G recovering the training data distribution and D equal to 1/2 everywhere. In the case where G and D are defined by multilayer perceptrons, the entire system can be trained with backpropagation. There is no need for any Markov chains or unrolled approximate inference networks during either training or generation of samples. Experiments demonstrate the potential of the framework through qualitative and quantitative evaluation of the generated samples.

**Key ideas**

- A contest between a forger and an appraiser: G starts from random noise and fabricates samples; D judges whether a sample is real or fake. As D grows more discerning, G can fool it only by fabricating ever more realistically. The training objective is written as a minimax game: D maximises the probability of judging correctly, G minimises it. In theory the game's equilibrium sits where G's distribution coincides exactly with the true data distribution, at which point D can only guess on any sample and outputs 1/2 everywhere.
- It sidesteps the old obstacles of generative modelling: no need to write down the data's probability density explicitly, no intractable partition-function estimation, and no Markov chain sampling for training. The supervisory signal comes from D's verdicts, and gradients flow back into G just the same — as long as G and D are both differentiable networks, backpropagation covers the whole process.
- The cost is built into the framework: a game is not the same as optimising one fixed objective, so training is inherently unstable. A great deal of later work went into mode collapse (G producing only a few kinds of sample that score well), vanishing gradients, and how to tell whether training has converged — open accounts this paper left behind.
- The idea's provenance is worth noting: it borrows the Nash equilibrium of game theory, not the language of probabilistic graphical models. The other generative route to appear in the same year, 2014, is the variational autoencoder of the next paper, and the two form a natural contrast.

**Relation to preceding and following work**

The deep belief network of Volume 1 is the old source of the generative line, but its training and inference were both cumbersome. The GAN and the VAE that follows it offered two new answers in the same year: one implicit and adversarial, the other explicit and variational. Generative modelling then raced along these two lines for years, until the diffusion models of Volume 3 overtook both.

**Original link**: https://arxiv.org/pdf/1406.2661

---

### 5. Auto-Encoding Variational Bayes

- **Authors & year**: Diederik P. Kingma and Max Welling, arXiv preprint, December 2013; published at the 2nd International Conference on Learning Representations (ICLR 2014), conference track
- **One-line significance**: The source of the variational autoencoder (VAE) — the reparameterisation trick lets stochastic latent variables pass gradients too, and variational inference was connected to deep learning.

**Abstract** (source: arXiv:1312.6114)

> How can we perform efficient inference and learning in directed probabilistic models, in the presence of continuous latent variables with intractable posterior distributions, and large datasets? We introduce a stochastic variational inference and learning algorithm that scales to large datasets and, under some mild differentiability conditions, even works in the intractable case. Our contributions are two-fold. First, we show that a reparameterization of the variational lower bound yields a lower bound estimator that can be straightforwardly optimized using standard stochastic gradient methods. Second, we show that for i.i.d. datasets with continuous latent variables per datapoint, posterior inference can be made especially efficient by fitting an approximate inference model (also called a recognition model) to the intractable posterior using the proposed lower bound estimator. Theoretical advantages are reflected in experimental results.

**Key ideas**

- The roadblock is that sampling is not differentiable: the latent variable *z* must be sampled from the approximate posterior *q*(*z*|*x*), the sampling step has no gradient, and backpropagation cannot reach the encoder. The reparameterisation trick outsources the randomness: instead of sampling from *q* directly, draw *ε* from a noise distribution that does not depend on the parameters, and let *z* be computed from *x* and *ε* by a deterministic function. The randomness moves to the input side, and the rest of the path is differentiable throughout.
- The recognition model "amortises" inference: classical variational inference optimises posterior parameters separately, by iteration, for every data point; here a single neural network (the encoder) looks at *x* and outputs the posterior parameters directly, and all data points share this one inference network. The cost of inference drops from per-point iteration to a single forward pass.
- The objective is the variational lower bound (also called the evidence lower bound, ELBO), which splits naturally into two terms: reconstruction quality — how well the decoder (the generative model) restores *x* from *z* — and the KL divergence between the approximate posterior and the prior, which acts as a regulariser and keeps the latent space continuous and well-organised, so that new samples can be generated from it. Encoder plus decoder is the finished product: the variational autoencoder.
- Against the deep belief network of Volume 1, there is no layer-wise greediness and no contrastive divergence here; the whole apparatus trains end-to-end in one pass. Against the GAN of the same year, this is an explicit probabilistic model: the likelihood has a computable lower bound and the latent space has a definite meaning — at the price of samples that tend to be blurry.

**Relation to preceding and following work**

It picks up the generative-model thread of Volume 1 (No. 8), replacing Boltzmann-machine-style intractable inference with differentiable, amortised inference. Together with the GAN it stands as one of the two great sources of deep generative modelling. Further ahead, the latent diffusion models of Volume 3 first use an autoencoder of this kind to compress images into a latent space and then generate within that space — the latent-variable line is alive to this day.

**Original link**: https://arxiv.org/pdf/1312.6114

---

### 6. Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift

- **Authors & year**: Sergey Ioffe and Christian Szegedy, 2015, 32nd International Conference on Machine Learning (ICML 2015)
- **One-line significance**: Normalisation made into a standard layer of the network — deep training could thereafter use much higher learning rates and stop worrying about initialisation, with training steps cut by an order of magnitude.

**Abstract** (source: ICML 2015 proceedings as published in PMLR 37)

> Training Deep Neural Networks is complicated by the fact that the distribution of each layer's inputs changes during training, as the parameters of the previous layers change. This slows down the training by requiring lower learning rates and careful parameter initialization, and makes it notoriously hard to train models with saturating nonlinearities. We refer to this phenomenon as internal covariate shift, and address the problem by normalizing layer inputs. Our method draws its strength from making normalization a part of the model architecture and performing the normalization for each training mini-batch. Batch Normalization allows us to use much higher learning rates and be less careful about initialization, and in some cases eliminates the need for Dropout. Applied to a state-of-the-art image classification model, Batch Normalization achieves the same accuracy with 14 times fewer training steps, and beats the original model by a significant margin. Using an ensemble of batch-normalized networks, we improve upon the best published result on ImageNet classification: reaching 4.82% top-5 test error, exceeding the accuracy of human raters.

**Key ideas**

- First, name the disease: internal covariate shift. Layer *k* has just adapted to its input distribution when an update to the parameters below shifts that distribution again, so every layer is perpetually chasing a moving target. The only defences were slower learning rates and careful initialisation, and the deeper the network, the more temperamental it became.
- The prescription is to pin the distribution down: for each mini-batch, normalise each layer's inputs to zero mean and unit variance using that batch's statistics, so the distribution every layer sees stays broadly stable. The decisive choice is to make this step a layer inside the architecture rather than a preprocessing stage, so it travels with training and gradients can pass through it.
- Normalisation is immediately followed by a pair of learnable parameters, a scale *γ* and a shift *β*. Pure standardisation could weaken what a layer can express (for instance, by pinning the distribution where a saturating nonlinearity cannot be exploited), so *γ* and *β* let the network restore the original distribution when it needs to — normalisation offers a choice, not a compulsion.
- At inference time there is no "batch" to take statistics from, so the method switches to running averages of the mean and variance accumulated over training. The batch noise during training doubles as a regulariser, to the point that Dropout could be dropped in their experiments. The measured payoff is concrete: batch normalisation applied to the Inception model reached the same accuracy in 1/14 of the training steps.

**Relation to preceding and following work**

AlexNet proved that deep networks are useful; the theme of the following three years was how to make them trainable at all. Batch normalisation stabilises each layer's input distribution, and the next paper, ResNet, changes each layer's learning target; the two together pushed usable depth from the teens of layers into the hundreds. It also founded a family of normalisation methods — layer normalisation, a later descendant, became standard equipment in the Transformer.

**Original link**: https://proceedings.mlr.press/v37/ioffe15.html

---

### 7. Deep Residual Learning for Image Recognition ★

- **Authors & year**: Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, arXiv preprint, December 2015; published at the IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2016)
- **One-line significance**: Residual connections lifted network depth from tens of layers to hundreds, and "deep" ceased to be a luxury that had to be handled with extreme care.

**Abstract** (source: arXiv:1512.03385)

> Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions. We provide comprehensive empirical evidence showing that these residual networks are easier to optimize, and can gain accuracy from considerably increased depth. On the ImageNet dataset we evaluate residual nets with a depth of up to 152 layers---8x deeper than VGG nets but still having lower complexity. An ensemble of these residual nets achieves 3.57% error on the ImageNet test set. This result won the 1st place on the ILSVRC 2015 classification task. We also present analysis on CIFAR-10 with 100 and 1000 layers. The depth of representations is of central importance for many visual recognition tasks. Solely due to our extremely deep representations, we obtain a 28% relative improvement on the COCO object detection dataset. Deep residual nets are foundations of our submissions to ILSVRC & COCO 2015 competitions, where we also won the 1st places on the tasks of ImageNet detection, ImageNet localization, COCO detection, and COCO segmentation.

**Key ideas**

- The problem is first characterised correctly: stack layers directly and, past a point, the *training* error rises as the network deepens. That is not overfitting — overfitting is a different failure, in which training error falls as it should; here the training error itself goes up. It is optimisation failure, called the degradation problem: existing optimisers cannot manage even so simple a task as learning the added layers to be identity mappings.
- The fix is to change the learning target: if a stack of layers is to fit a desired mapping *H*(*x*), let it fit the residual *F*(*x*) = *H*(*x*) − *x* instead, with the final output *F*(*x*) + *x*. If the identity mapping is already optimal, one need only push *F* towards zero — far easier than having a chain of nonlinear layers happen to compose into an identity. Each layer's job changes from "reconstruct everything" to "supply a small correction."
- The implementation adds a single line: a shortcut connection carries the input *x* unchanged across several layers and adds it to their output. An identity shortcut introduces no parameters and no computation, yet it gives the gradient a direct path that does not decay — in backpropagation, the error can take the shortcut instead of being multiplied layer by layer. The idea is of one lineage with the LSTM of Volume 1, which used constant error flow against the decay of gradients over long time lags.
- Going as deep as 152 layers while staying affordable relies on the bottleneck structure: a 1×1 convolution first reduces the number of channels, a 3×3 convolution does the work in the middle, and a 1×1 convolution raises the channels again. The parameters saved mean the 152-layer network costs less computation than the 19-layer VGG. The result was a win on depth and accuracy together: first place in all five of the ILSVRC and COCO 2015 tasks entered.

**Relation to preceding and following work**

It complements the batch normalisation of the same year: the preceding paper stabilises distributions, this one changes the target, and together they made "very deep" routine. Its identity shortcut proved even longer-lived — in the Transformer of Volume 3, every sub-layer is wrapped in a shell of "residual connection plus normalisation," where the two legacies of the deep learning revolution converge and go on to support large models of a hundred layers and more.

**Original link**: https://arxiv.org/pdf/1512.03385

---

### 8. Mastering the Game of Go with Deep Neural Networks and Tree Search

- **Authors & year**: David Silver, Aja Huang, Chris J. Maddison, et al. (the Google DeepMind team; 20 authors in total), 2016, *Nature*, Vol. 529, No. 7587, pp. 484–489
- **One-line significance**: The AlphaGo paper — deep learning plus tree search defeated a human professional player at full-size Go for the first time, something generally reckoned to be at least a decade away, a decade early.

**Abstract** (source: Nature, publisher's page for the article)

> The game of Go has long been viewed as the most challenging of classic games for artificial intelligence owing to its enormous search space and the difficulty of evaluating board positions and moves. Here we introduce a new approach to computer Go that uses 'value networks' to evaluate board positions and 'policy networks' to select moves. These deep neural networks are trained by a novel combination of supervised learning from human expert games, and reinforcement learning from games of self-play. Without any lookahead search, the neural networks play Go at the level of state-of-the-art Monte Carlo tree search programs that simulate thousands of random games of self-play. We also introduce a new search algorithm that combines Monte Carlo simulation with value and policy networks. Using this search algorithm, our program AlphaGo achieved a 99.8% winning rate against other Go programs, and defeated the human European Go champion by 5 games to 0. This is the first time that a computer program has defeated a human professional player in the full-sized game of Go, a feat previously thought to be at least a decade away.

**Key ideas**

- Go is hard in two places: the branching is enormous (roughly 250 legal moves per turn and games of 150 moves or more, so exhaustive search is out of the question), and positions are inscrutable — midway through a game, who stands better is something strong players judge by intuition and beginners cannot approach at all. Earlier programs propped themselves up with Monte Carlo tree search plus hand-crafted features and stalled at strong-amateur level. This paper hands the two difficulties to two networks: a policy network answers "which moves are worth considering," and a value network answers "how good is the current position" — learned intuition in place of exhaustive search and hand-built evaluation.
- Training is a two-stage relay. First, supervised learning: a policy network is trained on 30 million positions from human expert games and learns to play as humans do (57% accuracy at predicting human moves). Then reinforcement learning: the current policy plays against earlier versions of itself and is adjusted by wins and losses, rising above the ceiling of the human game records. The value network is trained on positions from self-play together with the games' final outcomes, and estimates the whole game in a single number.
- Search is not discarded but married to the networks: play still uses Monte Carlo tree search, but the policy network concentrates the search on a few promising branches (cutting the width), while the value network, together with random rollouts from a fast rollout policy, evaluates leaf positions (supplying the depth). Taken separately, the two networks already matched the strongest traditional programs; combined with search, they broke through the professional ceiling.
- The methodological signal is larger than the game itself: supervised learning as the foundation, reinforcement learning for self-improvement, and assembly with a classical search algorithm — "learned intuition plus computable search" became the template for a whole class of later systems, from the sequel AlphaGo Zero (2017, which dispensed even with human game records) to problems in robotics and scientific computing.

**Relation to preceding and following work**

Most of the first seven papers in this volume manufacture components — convolution, sequence modelling, attention, generation, normalisation, residuals. AlphaGo is the assembly of components into a system, and it is also reinforcement learning's foothold in this lineage. It reads the board with this volume's convolutional networks and tunes them to superhuman strength with the two training regimes of supervised and reinforcement learning; the sensation it caused then paved the way for the flood of investment and talent into AI after 2017, with which the story of Volume 3 opens.

**Original link**: https://doi.org/10.1038/nature16961

---

*Verification note for this volume: the bibliographic details and abstracts of all eight papers were checked against the arXiv abstract pages or the paper PDFs (Seq2Seq, Bahdanau et al., GAN, VAE, ResNet), the original NeurIPS 2012 paper (AlexNet, which has no arXiv preprint), the ICML proceedings as published in PMLR (Batch Normalization), and the abstract on the Nature website (AlphaGo, cross-checked against the journal's bibliographic record and independent transcriptions). Two version differences are noted: the arXiv preprint of Batch Normalization ends its abstract with the figures "4.9% top-5 validation error (and 4.8% test error)," whereas this volume follows the ICML published version, which gives a 4.82% top-5 test error; and the Seq2Seq abstract is taken from the current arXiv version (v3), which differs from the NeurIPS published version only in a few words.*
