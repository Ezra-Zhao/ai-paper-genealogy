# AI Landmark Papers: From Foundations to the Frontier — Vol. 3 The Transformer and the Large Language Models (2017–2022)

English edition; abstracts are quoted verbatim from the original papers where available.

This volume collects thirteen papers. In 2017 the Transformer appeared, and for the five years that followed the main thread was a single story: scale this architecture up, pre-train it, then align it. From the fork between the BERT and GPT routes, to the scaling laws that supplied the engineering blueprint, to RLHF, which taught "large" to become "obedient." The diffusion side branch also joins the main line in this volume and is finally stitched together with the attention mechanism. Two more branches join in this volume — retrieval augmentation and agent-style acting: one bolts an external memory onto the model, the other carries it from answering questions to getting things done. Each entry gives the bibliographic record, the abstract, the key ideas, and the paper's place in the lineage.

---

### 1. Attention Is All You Need ★

- **Authors & year**: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin (Google), 2017, 31st Conference on Neural Information Processing Systems (NeurIPS 2017)
- **One-line significance**: The foundation of the following decade of deep learning — it discarded recurrence and convolution and built a sequence model out of attention alone; the other twelve papers in this volume all stand on it.

**Abstract** (source: arXiv:1706.03762, current version)

> The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.

**Key ideas**

- Self-attention is the only mixing mechanism: every position in the sequence attends directly to every other position, weighting and aggregating information by relevance, so the path length between any two positions is constant rather than a relay passed step by step as in a recurrent network. The old problem from Volume 2 — that long-range dependencies are hard to learn — is dismantled structurally.
- The price is that attention itself knows nothing of word order, so positional encodings inject order information into the inputs; multi-head attention lets the model compute several sets of relevance weights simultaneously in different subspaces, each capturing a different kind of relation.
- Parallelisability is the real killer feature: a recurrent network must compute serially, time step by time step, whereas the Transformer can lay the whole sequence out at once and feed it to the GPU. That models could later be scaled to hundreds of billions of parameters has this step as its precondition — the architecture's temperament finally matched the hardware.
- The skeleton is still the encoder–decoder of Volume 2: the encoder reads the source sentence into a sequence of context-bearing vectors, and the decoder, equipped with masked self-attention, generates word by word while looking back at the encoder output through cross-attention. Bahdanau's alignment weights are promoted here from an attachment to the main structure.

**Relation to preceding and following work**

It promoted Bahdanau's attention in Volume 2 from a patch on a recurrent network to the whole architecture; the Seq2Seq encoder–decoder framework was retained, and the LSTM was shown the door. Afterwards BERT took its encoder and GPT its decoder — both main lines of this volume fork from this one paper.

**Original link**: https://arxiv.org/abs/1706.03762?ref=mlcontests

---

### 2. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding ★

- **Authors & year**: Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova (Google AI Language), 2019, NAACL-HLT 2019 (arXiv preprint October 2018)
- **One-line significance**: The paper that established the pre-train–fine-tune paradigm — first learn language itself from unlabelled text, then attach a small output layer for each task.

**Abstract** (source: arXiv:1810.04805, revised version, v2)

> We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications.
>
> BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE score to 80.5% (7.7% point absolute improvement), MultiNLI accuracy to 86.7% (4.6% absolute improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point absolute improvement) and SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute improvement).

**Key ideas**

- "Bidirectional" is the crux of the whole paper. Earlier language models could only read left to right, so a word's representation could not see the words to its right; BERT conditions on both left and right context in every layer, so a word's meaning is determined by the whole sentence rather than by its first half alone.
- Bidirectionality is achieved through the masked language model (MLM): about 15% of the input tokens are masked at random, and the model must guess them back from context. The masking manoeuvre slips past a deadlock — if a bidirectional model saw the whole sentence directly, predicting a target word would amount to peeking at the answer; removing the target word from the input leaves it no choice but to genuinely understand the context.
- The other pre-training task is next sentence prediction (NSP): judging whether two sentences really follow one another, which lays the groundwork for "sentence-pair" tasks such as question answering and inference. The pre-training corpus is BooksCorpus together with English Wikipedia, some 3.3 billion words in total.
- In use it parts company with the ELMo style of "feature extraction": BERT does not treat representations as fixed features to be fed into each task's own architecture, but fine-tunes the entire model, updating all parameters together with the task. That eleven tasks were refreshed at once rests on exactly this "one base, fine-tuned everywhere" approach.

**Relation to preceding and following work**

It and the next paper, GPT-2, are two ways of living for the Transformer in the same year: BERT takes the encoder and pursues bidirectional understanding; GPT takes the decoder and pursues unidirectional generation. Over the next two years almost every NLP leaderboard was rewritten by these two routes in turn, and "pre-train first" became the default starting point rather than an option.

**Original link**: https://arxiv.org/html/1810.04805v2

---

### 3. Language Models are Unsupervised Multitask Learners

- **Authors & year**: Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever (OpenAI), 2019, OpenAI technical report
- **One-line significance**: The first serious demonstration that a sufficiently large language model, with no task-specific training at all, can get to work zero-shot from a prompt alone.

**Abstract** (source: OpenAI technical report, original text; the paper has no arXiv preprint)

> Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on task-specific datasets. We demonstrate that language models begin to learn these tasks without any explicit supervision when trained on a new dataset of millions of webpages called WebText. When conditioned on a document plus questions, the answers generated by the language model reach 55 F1 on the CoQA dataset - matching or exceeding the performance of 3 out of 4 baseline systems without using the 127,000+ training examples. The capacity of the language model is essential to the success of zero-shot task transfer and increasing it improves performance in a log-linear fashion across tasks. Our largest model, GPT-2, is a 1.5B parameter Transformer that achieves state of the art results on 7 out of 8 tested language modeling datasets in a zero-shot transfer setting but still underfits WebText. Samples from the model reflect these improvements and contain coherent paragraphs of text. These findings suggest a promising path towards building language processing systems which learn to perform tasks from their naturally occurring demonstrations.

**Key ideas**

- The stance is written into the title: tasks need not be conquered one by one with supervised learning. Language naturally contains demonstrations of tasks — articles hold questions and answers, parallel translations, summaries alongside their source texts — and in learning to "predict the next word" the model picks these tasks up along the way. That is what "unsupervised multitask learner" means.
- Data is the key variable. WebText was collected by an ingenious shortcut: scrape the outbound links posted on Reddit that had earned at least 3 karma, using the crowd's endorsement as a filter, yielding some 8 million documents and 40GB of text. This "human screening in place of manual annotation" became the prototype for later data pipelines everywhere.
- The evaluation discipline is zero-shot throughout: no training examples for the task, no fine-tuning — the task is simply written as natural text and fed in, and the model's continuation is observed. The CoQA reading-comprehension score of 55 F1 and new records on 7 of 8 language modelling datasets were all obtained in this "bare examination."
- The paper also made a famous off-stage decision: citing misuse concerns, OpenAI released the weights in stages, and the largest, 1.5B-parameter version did not appear until November 2019. Whatever the sincerity of that caution, it opened the long-running argument over whether large models should be open-sourced.

**Relation to preceding and following work**

It inherits the same team's 2018 GPT-1 recipe of "generative pre-training plus fine-tuning," but abolishes the fine-tuning step itself, pushing the question forward: if zero-shot already works, what happens when the model is scaled up by another order of magnitude? The next paper, GPT-3, is the head-on answer to that question.

**Original link**: https://www.semanticscholar.org/paper/Language-Models-are-Unsupervised-Multitask-Learners-Radford-Wu/9405cc0d6169988371b2755e573cc28650d14dfe

---

### 4. Language Models are Few-Shot Learners ★

- **Authors & year**: Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, et al., 31 authors in total (OpenAI), 2020, 34th Conference on Neural Information Processing Systems (NeurIPS 2020)
- **One-line significance**: A demonstration of scale's tipping point — at 175 billion parameters, "show it a few examples and it can do the task" turned from a sporadic phenomenon into a stable ability, and the era of large models formally opened.

**Abstract** (source: arXiv:2005.14165)

> Recent work has demonstrated substantial gains on many NLP tasks and benchmarks by pre-training on a large corpus of text followed by fine-tuning on a specific task. While typically task-agnostic in architecture, this method still requires task-specific fine-tuning datasets of thousands or tens of thousands of examples. By contrast, humans can generally perform a new language task from only a few examples or from simple instructions - something which current NLP systems still largely struggle to do. Here we show that scaling up language models greatly improves task-agnostic, few-shot performance, sometimes even reaching competitiveness with prior state-of-the-art fine-tuning approaches. Specifically, we train GPT-3, an autoregressive language model with 175 billion parameters, 10x more than any previous non-sparse language model, and test its performance in the few-shot setting. For all tasks, GPT-3 is applied without any gradient updates or fine-tuning, with tasks and few-shot demonstrations specified purely via text interaction with the model. GPT-3 achieves strong performance on many NLP datasets, including translation, question-answering, and cloze tasks, as well as several tasks that require on-the-fly reasoning or domain adaptation, such as unscrambling words, using a novel word in a sentence, or performing 3-digit arithmetic. At the same time, we also identify some datasets where GPT-3's few-shot learning still struggles, as well as some datasets where GPT-3 faces methodological issues related to training on large web corpora. Finally, we find that GPT-3 can generate samples of news articles which human evaluators have difficulty distinguishing from articles written by humans. We discuss broader societal impacts of this finding and of GPT-3 in general.

**Key ideas**

- Methodologically there is no new structure: it is still a Transformer decoder, still predicting the next word. The only action is scaling — over 100 times the parameters of GPT-2, trained on roughly 300 billion tokens. The paper's wager is that quantity itself produces a qualitative change.
- The form in which the wager pays off is called in-context learning: no parameters are updated; the input simply contains a task description and a few examples (few-shot), and the model grasps the task on the spot and follows suit. As the number of examples rises from zero to a few to a few dozen, performance climbs with it — learning happens in the inference-time context, not in the weights.
- The name the authors give this mechanism is meta-learning: having seen sufficiently varied textual structures during pre-training, the model has internally formed the general knack of "recognising the task from the context." The prompt thereby changes from ordinary input text into a kind of programming interface — the origin of the later prompt engineering.
- The paper is also unusually candid about its weaknesses: few-shot learning still struggles on some datasets, large-scale web corpora bring data contamination and bias, and the news articles the model generates can already fool human evaluators. Ability and risk on the same page became the convention for large-model technical reports thereafter.

**Relation to preceding and following work**

It directly cashes in the question GPT-2 left behind, and pushes the "pre-train plus fine-tune" paradigm one step further: even fine-tuning can be dispensed with. The scaling laws that follow immediately (No. 5 in this volume, with much authorial overlap) then attempt to turn this paper's demonstration that "bigger is stronger" from a phenomenon into a computable formula.

**Original link**: https://arxiv.org/abs/2005.14165v2

---

### 5. Scaling Laws for Neural Language Models

- **Authors & year**: Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei (OpenAI and Johns Hopkins University), 2020, arXiv preprint
- **One-line significance**: The engineering blueprint for "scaling up" — before burning an enormous compute budget, use small experiments to calculate how far the loss will fall.

**Abstract** (source: arXiv:2001.08361)

> We study empirical scaling laws for language model performance on the cross-entropy loss. The loss scales as a power-law with model size, dataset size, and the amount of compute used for training, with some trends spanning more than seven orders of magnitude. Other architectural details such as network width or depth have minimal effects within a wide range. Simple equations govern the dependence of overfitting on model/dataset size and the dependence of training speed on model size. These relationships allow us to determine the optimal allocation of a fixed compute budget. Larger models are significantly more sample-efficient, such that optimally compute-efficient training involves training very large models on a relatively modest amount of data and stopping significantly before convergence.

**Key ideas**

- The core finding is a set of power laws: cross-entropy loss falls smoothly as a power law in parameter count, data volume, and compute respectively, with trends that do not bend across more than seven orders of magnitude. This means a large model's final performance is broadly predictable before training begins — deep learning acquired, for the first time, a tool akin to engineering extrapolation.
- Scale overwhelms shape: over a wide range, the proportions of depth and width (a little deeper or a little wider) have minimal effect on the final loss. Researchers' attention accordingly shifted from "tuning the structure" to "allocating the budget," and the architecture race visibly cooled after this paper.
- The power laws can be run backwards to allocate a budget: for fixed compute there is an optimal pairing of parameter count and data volume; and because larger models are significantly more sample-efficient, the compute-optimal play is to make the model very large, train it on a relatively modest amount of data, and stop well before convergence — a conclusion that directly dictated GPT-3's configuration.
- Overfitting has a formula too: the ratio of parameters to data determines the overfitting penalty, and the two must be scaled together in a fixed proportion (parameters up 8-fold, data up roughly 5-fold). "Model and data must grow together" was later corrected in its proportions by Chinchilla, but the framework came from this paper.

**Relation to preceding and following work**

It turns the preceding paper's intuition into formulas; the two share much of the same team, one supplying the phenomenon and the other the law. But its allocation conclusion — "feed the parameters more, the data less" — rested on a particular way of fitting, and two years later it was overturned by the more complete experiments of No. 8 in this volume, Chinchilla: a rare scene in this lineage of a law being corrected by a law.

**Original link**: https://arxiv.org/html/2001.08361v1

---

### 6. Denoising Diffusion Probabilistic Models

- **Authors & year**: Jonathan Ho, Ajay Jain, and Pieter Abbeel (University of California, Berkeley), 2020, 34th Conference on Neural Information Processing Systems (NeurIPS 2020)
- **One-line significance**: The paper that put diffusion models on the march — generation became "learning to denoise," and the throne of image generation changed hands within two years.

**Abstract** (source: arXiv:2006.11239)

> We present high quality image synthesis results using diffusion probabilistic models, a class of latent variable models inspired by considerations from nonequilibrium thermodynamics. Our best results are obtained by training on a weighted variational bound designed according to a novel connection between diffusion probabilistic models and denoising score matching with Langevin dynamics, and our models naturally admit a progressive lossy decompression scheme that can be interpreted as a generalization of autoregressive decoding. On the unconditional CIFAR10 dataset, we obtain an Inception score of 9.46 and a state-of-the-art FID score of 3.17. On 256x256 LSUN, we obtain sample quality similar to ProgressiveGAN. Our implementation is available at https://github.com/hojonathanho/diffusion.

**Key ideas**

- Generation is broken into a thousand small steps: first define a forward process that adds Gaussian noise to an image little by little, on a fixed schedule, until it becomes pure noise; what the model must learn is the reverse process — starting from pure noise, removing the noise step by step until an image is restored. Each step removes only a little noise, so a hard-to-learn distributional transformation is spread across many easy small transformations.
- The training objective is strikingly plain: show the model an image corrupted with a known amount of noise and have it predict "what noise was just added," by mean-squared-error regression. The paper proves this objective is a weighted version of a variational bound; after all the theory, what lands in practice is a single regression loss.
- Because every step is controllable, the sampling process can be viewed as a kind of progressive decompression, which the paper describes as a generalisation of autoregressive decoding. And because it proceeds step by step, sample diversity is preserved, unlike GANs, which readily generate only a few kinds of "safe sample" — mode collapse.
- The cost is stated plainly: generating one image requires running the network sequentially a thousand times, which is slow. This paper does not solve speed; it solves quality and training stability — no adversarial game, no delicate balancing act, training is stable and generation is good.

**Relation to preceding and following work**

It takes over from the generative line of Volume 2 (GAN, VAE): GANs rely on an adversarial game and are hard to train; diffusion switches to "a thousand denoising regressions," pulling generative-model training back onto the old road of supervised learning. It looks unrelated to this volume's language main line — until No. 13, latent diffusion, moves the mechanism into a latent space and attaches text conditioning, and the great explosion of image generation truly arrives.

**Original link**: https://arxiv.org/pdf/2006.11239

---

### 7. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks

- **Authors & year**: Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela (Facebook AI, University College London, and New York University), 2020, 34th Conference on Neural Information Processing Systems (NeurIPS 2020)
- **One-line significance**: The founding paper of external memory for large models — the parameters handle language and reasoning while the facts live in a retrievable index, and "look it up before you write" became the default assembly for putting models to work.

**Abstract** (source: arXiv:2005.11401, current version, v4)

> Large pre-trained language models have been shown to store factual knowledge in their parameters, and achieve state-of-the-art results when fine-tuned on downstream NLP tasks. However, their ability to access and precisely manipulate knowledge is still limited, and hence on knowledge-intensive tasks, their performance lags behind task-specific architectures. Additionally, providing provenance for their decisions and updating their world knowledge remain open research problems. Pre-trained models with a differentiable access mechanism to explicit non-parametric memory can overcome this issue, but have so far been only investigated for extractive downstream tasks. We explore a general-purpose fine-tuning recipe for retrieval-augmented generation (RAG) -- models which combine pre-trained parametric and non-parametric memory for language generation. We introduce RAG models where the parametric memory is a pre-trained seq2seq model and the non-parametric memory is a dense vector index of Wikipedia, accessed with a pre-trained neural retriever. We compare two RAG formulations, one which conditions on the same retrieved passages across the whole generated sequence, the other can use different passages per token. We fine-tune and evaluate our models on a wide range of knowledge-intensive NLP tasks and set the state-of-the-art on three open domain QA tasks, outperforming parametric seq2seq models and task-specific retrieve-and-extract architectures. For language generation tasks, we find that RAG models generate more specific, diverse and factual language than a state-of-the-art parametric-only seq2seq baseline.

**Key ideas**

- The central claim splits "memory" in two: parametric memory is what sits in the model's weights and can be changed only by retraining; non-parametric memory is an external text index, changed by swapping documents. Language ability belongs to the parameters, factual knowledge to the index — and that split is why RAG has endured: stale knowledge, provenance, and domain changes are all index-level matters that leave the model untouched.
- The assembly: a dense retriever (the paper uses DPR, published the same year) turns Wikipedia into a vector index of some 21 million passages; given a question, the most relevant passages are retrieved and fed, together with the question, to a generator (the paper uses BART), which writes its answer with the material in front of it. The retrieval step is differentiable, so retriever and generator can be fine-tuned end to end.
- The two formulations serve two uses: RAG-Sequence conditions on the same set of passages throughout the whole answer, suited to answers drawn from a single source; RAG-Token lets each token draw on different passages, suited to stitching several sources into one sentence.
- It also answers "why do models make things up": knowledge inside parameters can be neither retrieved precisely nor cited; placing the facts in front of the generator compresses the space for fabrication, and answers can point back to specific passages — provenance became a proper requirement of generation systems from this paper on.

**Relation to preceding and following work**

It and GPT-3, from the same year, are two answers to the question of where knowledge should live: GPT-3 presses knowledge into the parameters and internalises it by scale; RAG externalises it, leaving the parameters to concentrate on language and reasoning. Its skeleton is still the seq2seq of Volume 2, with a retrieval step placed in front of it. The line it opened afterwards is the larger one: after 2022, "connect the model to a knowledge base" became the default move in deployment — enterprise knowledge question answering, customer support, and private-document assistants are all descendants of this line, and the RAG section of this repository's application track leads back to this entry.

**Original link**: https://arxiv.org/abs/2005.11401

---

### 8. Training Compute-Optimal Large Language Models

- **Authors & year**: Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, et al. (DeepMind), 2022, 36th Conference on Neural Information Processing Systems (NeurIPS 2022)
- **One-line significance**: A measured overturning of the scaling laws — the point is not that bigger is always better, but that every large model of the day was "underfed."

**Abstract** (source: arXiv:2203.15556)

> We investigate the optimal model size and number of tokens for training a transformer language model under a given compute budget. We find that current large language models are significantly undertrained, a consequence of the recent focus on scaling language models whilst keeping the amount of training data constant. By training over 400 language models ranging from 70 million to over 16 billion parameters on 5 to 500 billion tokens, we find that for compute-optimal training, the model size and the number of training tokens should be scaled equally: for every doubling of model size the number of training tokens should also be doubled. We test this hypothesis by training a predicted compute-optimal model, Chinchilla, that uses the same compute budget as Gopher but with 70B parameters and 4× more more data. Chinchilla uniformly and significantly outperforms Gopher (280B), GPT-3 (175B), Jurassic-1 (178B), and Megatron-Turing NLG (530B) on a large range of downstream evaluation tasks. This also means that Chinchilla uses substantially less compute for fine-tuning and inference, greatly facilitating downstream usage. As a highlight, Chinchilla reaches a state-of-the-art average accuracy of 67.5% on the MMLU benchmark, greater than a 7% improvement over Gopher.

**Key ideas**

- There is only one question: with a fixed compute budget, how should parameter count and training tokens be divided? They trained over 400 models and cross-fitted with three methods, reaching a consistent answer: the two should be scaled in equal proportion. Converted into a rule of thumb, that is roughly 20 training tokens per parameter.
- Judged by this standard, the whole industry had the allocation wrong: GPT-3 trained 175 billion parameters on roughly 300 billion tokens, whereas the new law says it should have been fed some 3.5 trillion tokens to count as fully trained. The race to stack parameters while skimping on data was partly misled by the preceding scaling-laws paper's fitting method — Chinchilla points out that it held details such as the learning rate schedule fixed, biasing the conclusion towards large models.
- The verification is blunt and clean: train a model to the new law — 70 billion parameters, 1.4 trillion tokens, named Chinchilla — on the same compute as the 280-billion-parameter Gopher, and it wins across downstream tasks, beating opponents several times its size as well.
- Small but fully trained also carries a long-term dividend: inference and fine-tuning are both cheaper. Afterwards every lab (the LLaMA family most typically) would rather train models on data far beyond the Chinchilla ratio than give up deployment cost — the "train once, use many times" accounting rewrote the optimum of the ratio.

**Relation to preceding and following work**

It and No. 5 form a pair within the lineage: the same scaling-law story, corrected two years later by finer experiments. It is also the GPT-3 route correcting itself — proving that "scaling up" does not mean maxing out the parameter dial alone, but growing parameters and data together in proportion.

**Original link**: https://arxiv.org/html/2203.15556v1

---

### 9. LoRA: Low-Rank Adaptation of Large Language Models

- **Authors & year**: Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen (Microsoft), arXiv preprint June 2021, 10th International Conference on Learning Representations (ICLR 2022)
- **One-line significance**: The de facto standard for parameter-efficient fine-tuning — freeze the large model, train only a pair of small matrices, and the cost of customising a hundred-billion-parameter model drops by four orders of magnitude.

**Abstract** (source: arXiv:2106.09685, v2)

> An important paradigm of natural language processing consists of large-scale pre-training on general domain data and adaptation to particular tasks or domains. As we pre-train larger models, full fine-tuning, which retrains all model parameters, becomes less feasible. Using GPT-3 175B as an example -- deploying independent instances of fine-tuned models, each with 175B parameters, is prohibitively expensive. We propose Low-Rank Adaptation, or LoRA, which freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks. Compared to GPT-3 175B fine-tuned with Adam, LoRA can reduce the number of trainable parameters by 10,000 times and the GPU memory requirement by 3 times. LoRA performs on-par or better than fine-tuning in model quality on RoBERTa, DeBERTa, GPT-2, and GPT-3, despite having fewer trainable parameters, a higher training throughput, and, unlike adapters, no additional inference latency. We also provide an empirical investigation into rank-deficiency in language model adaptation, which sheds light on the efficacy of LoRA. We release a package that facilitates the integration of LoRA with PyTorch models and provide our implementations and model checkpoints for RoBERTa, DeBERTa, and GPT-2 at https://github.com/microsoft/LoRA.

**Key ideas**

- The starting point is an empirical observation: when a model adapts to a new task, the weight update itself is "low-rank" — the degrees of freedom that actually matter are far fewer than the matrix's dimensions. So instead of learning the full update, learn only its low-rank factorisation: two thin matrices multiplied together, with a rank of 1 to 8 sufficing.
- During training the original weights are all frozen and only this pair of small matrices is updated; at inference the learned increment is added back into the original weights and merged into one, introducing no additional latency at all — the point that distinguishes it from adapter methods, which insert extra modules into the inference path.
- The numerical comparison makes the case best: on GPT-3 175B, trainable parameters fall to on the order of one ten-thousandth of full fine-tuning, GPU memory requirements to one third, with quality matching or better. One base model plus countless LoRA weight files of a few dozen megabytes changed the shape of deployment.
- It also answers "how large a rank, and where": experiments find the greatest benefit from placing LoRA on the attention query and value projections, and that beyond a single-digit rank the marginal gains vanish quickly — indirect confirmation of the hypothesis that "adaptation is a low-dimensional event."

**Relation to preceding and following work**

It is the life-extension of the BERT-era "fine-tuning" tradition: when models grew too large for full fine-tuning to be affordable, fine-tuning survived in this form. It is also a close relative of the next paper, InstructGPT — RLHF's policy-model fine-tuning later widely borrowed LoRA to cut costs, and "frozen base plus lightweight adaptation" became the default assembly for putting large models to work.

**Original link**: https://arxiv.org/html/2106.09685v2

---

### 10. Training Language Models to Follow Instructions with Human Feedback ★

- **Authors & year**: Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, et al. (OpenAI), 2022, 36th Conference on Neural Information Processing Systems (NeurIPS 2022)
- **One-line significance**: The birth certificate of RLHF — after tuning by human feedback, a 1.3-billion-parameter model was preferred by humans over a GPT-3 a hundred times its size.

**Abstract** (source: arXiv:2203.02155)

> Making language models bigger does not inherently make them better at following a user's intent. For example, large language models can generate outputs that are untruthful, toxic, or simply not helpful to the user. In other words, these models are not aligned with their users. In this paper, we show an avenue for aligning language models with user intent on a wide range of tasks by fine-tuning with human feedback. Starting with a set of labeler-written prompts and prompts submitted through the OpenAI API, we collect a dataset of labeler demonstrations of the desired model behavior, which we use to fine-tune GPT-3 using supervised learning. We then collect a dataset of rankings of model outputs, which we use to further fine-tune this supervised model using reinforcement learning from human feedback. We call the resulting models InstructGPT. In human evaluations on our prompt distribution, outputs from the 1.3B parameter InstructGPT model are preferred to outputs from the 175B GPT-3, despite having 100x fewer parameters. Moreover, InstructGPT models show improvements in truthfulness and reductions in toxic output generation while having minimal performance regressions on public NLP datasets. Even though InstructGPT still makes simple mistakes, our results show that fine-tuning with human feedback is a promising direction for aligning language models with human intent.

**Key ideas**

- The paper first states the disease plainly: a language model's training objective is "predict the next word," whereas the user's intent is "be helpful, tell the truth, do no harm" — the two are not the same thing, and that is the alignment problem. Making the model larger does not make the mismatch disappear on its own.
- The prescription has three steps, later known as the RLHF pipeline. Step one, supervised fine-tuning (SFT): labelers hand-write ideal responses, and GPT-3 is fine-tuned on these demonstrations. Step two, train a reward model (RM): the model generates several responses to the same prompt, labelers rank them, and the model learns to predict which response humans would score highly. Step three, reinforcement learning: with the reward model's scores as the objective, the policy model is further optimised with the PPO algorithm, under a constraint that stops it from drifting into gibberish that games the score.
- The data all come from real settings: the prompts were partly written by labelers to specification and partly drawn (after de-identification) from real requests submitted by users of the OpenAI API. This means InstructGPT was aligned to the "real user distribution," not to some academic benchmark.
- The most glaring result is the parameter comparison: the 1.3B-parameter InstructGPT beat the 175B-parameter GPT-3 in blind human evaluation. The paper in effect announced that model competition from then on would not be only about pre-training scale — a whole discipline of alignment work remained to be done. And this three-step pipeline is the direct predecessor of ChatGPT at the end of 2022, which was the same pipeline fitted with GPT-3.5.

**Relation to preceding and following work**

It stands on GPT-3's shoulders yet rewrites the rules of the game after GPT-3: capability comes from pre-training, usability from alignment. It complements No. 9, LoRA — one solves "how to make the model obedient," the other "how to change the model cheaply"; and with No. 11, chain-of-thought, it represents the two routes by which model performance was thereafter improved: one changes the training, the other changes the prompt.

**Original link**: https://arxiv.org/html/2203.02155v1

---

### 11. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models

- **Authors & year**: Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou (Google Research), 2022, 36th Conference on Neural Information Processing Systems (NeurIPS 2022)
- **One-line significance**: The watershed of prompt engineering — without changing a single parameter, merely writing the "working" into the exemplars immediately doubled large models' reasoning scores.

**Abstract** (source: arXiv:2201.11903, v6)

> We explore how generating a chain of thought -- a series of intermediate reasoning steps -- significantly improves the ability of large language models to perform complex reasoning. In particular, we show how such reasoning abilities emerge naturally in sufficiently large language models via a simple method called chain of thought prompting, where a few chain of thought demonstrations are provided as exemplars in prompting. Experiments on three large language models show that chain of thought prompting improves performance on a range of arithmetic, commonsense, and symbolic reasoning tasks. The empirical gains can be striking. For instance, prompting a 540B-parameter language model with just eight chain of thought exemplars achieves state of the art accuracy on the GSM8K benchmark of math word problems, surpassing even finetuned GPT-3 with a verifier.

**Key ideas**

- The method is simple to the point of not looking like a paper: in few-shot prompts, the exemplars are no longer "question to answer" but "question, a series of intermediate reasoning steps, answer." Led by the exemplars, the model first works step by step and only then concludes; that series of intermediate steps is the chain of thought.
- The paper's explanation of why it works has three layers: the model can spread a multi-step computation over more tokens, in effect trading generation length for computation; the reasoning steps decompose a complex problem into small problems it has seen before; and the text itself is an inspectable work record — a human can see at which step it went wrong.
- One threshold must be stated: the benefits of chain-of-thought appear only in sufficiently large models (on the order of a hundred billion parameters); small models' reasoning chains are often logically incoherent and actually lower their scores. The authors file it under emergent abilities — another instance of "only there at scale," of the same lineage as GPT-3's in-context learning.
- The flagship number is the 540B-parameter PaLM with eight chain-of-thought exemplars setting a new record on GSM8K, surpassing a GPT-3 fine-tuned with a verifier. Afterwards "let the model think before it answers" became the default move, and successors such as self-consistency and tree-of-thoughts all grew out of this paper.

**Relation to preceding and following work**

It and GPT-3 are two faces of the same discovery: GPT-3 proved exemplars can teach a model "what to do," and this paper proved exemplars can also teach it "how to do it." Nor does it conflict with InstructGPT — one improves performance from the prompt side, the other from the training side, and later models use both: first RLHF alignment, then chain-of-thought problem-solving.

**Original link**: https://arxiv.org/html/2201.11903v6

---

### 12. ReAct: Synergizing Reasoning and Acting in Language Models

- **Authors & year**: Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao (Princeton University and Google Research), arXiv preprint October 2022, 11th International Conference on Learning Representations (ICLR 2023)
- **One-line significance**: The birth certificate of the agent — by interleaving "thinking" and "doing," with reasoning setting the plan and actions fetching the information, the language model turns from an answerer into a doer.

**Abstract** (source: arXiv:2210.03629, current version, v3)

> While large language models (LLMs) have demonstrated impressive capabilities across tasks in language understanding and interactive decision making, their abilities for reasoning (e.g. chain-of-thought prompting) and acting (e.g. action plan generation) have primarily been studied as separate topics. In this paper, we explore the use of LLMs to generate both reasoning traces and task-specific actions in an interleaved manner, allowing for greater synergy between the two: reasoning traces help the model induce, track, and update action plans as well as handle exceptions, while actions allow it to interface with external sources, such as knowledge bases or environments, to gather additional information. We apply our approach, named ReAct, to a diverse set of language and decision making tasks and demonstrate its effectiveness over state-of-the-art baselines, as well as improved human interpretability and trustworthiness over methods without reasoning or acting components. Concretely, on question answering (HotpotQA) and fact verification (Fever), ReAct overcomes issues of hallucination and error propagation prevalent in chain-of-thought reasoning by interacting with a simple Wikipedia API, and generates human-like task-solving trajectories that are more interpretable than baselines without reasoning traces. On two interactive decision making benchmarks (ALFWorld and WebShop), ReAct outperforms imitation and reinforcement learning methods by an absolute success rate of 34% and 10% respectively, while being prompted with only one or two in-context examples.

**Key ideas**

- The mechanism is a single loop: think a step (thought), act a step (action), look at the result (observation), then think again. The reasoning trace makes and revises the plan; the actions reach into the outside world — querying a knowledge base, operating an environment — and bring new information back. Neither half works alone: pure reasoning argues in a closed room, pure acting gropes blindly.
- It is the direct sequel to No. 11, chain-of-thought, and it treats CoT's disease: chain-of-thought locks reasoning inside the model, so one misremembered fact poisons every later step with no way to correct it. ReAct lets the model stop mid-reasoning and query a Wikipedia API, using external facts to straighten the chain — the first mechanistic remedy for hallucination and error propagation.
- The trajectory is itself the documentation: the alternation of thought and action is written out in the open, so a human can see what information each decision rested on and why that action was taken. The paper makes interpretability and diagnosability formal comparison points against the baselines — a large part of why the engineering world adopted the pattern.
- And it is still only prompting: on ALFWorld and WebShop, one or two in-context examples and no training at all beat imitation and reinforcement learning methods trained on tens of thousands of task instances. "Think–act–observe" is not a trick for one task but a general interface — the execution loop of nearly every agent framework since has this shape.

**Relation to preceding and following work**

It joins two lines of this volume: No. 11, chain-of-thought, supplies the "thinking," and the external retrieval represented by No. 7, RAG, supplies the "fetching of information" — ReAct lets the model itself decide, inside the reasoning loop, when to fetch. It also continues No. 10, InstructGPT: only a model first trained to follow instructions can be expected to emit actions in the required format and walk the loop obediently. From here on, "language model + tools + loop" is the standard definition of an agent, and this repository's application track (Agent in Practice) is built on this paper.

**Original link**: https://arxiv.org/abs/2210.03629

---

### 13. High-Resolution Image Synthesis with Latent Diffusion Models

- **Authors & year**: Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer (Ludwig Maximilian University of Munich and Runway), 2022, IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2022)
- **One-line significance**: The paper behind Stable Diffusion itself — moving diffusion from pixel space into latent space brought the compute threshold for generating high-resolution images down to consumer-grade graphics cards.

**Abstract** (source: CVPR 2022 published version, identical to arXiv:2112.10752v2, PDF text)

> By decomposing the image formation process into a sequential application of denoising autoencoders, diffusion models (DMs) achieve state-of-the-art synthesis results on image data and beyond. Additionally, their formulation allows for a guiding mechanism to control the image generation process without retraining. However, since these models typically operate directly in pixel space, optimization of powerful DMs often consumes hundreds of GPU days and inference is expensive due to sequential evaluations. To enable DM training on limited computational resources while retaining their quality and flexibility, we apply them in the latent space of powerful pretrained autoencoders. In contrast to previous work, training diffusion models on such a representation allows for the first time to reach a near-optimal point between complexity reduction and detail preservation, greatly boosting visual fidelity. By introducing cross-attention layers into the model architecture, we turn diffusion models into powerful and flexible generators for general conditioning inputs such as text or bounding boxes and high-resolution synthesis becomes possible in a convolutional manner. Our latent diffusion models (LDMs) achieve new state-of-the-art scores for image inpainting and class-conditional image synthesis and highly competitive performance on various tasks, including text-to-image synthesis, unconditional image generation and super-resolution, while significantly reducing computational requirements compared to pixel-based DMs.

**Key ideas**

- The insight is a division of labour: most of the compute in an image is eaten by pixel-level detail, while composition and content at the semantic level do not need to be processed at full resolution. So an autoencoder is trained first to compress the image into a low-dimensional latent space (spatial size reduced by a factor of roughly 4 to 8), the entire diffusion process moves into the latent space, and only at the end is the result decoded back to pixels.
- The degree of compression must be judged carefully: compress too hard and the detail cannot come back; too little and no compute is saved. The paper systematically compares different downsampling factors and finds a near-optimal working point — where it pulls away from earlier two-stage generation (such as VQGAN plus an autoregressive model): the latent representation retains what is perceptually important, leaving diffusion free to concentrate on generation.
- Conditioning is handled by cross-attention: conditions such as text and bounding boxes are encoded and injected into the diffusion denoising network through cross-attention layers — a component from No. 1 in this volume, the Transformer, installed inside an image generator. Text-to-image acquired a scalable implementation route from here on.
- The effect is in the compute accounting: a decent consumer-grade GPU can train and run high-resolution generation. In August 2022, Stability AI open-sourced Stable Diffusion on the basis of this paper, and the ecosystem explosion brought by public weights (plug-ins, fine-tunes, ControlNet, and more) is something closed models have not replicated.

**Relation to preceding and following work**

It is the direct descendant of No. 6, DDPM, solving precisely the "too slow, too expensive" that DDPM left behind. At the same time it stitches this volume's two lines together: the denoising skeleton of diffusion, plus the attention conditioning mechanism of the Transformer — multimodal generation after 2022 has largely followed this seam.

**Original link**: https://ar5iv.labs.arxiv.org/html/2112.10752

---

*Verification note for this volume: the bibliographic details and abstracts of all thirteen papers were checked against the arXiv abstract pages or the original paper texts. GPT-2 has no arXiv preprint; its abstract was checked against the original OpenAI technical report (cross-checked against the transcription in Semantic Scholar's bibliographic record). Four version differences are noted. The Transformer abstract follows the current arXiv version (English-to-French BLEU 41.8); the first version gave 41.0 for that figure, with slightly different wording. The BERT abstract follows the revised arXiv version (GLUE 80.5%, including SQuAD v2.0 at 83.1); the first version's figures differed slightly (GLUE 80.4%, the MultiNLI improvement stated as 5.6 points, and no SQuAD v2.0). The LoRA abstract follows arXiv v2, which added the GLUE experiments and the discussion of adapter latency; the v1 abstract was more briefly worded. The latent diffusion abstract follows the CVPR 2022 published version (i.e., arXiv v2), whose task list differs slightly from arXiv v1 (v1 read "semantic scene synthesis," where the published version reads "text-to-image synthesis"). In addition, the arXiv abstract of Chinchilla contains a typographical duplication ("4× more more data"); it is reproduced verbatim above, and the NeurIPS published version corrects the error. Two entries were added in October 2026 — RAG (No. 7) and ReAct (No. 12), the foundation papers of the agent application track; their bibliographic details and abstracts were checked against the arXiv abstract pages, following the current versions (RAG v4, revised April 2021; ReAct v3, revised March 2023).*
