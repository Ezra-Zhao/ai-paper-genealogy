# AI Landmark Papers: From Foundations to the Frontier — Vol. 4 The Frontier (2023–2026)

English edition; abstracts are quoted verbatim from the original papers where available.

This volume collects five papers spanning little more than three years, during which the main thread changed course several times. In 2023, GPT-4 pushed the capability of closed models to its peak; in the same year LLaMA let open weights out of the cage, and Mamba promptly questioned the Transformer backbone itself. In 2025, DeepSeek-R1 showed that reasoning ability can be grown by pure reinforcement learning. In 2026, the Kimi team dug up and rebuilt even the residual connection — a piece of foundation untouched for a decade. It was seeing that last paper reposted by Elon Musk that prompted Ezra to commission this lineage in the first place. Each entry gives the bibliographic record, the abstract, the key ideas, and the paper's place in the lineage.

---

### 1. GPT-4 Technical Report

- **Authors & year**: OpenAI (collective authorship), 2023, technical report (arXiv preprint, submitted 15 March 2023)
- **One-line significance**: The peak work of the closed-source era — its capability touched the pass marks of human professional examinations, yet it withheld almost all details of architecture and training, turning the technical report from a "blueprint" into a "report card."

**Abstract** (source: arXiv:2303.08774, current version, v6)

> We report the development of GPT-4, a large-scale, multimodal model which can accept image and text inputs and produce text outputs. While less capable than humans in many real-world scenarios, GPT-4 exhibits human-level performance on various professional and academic benchmarks, including passing a simulated bar exam with a score around the top 10% of test takers. GPT-4 is a Transformer-based model pre-trained to predict the next token in a document. The post-training alignment process results in improved performance on measures of factuality and adherence to desired behavior. A core component of this project was developing infrastructure and optimization methods that behave predictably across a wide range of scales. This allowed us to accurately predict some aspects of GPT-4's performance based on models trained with no more than 1/1,000th the compute of GPT-4.

**Key ideas**

- Multimodality became standard on a flagship model: the input side expanded from text alone to image and text, while the output remained text. A simulated bar exam score around the top 10% and human-level performance across a range of professional and academic benchmarks — the persuasiveness of this report lies not in method but in the density of its report card.
- Predictable scaling is its real engineering claim: the infrastructure and optimisation methods were built to behave consistently across scales spanning several orders of magnitude, so results from small-model experiments could be extrapolated to large models. Predicting aspects of GPT-4's final performance from models trained with no more than one-thousandth of its compute means that a substantial part of what is called "emergence" was in fact on the schedule — the engineering fulfilment of the scaling laws and Chinchilla in Volume 3.
- Post-training alignment is written into the main line: pre-training supplies the capability, and post-training — RLHF as its representative — lifts factuality and adherence to desired behaviour. The source of capability and the source of behaviour are explicitly separated, a division of labour that became the standard narrative structure of every flagship report thereafter.
- The policy of withholding is stated in the report itself: for reasons of competition and safety, it discloses nothing about architecture, parameter count, training data, or method. It is therefore a rare exception in this lineage — selected for its influence rather than its reproducibility, with the methods section nearly blank.

**Relation to preceding and following work**

It directly inherits GPT-3 and InstructGPT from Volume 3: another step up in scale, with alignment now standard. But the pattern it set — "results only, no method" — is exactly what the next paper, LLaMA, set out to do the opposite of. In the same year, 2023, the two papers split the whole field into the closed and open roads.

**Original link**: https://arxiv.org/abs/2303.08774

---

### 2. LLaMA: Open and Efficient Foundation Language Models

- **Authors & year**: Hugo Touvron, Thibaut Lavril, Gautier Izacard, et al., 14 authors in total (Meta AI), 2023, arXiv preprint (27 February 2023)
- **One-line significance**: The igniter of the open-weights movement — trained only on public data, it produced a family of small but strong models in which 13 billion parameters beat the 175-billion-parameter GPT-3, returning large-model research from a handful of companies to academia.

**Abstract** (source: arXiv:2302.13971)

> We introduce LLaMA, a collection of foundation language models ranging from 7B to 65B parameters. We train our models on trillions of tokens, and show that it is possible to train state-of-the-art models using publicly available datasets exclusively, without resorting to proprietary and inaccessible datasets. In particular, LLaMA-13B outperforms GPT-3 (175B) on most benchmarks, and LLaMA-65B is competitive with the best models, Chinchilla-70B and PaLM-540B. We release all our models to the research community.

**Key ideas**

- Recipe over scale: it took Chinchilla's implication seriously — rather than train a larger model, feed a small model more data. This family was fed far more tokens than was customary for its size, while keeping the parameter count much smaller; training is expensive once, inference is cheap every time. For those who deploy models, it is the latter that is the long-term bill.
- Fully public data was a deliberate stance: the training corpora were drawn only from publicly available sources — web pages, code, Wikipedia, academic papers, and the like — and still reached the best level of performance. This answered the data opacity of GPT-3 and GPT-4 head-on: "what you can do, public data can do too."
- There is no new invention in the architecture; the win lies in careful selection of parts and in training discipline: pre-normalisation with RMSNorm, the SwiGLU activation function, and rotary positional embeddings (RoPE) — all components that had already appeared in Volume 3 — with the craft in how they are combined and tuned.
- The mode of release changed the ecosystem: once the weights were released under a research licence they spread rapidly, and within months fine-tuned derivatives were everywhere; "open weights" went from slogan to default option. Almost every open model since — including DeepSeek, the home of No. 4 in this volume — stands on this line.

**Relation to preceding and following work**

It is the first large-scale empirical validation of the Chinchilla recipe (Volume 3), and a "small-model counterattack" against GPT-3 in the same volume. Looking ahead, the open ecosystem left room for new architectures to be tested: a non-Transformer route such as Mamba, the next paper, was able to prove itself on real models through the open community rather than remaining at the level of theory.

**Original link**: https://arxiv.org/abs/2302.13971

---

### 3. Mamba: Linear-Time Sequence Modeling with Selective State Spaces

- **Authors & year**: Albert Gu (Carnegie Mellon University) and Tri Dao (Princeton University), 2023, arXiv preprint (1 December 2023), later presented at the 1st Conference on Language Modeling (COLM 2024)
- **One-line significance**: The first genuinely competitive challenger after six years of Transformer rule — it taught state space models to select by content, reaching five times the inference throughput of a Transformer, with complexity that grows linearly in sequence length.

**Abstract** (source: arXiv:2312.00752, current version, v2)

> Foundation models, now powering most of the exciting applications in deep learning, are almost universally based on the Transformer architecture and its core attention module. Many subquadratic-time architectures such as linear attention, gated convolution and recurrent models, and structured state space models (SSMs) have been developed to address Transformers' computational inefficiency on long sequences, but they have not performed as well as attention on important modalities such as language. We identify that a key weakness of such models is their inability to perform content-based reasoning, and make several improvements. First, simply letting the SSM parameters be functions of the input addresses their weakness with discrete modalities, allowing the model to selectively propagate or forget information along the sequence length dimension depending on the current token. Second, even though this change prevents the use of efficient convolutions, we design a hardware-aware parallel algorithm in recurrent mode. We integrate these selective SSMs into a simplified end-to-end neural network architecture without attention or even MLP blocks (Mamba). Mamba enjoys fast inference (5× higher throughput than Transformers) and linear scaling in sequence length, and its performance improves on real data up to million-length sequences. As a general sequence model backbone, Mamba achieves state-of-the-art performance across several modalities such as language, audio, and genomics. On language modeling, our Mamba-3B model outperforms Transformers of the same size and matches Transformers twice its size, both in pretraining and downstream evaluation.

**Key ideas**

- Diagnosis first: architectures that save compute had long existed; what they lost on was not efficiency but content. Attention is strong because it decides what to look at by content; in models such as linear attention and SSMs the parameters are independent of the input, so what to remember and what to forget is decided uniformly — an effective abandonment of content-based reasoning. Once the root cause is found, the prescription follows.
- The selectivity mechanism is the paper's central invention: the SSM's key parameters — the step size and the input and output matrices — are made to vary with each token. Important tokens are amplified and retained; irrelevant ones are filtered out. The recurrent structure thereby acquires an attention-like power of selection while keeping linear complexity.
- The cost has to be repaid with a hardware-aware algorithm: once the parameters vary with the input, the previously efficient convolution formula no longer applies. The authors instead compute in recurrent mode with a parallel scan, with operator fusion and recomputation tailored to the GPU memory hierarchy. Algorithm–hardware co-design is the paper's second main thread — the craft Dao had earlier brought to FlashAttention continues here.
- The backbone is built by subtraction: the selective SSM is fitted into a minimal block structure, with both attention and MLP blocks omitted. Mamba-3B outperforms Transformers of the same size and matches Transformers twice its size, showing that the removed components can be made up for by the selectivity mechanism.

**Relation to preceding and following work**

It takes up the old question left by the LSTM in Volume 1: how can a recurrent structure save compute without losing long-range information? The answer is to make forgetting itself learnable and triggered by content. It is also a head-on reply to the Transformer's monopoly in Volume 3: sequence mixing need not have attention as its only solution. Hybrid architectures mixing Transformer and SSM blocks subsequently became common in frontier models, and the models of Kimi — the home of No. 5 in this volume — are products of exactly this line.

**Original link**: https://arxiv.org/abs/2312.00752

---

### 4. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning

- **Authors & year**: DeepSeek-AI, 2025, arXiv preprint (22 January 2025), later published in *Nature* in 2025
- **One-line significance**: A proof that reasoning ability can come from pure reinforcement learning — given no human-annotated reasoning traces at all, only verifiable right and wrong, the model grew reflection, verification, and strategy-switching on its own.

**Abstract** (source: arXiv:2501.12948, current version, v2; identical to the abstract of the version formally published in *Nature* in 2025)

> General reasoning represents a long-standing and formidable challenge in artificial intelligence. Recent breakthroughs, exemplified by large language models (LLMs) and chain-of-thought prompting, have achieved considerable success on foundational reasoning tasks. However, this success is heavily contingent upon extensive human-annotated demonstrations, and models' capabilities are still insufficient for more complex problems. Here we show that the reasoning abilities of LLMs can be incentivized through pure reinforcement learning (RL), obviating the need for human-labeled reasoning trajectories. The proposed RL framework facilitates the emergent development of advanced reasoning patterns, such as self-reflection, verification, and dynamic strategy adaptation. Consequently, the trained model achieves superior performance on verifiable tasks such as mathematics, coding competitions, and STEM fields, surpassing its counterparts trained via conventional supervised learning on human demonstrations. Moreover, the emergent reasoning patterns exhibited by these large-scale models can be systematically harnessed to guide and enhance the reasoning capabilities of smaller models.

**Key ideas**

- A pure-RL start (R1-Zero): reinforcement learning is run directly on the base model, with rewards coming only from verifiable correctness of the answer and no human-annotated reasoning trajectories at all. In training, the model spontaneously lengthened its reasoning chains and developed behaviours such as self-reflection, going back to verify, and abandoning a strategy to start again — the "aha moment" recorded by the team is the paper's most widely retold detail. Reasoning was not taught; it was forced out by the reward.
- The released R1 is the engineering follow-up: the product of pure RL had poor readability and mixed Chinese and English. The remedy was first to supervised fine-tune on a small amount of high-quality cold-start data, then run reinforcement learning, then feed the high-quality trajectories produced by RL back in for another round of training, and finally run RL again — first prove the possibility, then restore usability.
- The nature of the reward signal changed: the reinforcement learning of InstructGPT in Volume 3 aligned the model to human preference — good or bad, a subjective judgement — whereas here it is aligned to objective correctness — right or wrong, and verifiable. Verifiable rewards do not need annotators scoring item by item and can scale with compute, which is the precondition for this line to work at all.
- Distillation democratised the capability: small models fine-tuned on reasoning trajectories generated by R1 obtained reasoning performance far beyond their own scale. Reasoning ability thereby changed from the preserve of the giants into a transferable, open-sourceable asset; model and data were released together, and the community took up the baton directly.

**Relation to preceding and following work**

It turns chain-of-thought in Volume 3 from "a technique elicited by prompting" into "an instinct produced by training," and continues the post-training route opened by InstructGPT, except that the use of reinforcement learning is changed from alignment to assault on hard problems. The race among flagships for "reasoning models" after 2025 all unfolded along this line. Like the next paper, it is an entry from a Chinese team — from this volume onward, the geography of the frontier is visibly rewritten.

**Original link**: https://arxiv.org/abs/2501.12948

---

### 5. Attention Residuals ★

- **Authors & year**: Kimi Team (Moonshot AI), 2026, arXiv preprint (submitted 16 March 2026, arXiv:2603.15031)
- **One-line significance**: A foundation untouched for a decade, renovated — the residual connection is changed from "equal-weight accumulation" to "attention along depth." Elon Musk reposted the paper on X with the comment "Impressive work from Kimi," an episode recorded here as cross-checked against multiple second-hand reports; it was seeing that repost that led Ezra to commission this lineage.

**Abstract** (source: arXiv:2603.15031)

> Residual connections with PreNorm are standard in modern LLMs, yet they accumulate all layer outputs with fixed unit weights. This uniform aggregation causes uncontrolled hidden-state growth with depth, progressively diluting each layer's contribution. We propose Attention Residuals (AttnRes), which replaces this fixed accumulation with softmax attention over preceding layer outputs, allowing each layer to selectively aggregate earlier representations with learned, input-dependent weights. To address the memory and communication overhead of attending over all preceding layer outputs for large-scale model training, we introduce Block AttnRes, which partitions layers into blocks and attends over block-level representations, reducing the memory footprint while preserving most of the gains of full AttnRes. Combined with cache-based pipeline communication and a two-phase computation strategy, Block AttnRes becomes a practical drop-in replacement for standard residual connections with minimal overhead.
>
> Scaling law experiments confirm that the improvement is consistent across model sizes, and ablations validate the benefit of content-dependent depth-wise selection. We further integrate AttnRes into the Kimi Linear architecture (48B total / 3B activated parameters) and pre-train on 1.4T tokens, where AttnRes mitigates PreNorm dilution, yielding more uniform output magnitudes and gradient distribution across depth, and improves downstream performance across all evaluated tasks.

**Key ideas**

- The lesion is in the words "equal weight": a standard residual adds every layer's output into the main stream unconditionally with weight 1. The deeper the network, the larger the magnitude of the hidden state grows, and the thinner each single layer's new contribution is spread as a share of the total — to be "heard," a deep layer has to output an ever larger magnitude, spending capacity on fighting for the microphone. This points to the same place as the recent discussion of the "curse of depth."
- The fix is to turn attention in a different direction: attention originally runs along the sequence and governs "which tokens to look at"; here it runs along depth and governs "which layers to look at." The current layer, using its own state as the query, applies softmax weighting over the outputs of all preceding layers and retrieves, by content, the information it genuinely needs. The most central component since Volume 1 is used to repair the most fundamental connection pattern since Volume 1.
- Practical deployment rests on blocking and caching: attending over all past layers requires storing and transmitting every layer's output, a burden neither memory nor the communication of pipeline parallelism can bear at scale. Block AttnRes aggregates by block and attends only over block-level representations, combined with cache-based point-to-point communication and a two-phase computation strategy, keeping the additional inference overhead within 2% — the point that makes it not a paper curiosity but a drop-in replacement that can enter an existing training stack.
- The gain survives scaling checks: scaling law experiments show the improvement is consistent across model sizes, equivalent to roughly a 1.25× advantage in compute — the same performance for about 80% of the compute; and once fitted into Kimi Linear for a real training run, downstream tasks such as mathematics and code improved across the board, with the clearest gains on multi-step reasoning. The learned attention weights do indeed show the pattern of "bypassing useless layers and taking key layers directly," matching the design intent.

**Relation to preceding and following work**

What it goes back to repair is the residual connection established by ResNet in Volume 2, and the method of repair is itself the attention of the Transformer in Volume 3 — the two thickest main threads of this lineage converge in this single paper. It shares Mamba's stance: no default component of the mainstream architecture is sacred any longer, and the skeleton can still be operated on. In July 2026, AttnRes was written into the architecture of Kimi K3, alongside Kimi Delta Attention as one of its two major mechanisms — the step from paper into flagship model.

**Original link**: https://arxiv.org/abs/2603.15031

---

*Verification note for this volume: the bibliographic details and abstracts of all five papers were checked word for word against the arXiv abstract pages. The current arXiv version of the GPT-4 Technical Report is v6 (revised in March 2024), and its abstract is unchanged from the first version. The DeepSeek-R1 abstract follows the current arXiv version (v2), which is identical to the abstract of the version formally published in Nature in 2025; the abstract of the first arXiv version in January 2025 was worded differently, its main thread being the training pipelines of R1-Zero and R1, and the Key ideas section above also covers the content of that first version. Paper No. 5, Attention Residuals, was verified directly on the official arXiv site: title, authoring team (Kimi Team, Moonshot AI), identifier arXiv:2603.15031, and submission date of 16 March 2026 are all correct, and the abstract is quoted verbatim from the arXiv abstract page; the official code repository is github.com/MoonshotAI/Attention-Residuals. The report that Elon Musk reposted the paper on X with the comment "Impressive work from Kimi" was cross-checked against multiple second-hand reports (aichina.news, chinaunread, KuCoin reposts, among others); accounts circulating on self-media about individual author details differ, and this volume, following its remit, records only the team and the paper itself, without adopting personal details. AttnRes was subsequently integrated into Kimi K3 (whose July 2026 technical report describes its architecture as "built on Kimi Delta Attention and Attention Residuals"), as checked against that report's arXiv record.*
