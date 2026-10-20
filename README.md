# AI 顶尖论文谱系：从奠基到前沿 — 总目录

> Ezra 2026-10-19 亲定立项的长期收藏工程：收齐 AI 史上最顶尖的论文，按时代整理成中文学习材料。每篇一个条目（中文译名 / 英文原题 / 作者与年份 / 一句话地位 / 摘要全文中译 / 核心思想导读 / 与前后论文的关系 / 原文链接）；标 ★ 的 10 篇另排全文精译队列，其余先做精译导读。

## 第一卷 奠基（1943–2006）→ `第一卷-奠基.md` ✅ 已完成（2026-10-19）

1. McCulloch & Pitts 1943 — 神经元的逻辑演算
2. Turing 1950 — 《计算机器与智能》
3. Rosenblatt 1958 — 感知机
4. ★ Rumelhart, Hinton & Williams 1986 — 反向传播（Nature）
5. LeCun et al. 1998 — 基于梯度的学习（LeNet）
6. Hochreiter & Schmidhuber 1997 — LSTM
7. Cortes & Vapnik 1995 — 支持向量机
8. Hinton et al. 2006 — 深度信念网络快速学习算法

## 第二卷 深度学习革命（2012–2016）→ `第二卷-深度学习革命.md` ✅ 已完成（2026-10-19）

★AlexNet 2012 / Seq2Seq 2014 / ★Bahdanau 注意力 2015 / GAN 2014 / VAE 2014 / BatchNorm 2015 / ★ResNet 2015 / AlphaGo 2016

## 第三卷 Transformer 与大模型（2017–2022）→ `第三卷-Transformer与大模型.md` ✅ 已完成（2026-10-19）

★Attention Is All You Need 2017 / ★BERT 2019 / GPT-2 2019 / ★GPT-3 2020 / Scaling Laws 2020 / DDPM 2020 / Chinchilla 2022 / LoRA 2021 / ★InstructGPT 2022 / Chain-of-Thought 2022 / Latent Diffusion 2022

## 第四卷 前沿（2023–2026）→ `第四卷-前沿.md` ✅ 已完成（2026-10-19）

GPT-4 技术报告 2023 / LLaMA 2023 / Mamba 2023 / DeepSeek-R1 2025 / ★Attention Residuals（Moonshot/Kimi，2026-03）

## 体例说明

- 翻译口径：去 AI 味、用词准确、术语首次出现给中英对照。
- 原文无摘要的早期论文（1943/1950 等）如实标注"原文无摘要"，不伪造译文，以导读替代。
- ★ 共 10 篇：反向传播、AlexNet、Bahdanau 注意力、ResNet、Attention Is All You Need、BERT、GPT-3、InstructGPT、Attention Residuals（外加一席待补，见各卷标注）。

## 进度

- 2026-10-19 立项；第一卷（8 篇）精译导读优先交付，Ezra 过目定体例后按卷推进，每卷完成在本窗口报一次。
- 2026-10-19 第一卷完成：8 篇书目与摘要均经出版社页面（Nature/IEEE/PubMed）或原文 PDF 核对；第 1 篇以正文首段代摘要，第 2、3 篇原文无现代式摘要已如实标注（第 3 篇为原文要点摘译，未伪造 Summary 译文）。注：DBN 为 2006 年，总目录卷名年份已由 1943–2005 更正为 1943–2006。
- 2026-10-19 他过目后批"体例可以，继续第二卷"；第二卷同日完成（8 篇全部核实，BatchNorm 按 ICML 正式版译出并注明与 arXiv 预印本的数字差异）。第三卷同日完成（11 篇全部核实，版本差异 5 处在卷末注明）。第四卷同日完成（5 篇全部核实）——**四卷精译导读全部收官，共 32 篇**。
- 《Attention Residuals》核实结论（2026-10-19）：真实存在，arXiv:2603.15031，Kimi Team（Moonshot AI），2026-03-16 提交；摘要据 arXiv 原文全译；马斯克原话经二手报道交叉核对为 "Impressive work from Kimi"；AttnRes 已于 2026-07 写进 Kimi K3 架构。个人作者细节未写入，只记团队与论文本身。
- ★ 全文精译队列 **9 篇全部完成**（2026-10-19 他点"开始 ★ 全文精译"）：`全文精译/01-反向传播-1986.md`（试点，据 Hinton 主页 Nature 扫描件逐页全译）、02-AlexNet、03-Bahdanau 注意力、04-ResNet、05-Attention Is All You Need、06-BERT、07-GPT-3、08-InstructGPT、09-Attention Residuals，合计约 713 KB。如实标注的缺漏：02 补充材料未收录；07 附录 F/G/H 为明示节译；08 附录作代表性处理；09 附录 A 作者名单按"只记团队"口径未录——均在各篇文件内注明，无静默删节。**第 10 篇之问已结**（2026-10-20 他回"都来吧"）：候选三篇 AlphaGo 2016 / DeepSeek-R1 2025 / 图灵 1950 全部补译，★ 队列扩为 12 篇，第二批三篇进行中。
- 2026-10-20 第二批 3 篇完成，★ 全文精译 12 篇收齐（合计约 1,036 KB）；全集（四卷导读 + 12 篇精译 + 英文版导读）已完成交叉审核（独立审稿 + 一手来源逐项核对），本仓即审核通过版。

## 版权与使用说明

- 导读、核心思想、承前启后等原创文字部分以 MIT 许可发布，署名 Guangyi Zhao。
- 各篇论文的摘要译文与全文精译为个人学习研究用途的翻译；原论文版权归其作者与出版方所有，引用与查阅请以原文链接所指的正式版本为准。如有权利人认为不妥，请开 issue 联系删除。
