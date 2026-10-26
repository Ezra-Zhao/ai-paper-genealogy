# 论文卡：Agent AI 综述（Durante et al. 2024）

> 收录来源：Ezra 2026-10-25 推荐收录（短视频「大模型老霍」介绍此文后指定查找并入仓）。
> 审核状态：书目信息与摘要经 arXiv 原页逐项核实（2026-10-25）；本卡为收录索引卡，非全文精译导读，后期写书与做内容时作参考调用。

## 书目

- 标题：Agent AI: Surveying the Horizons of Multimodal Interaction
- 作者：Zane Durante, Qiuyuan Huang, Naoki Wake, Ran Gong, Jae Sung Park, Bidipta Sarkar, Rohan Taori, Yusuke Noda, Demetri Terzopoulos, Yejin Choi, Katsushi Ikeuchi, Hoi Vo, Li Fei-Fei（李飞飞）, Jianfeng Gao（共 14 人，斯坦福大学与微软）
- 出处：arXiv:2401.03568 [cs.AI]，2024 年 1 月 7 日提交，v2 修订于 2024 年 1 月 25 日；全文约 80 页；许可 CC BY 4.0
- 链接：摘要页 https://arxiv.org/abs/2401.03568 ；PDF https://arxiv.org/pdf/2401.03568

## 内容要点（据原文摘要）

- 定义「Agent AI」：一类交互系统，能感知视觉刺激、语言输入与其他环境数据，并产生有意义的具身行动（embodied actions）。
- 主张：把 Agent 嵌入真实或虚拟环境、用「下一具身行动预测」来改进 Agent，并引入外部知识、多感官输入与人类反馈。
- 关键论点：在有环境锚定的场景中发展 agentic AI，可以缓解大基础模型的幻觉与「环境不正确」输出。
- 愿景：从物理世界的具身 Agent，延伸到人人可创建虚拟/模拟场景并与其中 Agent 交互。

## 框架归纳（据中文解读整理，非原文逐字）

中文解读（微信公众号「划重点KeyPoints」长文、CSDN 核心要点版）把全文框架归纳为五模块认知闭环：环境与感知 → 认知（LLM/VLM 为核心）→ 行动 → 学习 → 记忆，靠环境反馈校准现实、压减幻觉；应用面落在游戏 NPC、机器人、医疗三域。引用此归纳时注「据解读整理」，正式引用回原文核对。

## 与流行图解的区别（防误引）

短视频里流传的三张图解是网友二手总结：其中「Agent ＝ LLM（大模型）＋ Planning（规划）＋ Memory（记忆）＋ Tools（工具）」是流行简化公式，不是本文的原始框架；本文讲的是多模态具身 Agent 的认知闭环。写书或做内容引用时以原文为准。

## 引用（芝加哥体例）

Durante, Zane, Qiuyuan Huang, Naoki Wake, Ran Gong, Jae Sung Park, Bidipta Sarkar, Rohan Taori, Yusuke Noda, Demetri Terzopoulos, Yejin Choi, Katsushi Ikeuchi, Hoi Vo, Li Fei-Fei, and Jianfeng Gao. "Agent AI: Surveying the Horizons of Multimodal Interaction." arXiv:2401.03568 (2024). https://arxiv.org/abs/2401.03568.
