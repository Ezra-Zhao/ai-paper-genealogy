# llama.cpp：什么硬件都能跑的手动挡

llama.cpp 是一个用 C/C++ 写的推理引擎，目标就是"在尽可能多的硬件上把大模型跑起来"：新显卡、老显卡、纯 CPU、苹果芯片、AMD 显卡，甚至一些边缘设备。它是很多上层工具（包括 Ollama）的底层。

跟 Ollama 比，它什么都要自己来：自己找模型文件、自己决定放多少层到显卡上、参数全摊开给你调。换来的是控制力——显存抠到最后一百兆、模型大到一张卡装不下时怎么拆着跑，只有它能办。适合想搞明白原理的人，和硬件条件比较苛刻的人。

## 先把程序弄到手

**方式一：官方安装脚本（最省事）**

```bash
# macOS / Linux
curl -LsSf https://llama.app/install.sh | sh

# Windows PowerShell
irm https://llama.app/install.ps1 | iex
```

装好后官方的新用法是统一的 `llama` 命令，可以直接从 Hugging Face 拉模型开跑：

```bash
# 下载模型并进入对话
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# 起一个 OpenAI 兼容的 API 服务
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

**方式二：下载预编译包**

GitHub 上 llama.cpp 仓库的 Releases 页面提供各平台编译好的压缩包，解压即用，不用装编译环境。包里除了统一的 `llama` 命令，也常带传统的分开命名的工具：`llama-cli`（命令行对话）、`llama-server`（HTTP 服务）等，两种写法是同一套程序。本篇后面的参数讲解以这两个传统命令为准，因为参数最全、资料最多，新统一命令的用法与它们基本对应。

**方式三：自己编译（想要 GPU 加速、想用最新代码时）**

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

# 纯 CPU 编译，这样就能编
cmake -B build
cmake --build build --config Release -j

# 有 NVIDIA 显卡，加上 CUDA 后端
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j
```

其他后端：AMD 显卡把 `GGML_CUDA` 换成 `GGML_HIP=ON`（ROCm）；想用 Vulkan 跨平台 GPU 用 `GGML_VULKAN=ON`；苹果芯片的 Metal 加速在 macOS 上默认就是开的。编译产物在 `build/bin/` 目录下。

## 模型文件：GGUF

llama.cpp 只认 **GGUF** 格式——一个 `.gguf` 文件里装着权重、分词器和对话模板，下下来就是一个文件，干净利落（更老的 GGML `.bin` 格式已经废弃，看到老教程让你下 `.bin` 直接跳过）。

拿模型有两种方式：

```bash
# 方式一：让 llama.cpp 直接从 Hugging Face 下载并缓存
llama-cli -hf unsloth/Qwen3-4B-GGUF

# 方式二：自己下好 .gguf 文件，用 -m 指定路径
llama-cli -m ./Qwen3-4B-Q4_K_M.gguf
```

GGUF 文件去 Hugging Face 上找（下载来源的注意事项见本篇总览）：同一个模型通常会挂十几个文件，文件名里那串 `Q4_K_M`、`Q8_0` 就是量化等级——下一节讲怎么选。缓存下载的模型默认放在 `~/.cache/llama.cpp/`。

手头有 Hugging Face 原始格式的权重想转成 GGUF，仓库自带转换脚本：

```bash
python3 convert_hf_to_gguf.py /你的模型目录 --outfile my-model.gguf
```

## 量化等级怎么选

先用大白话说清量化是什么。一个参数原本用 16 位小数存，数字记得很细，但占地方。量化就是把这些数字"记粗一点"：16 位压到 8 位、5 位、4 位，文件立刻小一圈，代价是每个数字都不再精确，模型的回答质量会打折扣。可以类比照片：原图比作全精度，Q8_0 像高质量 JPEG——肉眼几乎看不出差别；Q4_K_M 像压缩得更狠一点的图，日常看没问题，放大细看能发现涂抹；压到 Q2，就像马赛克打得很厚的图，五官还在，细节别指望了。

常见的几档（以 8B 模型为例，大小为约数）：

| 量化 | 每参数约占位数 | 8B 文件大小 | 质量损失 | 适合什么时候 |
|---|---|---|---|---|
| F16 | 16 | 约 16 GB | 无（原始） | 显存多得用不完，或做基准对比 |
| Q8_0 | 约 8.5 | 约 8.5 GB | 几乎察觉不到 | 显存够、想要最接近原版的效果 |
| Q6_K | 约 6.6 | 约 6.5 GB | 很小 | 显存略有富裕 |
| Q5_K_M | 约 5.7 | 约 5.5 GB | 小 | Q4 之上再求稳妥一点 |
| **Q4_K_M** | 约 4.8 | 约 4.7 GB | 小，日常使用完全够 | **默认选它，不用纠结** |
| Q3_K_M | 约 3.9 | 约 3.5 GB | 明显 | 显存实在紧张，又想跑大一号模型 |
| Q2_K | 约 2.6–3.4 | 约 2.5–3 GB | 大 | 不推荐；这种情况下不如换小参数的模型用 Q4 |

选择顺序记住三条就行：

1. 先装 Q4_K_M，这是社区公认的性价比甜点，Ollama 的默认档也是它。
2. 显存还有富裕，往上加到 Q5_K_M 或 Q8_0，质量只涨不跌。
3. 显存不够，先考虑换小一号参数的模型再保持 Q4；只有当你明确就是要用某个大模型时，才往 Q3 压，Q2 基本不要碰。

还有一个规律：模型越大，量化掉的质量比例越小。70B 模型压到 Q4 依然很能打，3B 小模型压到 Q3 就可能明显变笨。

## 跑起来

命令行对话：

```bash
llama-cli -m ./Qwen3-4B-Q4_K_M.gguf -ngl 99
```

起一个带网页界面的 HTTP 服务（兼容 OpenAI 接口格式，默认端口 8080）：

```bash
llama-server -m ./Qwen3-4B-Q4_K_M.gguf -ngl 99 --port 8080
```

浏览器打开 `http://localhost:8080` 就有现成的聊天界面；程序调用走 `http://localhost:8080/v1/chat/completions`，和 OpenAI 的调用方式一样。

## 显存不够：层卸载（offload）

上面命令里的 `-ngl 99` 是这一篇最重要的参数。`-ngl` 是 `--n-gpu-layers` 的缩写，意思是"把模型的多少层放到显卡上算"。

模型的计算是一层一层往下传的（论文里的 Transformer 层数就是它）。显卡算得快但显存小，内存大但 CPU 算得慢。`-ngl` 就是在两者之间划线：

- `-ngl 99`（或写很大的数；新版也可以直接写 `-ngl all`，不写时默认 `auto` 自动判断）：所有层都放显卡，速度最快，前提是显存装得下。
- 装不下就往下减，比如 `-ngl 20`：20 层在显卡上，剩下的层放内存里由 CPU 算。速度会明显变慢（瓶颈在 CPU 那部分），但至少跑得起来——一个 32B 模型配一张 12 GB 的卡，就是靠这招硬扛的。
- `-ngl 0`：纯 CPU 跑，最慢兜底。

实操时可以这样试：先 `-ngl 99`，报显存不足就一次减几层，减到能加载为止；加载成功后看生成速度能不能接受，不能接受就回头降量化或换小模型。与其在参数上死磕，不如记住这个先后顺序：**降量化档位 → 减卸载层数**，两招配合，几乎任何机器都能找到一个"能跑"的组合，只是速度快慢的问题。

还有一个相关提醒：上下文开得越大（让它一次读更长的文），KV 缓存吃掉的显存越多，留给权重的空间就越少。如果加载时报显存不足，除了动 `-ngl`，把上下文长度调小也能挤出空间。

---

资料来源（2026 年 10 月核对）：llama.cpp 官方仓库 github.com/ggml-org/llama.cpp 的 README（安装方式、`llama cli` / `llama serve` 用法）与官方 build 文档（CMake 后端选项）；量化各档的位数、体积与质量对照综合自 GGUF 量化公开资料并与官方 `-hf` 默认选择（Q4_K_M）互校。
