# vLLM：给很多人同时用的服务器级部署

vLLM 出自 UC Berkeley，是一个专为"把模型当服务对外提供"设计的推理引擎。它的看家本领是显存管理和调度：用 PagedAttention 把 KV 缓存像操作系统管理内存分页那样管起来，再用连续批处理把不断到来的请求动态拼批。一句话——**同一张显卡，它能同时伺候的人比朴素实现多得多**。不少商业 API 服务和公司内部平台，底下跑的就是它。

## 先说清楚：什么时候才需要它

不需要的情况占大多数：

- 自己一个人用，或者给手头的程序当后端 → Ollama 就够了，别折腾。
- 机器是 Mac、Windows，或者没有 NVIDIA 显卡 → 直接放弃这条路线（原因见下）。
- 只是想学习、跑通流程 → Ollama 十分钟的事，vLLM 的环境能配一下午。

需要它的情况很明确：

- 你要**给团队、给产品、给一批用户提供 API 服务**，请求是并发来的，吞吐量就是钱。
- 要**批量离线处理**大量文本（比如一万条工单摘要），想把显卡的每一分算力榨干。
- 模型大到一张卡装不下，需要**多卡并行**把一个模型拆开跑。

单人聊天场景下，vLLM 不会比 Ollama 快，启动反而慢得多（它要预热、占显存、做图捕获）。工具没有高下，只有合不合场景。

## 门槛

按官方文档的要求：

- **系统**：Linux。vLLM 完整功能只在 Linux 上运行。
- **显卡**：以 NVIDIA 为主，GPU 计算能力 7.0 以上（V100、T4、RTX 20 系及更新的显卡）。AMD 显卡有 ROCm 路线，但折腾程度高一个量级，新手别碰。
- **Python**：3.11–3.14（按官方最新版文档；旧版本是 3.10–3.13，装之前顺手核对一眼文档），且强烈建议用一个全新的虚拟环境装，不要跟其他项目共用环境，它的依赖（尤其是 PyTorch 和 CUDA 版本）很容易跟别人打架。
- **模型格式**：vLLM 直接吃 Hugging Face 上的原始权重（safetensors 格式），**不是 GGUF**。别把 llama.cpp 的 `.gguf` 文件拿来喂它，两条路线的模型文件不通用。

## 安装

```bash
# 先建一个干净的虚拟环境
python3 -m venv .venv
source .venv/bin/activate

pip install vllm
```

装完验证一下：

```bash
vllm --help
```

能列出 `serve` 等子命令就说明装好了。生产环境也可以用官方 Docker 镜像 `vllm/vllm-openai`，容器里启动方式与下面一致。

## 启动服务

先用一个小模型把链路跑通（Qwen2.5 的 1.5B 版本，文件小、下载快、无需任何许可手续）：

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct
```

第一次运行会自动从 Hugging Face 下载模型权重。看到日志里出现 `Application startup complete`、服务监听在 `http://localhost:8000`，就是起来了。验证一下：

```bash
curl http://localhost:8000/v1/models
```

然后换成正经要用的模型，比如 8B 的指令模型：

```bash
vllm serve NousResearch/Meta-Llama-3-8B-Instruct --dtype auto --api-key token-abc123
```

### 常用启动参数

| 参数 | 作用 | 说明 |
|---|---|---|
| `--host` / `--port` | 监听地址和端口 | 默认 `localhost:8000`；对外提供服务时 `--host 0.0.0.0` |
| `--api-key` | 给接口加一把钥匙 | 对外服务时务必设置；也可以用环境变量 `VLLM_API_KEY` |
| `--max-model-len` | 上下文长度上限 | 模型自带的长度可能很大，KV 缓存会吃不消；按实际需求设（如 16384），显存立刻宽松 |
| `--gpu-memory-utilization` | 允许 vLLM 使用的显存比例 | 默认 0.9；这张卡还要干别的活就调低，启动时显存不够也先调它 |
| `--tensor-parallel-size` | 用几张卡并行跑一个模型 | 模型一张卡装不下时用，数值等于显卡数，如 70B 模型配 `--tensor-parallel-size 4` |
| `--dtype auto` | 权重精度 | 一般保持 auto，让它按模型配置来 |

举个多卡的完整例子：

```bash
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 4 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90
```

## 调用它

接口完全是 OpenAI 格式，OpenAI 的官方客户端把地址一换就能用：

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123",  # 与启动时的 --api-key 对应；没设钥匙时随便填
)

completion = client.chat.completions.create(
    model="NousResearch/Meta-Llama-3-8B-Instruct",
    messages=[{"role": "user", "content": "你好，介绍一下你自己"}],
)
print(completion.choices[0].message.content)
```

注意 `model` 字段要填启动命令里那个完整的模型名，一个字都不能差。

## 受限模型：先办许可再下载

像上面例子里的 `meta-llama/Llama-3.1-70B-Instruct`，属于受限发布模型：直接启动会在下载环节卡住，要求登录。流程是：

1. 在 Hugging Face 网站上打开该模型页面，同意它的许可协议（用你自己的账号）。
2. 在本机登录同一个账号：安装 `huggingface_hub` 后执行 `huggingface-cli login` 填入 token，或者把 token 放进环境变量 `HF_TOKEN`。
3. 再启动 `vllm serve`，下载就能通过。

第一次遇到下载报 401/403，先查这一步，十有八九是许可没点、账号没登。

## 上线之前再提醒两句

- vLLM 默认会把显卡显存占到九成（`--gpu-memory-utilization` 的默认值），这是它为吞吐做的设计，不是失控。但同一张卡上别再指望同时跑别的模型服务。
- 它一个服务进程只伺候一个模型。想同时提供两个模型，就起两个进程（不同端口），或者上网关分流——这已经是运维的范畴，不在本篇之内了。

---

资料来源（2026 年 10 月核对）：vLLM 官方文档 docs.vllm.ai 的 Quickstart（`vllm serve` 用法）、OpenAI-Compatible Server（接口与客户端示例）、Installation（系统与显卡要求）、Engine Args（参数默认值）。
