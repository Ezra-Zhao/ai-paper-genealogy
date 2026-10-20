# Ollama 上手：十分钟在本地聊上大模型

Ollama 是三条路线里最省事的一条：安装一条命令，运行一条命令，不需要懂编译，不需要配 Python 环境。它底层是 llama.cpp/ggml 这一系的推理技术，只是把下载模型、加载、起服务这些活全包了。适合第一次在本地跑模型的人，也适合只想安安静静自己用的人。

## 安装

三个系统各一条命令（来自 Ollama 官方仓库）：

macOS 和 Linux：

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Windows（在 PowerShell 里执行）：

```powershell
irm https://ollama.com/install.ps1 | iex
```

不习惯命令行的，macOS 和 Windows 也可以去 ollama.com 下载安装包（Ollama.dmg / OllamaSetup.exe），双击装，跟装普通软件一样。

Linux 上装完它会注册成系统服务自动运行；macOS 和 Windows 上是一个常驻后台的小程序。装好之后，服务默认监听在本机 `127.0.0.1:11434` 端口。

## 拉模型，跑起来

```bash
# 下载一个模型（以 Google 的 Gemma 4 为例）
ollama pull gemma4

# 直接开聊（没下载过的模型，run 会先自动下载再运行）
ollama run gemma4
```

执行 `ollama run` 之后终端里就进入对话，输入问题回车即可，输入 `/bye` 退出。

机器配置一般的，先从小模型试起：

```bash
ollama run llama3.2:3b
```

这是 Meta Llama 3.2 的 30 亿参数版本，Q4 量化下文件约 2 GB，老笔记本也能跑得动，先拿它把流程走通，再换大模型。

### 模型名后面的标签是什么意思

`llama3.2:3b` 里冒号后面是标签，写的是参数规模和量化版本，比如 `8b` 表示 80 亿参数，后面还可能带量化标记（如 `q4_K_M`）。不写标签时，Ollama 给你该模型的默认版本，通常就是大小和质量最均衡的那一档（多数模型的默认档是 4 位量化）。显存不够就往小标签选，显存富裕可以选参数更大或量化更高的标签。每个模型有哪些标签，去 ollama.com/library 的模型页面上查，页面会标每个文件的大小。

## 常用命令

| 命令 | 作用 |
|---|---|
| `ollama pull 模型名` | 只下载，不运行 |
| `ollama run 模型名` | 运行并进入对话（没有会先下载） |
| `ollama list` | 看本地已经下载了哪些模型 |
| `ollama ps` | 看当前正加载在内存/显存里的模型 |
| `ollama show 模型名` | 看某个模型的参数、上下文长度等信息 |
| `ollama rm 模型名` | 删除本地模型，腾硬盘 |
| `ollama serve` | 手动启动服务（服务没在跑时用） |

下载中途断网不用慌，重新执行 `ollama pull` 会接着下。

## 让程序调用它：API

Ollama 装好之后不只是个聊天工具，它同时是一个本地 API 服务，两种接口都提供。

它自己的接口：

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "gemma4",
  "messages": [{"role": "user", "content": "为什么天空是蓝色的？"}],
  "stream": false
}'
```

兼容 OpenAI 格式的接口（路径是 `/v1`）：

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # 必填但可以随便写，Ollama 不校验
)

response = client.chat.completions.create(
    model="gemma4",
    messages=[{"role": "user", "content": "用一句话解释什么是量化"}],
)
print(response.choices[0].message.content)
```

这个兼容接口的意义在于：你以前写过的调用 OpenAI 的代码、各种支持填自定义 API 地址的客户端软件，把地址换成 `http://localhost:11434/v1` 就能改用本地模型，一行业务逻辑都不用动。本仓库后面讲 Agent 和应用开发的部分，默认就用这个接口连本地模型。

## 让局域网里的其他设备也能用

默认情况下 Ollama 只听本机连接。想用手机或另一台电脑连过来（服务仍在你自己的局域网内），把监听地址放开：

```bash
OLLAMA_HOST=0.0.0.0 ollama serve
```

然后其他设备用 `http://这台机器的局域网IP:11434` 访问。

必须提醒一句：Ollama 本身没有账号密码这类鉴权，放开监听等于局域网里谁都能调你的模型。只在自己家的网络里这么做，**不要把 11434 端口映射到公网**。真有远程访问需求，用 Tailscale 这类组网工具，别裸奔。

## 模型文件放在哪，能不能用自己的

通过 Ollama 拉的模型由它自己管理存放，不需要你操心路径。如果你在 Hugging Face 上下到了 GGUF 文件想让 Ollama 跑，或者手头有其他格式的权重想导入，Ollama 提供导入功能（通过 Modelfile 指定文件），具体做法看官方文档 docs.ollama.com 的 Import 一节。刚上手时用不上，先把模型库里的模型玩熟。

## 常见问题

- **跑起来很慢**：先用 `ollama ps` 确认模型有没有跑在 GPU 上。如果是在纯 CPU 上跑，把参数规模换小一号（比如 8B 换 3B）立竿见影；只降量化档位帮助有限，而且回答质量还会跟着降。
- **提示显存不足 / 加载失败**：选更小的标签，或者关掉其他占显存的程序再试。
- **想试新模型**：去 ollama.com/library 逛，看到名字记下来，`ollama pull` 即可。试完不想要，`ollama rm` 删掉，硬盘别攒一堆用不上的模型。

---

资料来源（2026 年 10 月核对）：Ollama 官方仓库 github.com/ollama/ollama（安装命令、API 示例）、官方文档 docs.ollama.com（CLI 参考、OpenAI 兼容接口说明）。
