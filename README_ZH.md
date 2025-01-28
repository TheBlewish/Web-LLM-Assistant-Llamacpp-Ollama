# Web-LLM Assistant

## 概述

Web-LLM Assistant 是一个简单的网络搜索助手，它利用通过 Llama.cpp 或 Ollama 运行的大型语言模型（LLM），为用户提供信息丰富且具有上下文感知能力的回答。该项目结合了 LLM 的强大功能和实时网络搜索能力，使其能够获取最新的信息并综合出全面的答案。

以下是它在实际中的工作方式：

你可以向 LLM 提问，例如：“波音星际客机是否仍然停留在国际空间站上”，然后 LLM 将决定一个搜索查询和一个时间范围，以获取搜索结果，例如，根据你的具体问题的需要，获取最近一天或最近一年的结果。

然后它将进行网络搜索，并收集前 10 个结果及其包含的信息，然后选择 2 个最相关的结果并进行网络爬取以获取这些结果中包含的信息，在审查信息后，它将决定这些信息是否足以回答你的问题。如果足够，LLM 将回答问题；如果不足够，LLM 将进行新的搜索，可能会重新措辞搜索词和/或时间范围，以找到更合适和相关的信息来回答你的问题，它可以继续进行多次搜索，细化搜索词或时间范围，直到它有足够的信息来真正回答用户的问题，或者直到它进行了 5 次单独的搜索，从 LLM 决定的每次搜索的前 2 个相关结果中获取信息，如果此时它仍然无法找到回答用户问题所需的信息，它将尽力提供从搜索中获取的任何信息，以尽可能好地回答你的问题。

因此，你可以向它询问有关最近事件的查询，或者任何可能实际上不在其训练数据中的内容。现在，通过这个 Python 程序，即使答案不在 LLM 的训练数据中，它也可以通过网络搜索和检索这些搜索中的信息来确定你问题的答案。

## 项目演示

[![Web-LLM-Assistant Llama-cpp demonstration](https://img.youtube.com/vi/ZXbMCet5kjo/0.jpg)](https://youtu.be/ZXbMCet5kjo "Web-LLM-Assistant Llama-cpp demonstration")

点击上面的图片，观看 Web-LLM-Assistant Llama-cpp 的演示。

## 特性

- 通过 llama_cpp 或 ollama 使用本地 LLM。
- 网络爬取搜索结果，为 LLM 提供完整信息以供使用
- 使用 DuckDuckGo 进行网络搜索，以隐私为中心搜索页面以供爬取
- 自我改进的搜索机制，根据初始结果细化查询
- 丰富的控制台输出，带有彩色和动画指示器，以提供更好的用户体验
- 多次尝试搜索，智能评估搜索结果
- 综合使用 LLM 知识、网络搜索结果以及 LLM 选定网页的爬取信息，合成全面的答案

## 安装方法

1. 克隆仓库： 从 GitHub 克隆仓库：

git clone https://github.com/TheBlewish/Web-LLM-Assistant-Llamacpp-Ollama

然后导航到项目目录。

1. 创建虚拟环境（可选但推荐）： python -m venv venv source venv/bin/activate  # 在 Windows 上，使用 venv\Scripts\activate
2. 安装所需的依赖项： 通过运行 pip install -r requirements.txt 安装所需的包
3. 安装并设置 Ollama 或 Llama.cpp，然后根据你计划使用的，继续下面的 Ollama 或 Llama.cpp 使用说明。

## 使用方法

使用 Ollama：

1. 启动 Ollama 服务器： 运行命令 ollama serve 启动 Ollama 服务器。
2. 使用 Ollama 下载你想要的模型： 使用 ollama pull 命令，后跟你要使用的模型名称，例如，ollama pull gemma2:9b-instruct-q5_K_M
3. 配置 LLM 设置： 打开 llm_config.py 文件，将 LLM_TYPE 更新为 "ollama"，并将 "model_name" 设置为你下载的 Ollama 模型的名称。
4. 运行主脚本： 通过运行 python Web-LLM.py 执行主脚本。

使用 Llama.cpp：

1. 准备你的模型文件： 确保你有一个兼容的模型文件（例如，Phi-3-medium-128k-instruct-Q6_K.gguf）在你期望的位置。
2. 配置 LLM 设置： 打开 llm_config.py 文件，将 LLM_TYPE 更新为 "llama_cpp"。将 MODEL_PATH 设置为你的模型文件的路径。根据需要更新配置文件中 llama.cpp 部分的其他设置。
3. 运行主脚本： 通过运行 python Web-LLM.py 执行主脚本。

与助手互动：

对于正常互动，只需输入你的消息并按 CTRL+D 提交。 要请求网络搜索，请以 '/' 开始你的消息。 示例："/latest news on AI advancements"

AI 将处理你的输入，如果请求，将执行搜索，并提供有根据的回答。

## 配置

你可以在 `llm_config.py` 文件中修改各种 llama.cpp 或 ollama 参数。

## 依赖项目

- llama-cpp-python 或 ollama
- 在 requirements.txt 中查看完整的依赖项列表
- 鉴于这是通过 Llama.cpp 或 ollama 运行的 LLM，请确保你安装了适当的模型，我推荐基本上任何 instruct 模型，你可以自由尝试并找到最适合你系统的模型！

## 贡献

欢迎并鼓励对 Web-LLM Assistant 进行改进的贡献！请随时提交拉取请求。

## 许可

本项目根据 MIT 许可证授权 - 请查看 [LICENSE] 文件了解详细信息。

## 致谢

- 感谢 Llama.cpp 和 Ollama 的创作者提供了本地 LLM 使用的基础。
- 感谢 DuckDuckGo 提供了他们的搜索 API。

## 免责声明

本项目仅用于教育目的。请确保你遵守所有 API 和服务的使用条款。

## 作者一些话

我尽力创建了一个允许使用本地 Llama.cpp 运行的 LLM 进行网络搜索的东西，过去我一直对像 ChatGPT 这样的服务能够进行网络搜索，而本地模型却从未能够轻松做到这一点感到沮丧。当然，现在这个程序已经更新，也可以与 Ollama 一起使用，在多次请求后，我尽力满足人们的需求，现在可以使用 Ollama 以及 Llama.cpp 与这个程序一起使用！

Web-LLM Assistant 代表了无数小时的学习、编码和解决问题的努力，这实际上是我在编码方面的第一次尝试，尤其是我从零开始构建的任何东西。

我希望你发现这个程序既实用又有趣！请在 github 上留下这个程序的任何问题或任何建议，我将修复问题并在这个程序上实现任何合理的关于新功能的建议！