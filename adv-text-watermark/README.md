# Adversarial Text Watermarking via a Three-Layer Attack Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

这是一个用于学术研究的概念验证项目，旨在探索一种新颖的、基于三层对抗性攻击的文本水印方法。该框架通过在模型的词嵌入空间中进行微小且有针对性的扰动，将水印信息嵌入到由目标模型生成的文本中，同时尽可能保持原始任务的性能。

## 简介 (Introduction)

本项目实现了一个端到端的系统，用于在基于Bi-LSTM的文本分类任务中嵌入和检测不可感知的文本水印。其核心是一个创新的**三层攻击框架**，它将不同目标的扰动进行融合，以在水印的鲁棒性和任务性能之间取得平衡。

整个过程是**完全可微分的**，允许通过梯度下降进行端到端的联合优化。

## 核心特性 (Features)

* **端到端的可微框架**: 整个水印嵌入和检测过程都在PyTorch中实现，允许联合训练。
* **三层攻击架构**:
    1.  **显式攻击 (Explicit Attack)**: 基于预设规则，在文本的特定位置施加可学习的扰动以编码水印比特。
    2.  **隐式攻击 (Implicit Attack)**: 利用Actor-Critic强化学习框架，学习一种覆盖整个序列的动态扰动策略，以提高水印的隐蔽性。
    3.  **干扰攻击 (Interference Attack)**: 学习一个通用的、旨在最小化目标模型任务损失的静态扰动场，以补偿嵌入水印可能带来的性能下降。
* **模块化设计**: 高度解耦的代码结构，易于理解、修改和扩展。
* **可配置性**: 所有的重要超参数，如学习率、模型维度和攻击权重，都在一个中心化的`config.py`文件中进行管理。

## 项目结构 (Project Structure)

adv-text-watermark/
├── adv_text_watermark/
│ ├── attacks/
│ │ ├── explicit_attack.py # 显式层攻击
│ │ ├── implicit_attack.py # 隐式层攻击 (Actor-Critic)
│ │ ├── interference_attack.py # 干扰层攻击
│ │ ├── fusion.py # 三层攻击的融合模块
│ │ └── init.py
│ ├── data/
│ │ ├── dataset.py # PyTorch数据集类
│ │ ├── tokenizer.py # 自定义分词器
│ │ ├── watermark_utils.py # 水印处理相关函数
│ │ └── init.py
│ ├── models/
│ │ ├── target_model.py # 目标模型 (Bi-LSTM)
│ │ ├── detector.py # 水印检测器 (Bi-LSTM with Attention)
│ │ └── init.py
│ ├── utils/
│ │ ├── config.py # 全局配置文件
│ │ └── init.py
│ └── init.py
├── experiments/
│ └── run_attack.py # 项目主执行脚本
├── data/ # 存放原始数据 (例如 train.csv, val.csv)
├── requirements.txt # 项目依赖
├── setup.py # 项目安装配置文件
└── README.md # 项目说明


## 安装指南 (Installation)

1.  **克隆仓库**
    ```bash
    git clone <your-repository-url>
    cd adv-text-watermark
    ```

2.
````  **创建虚拟环境 (推荐)**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On
```` Windows, use `venv\Scripts\activate`
    ```

3.  **安装依赖**
    项目提供了`setup.py`，它
````会自动处理`requirements.txt`中的依赖。使用以下命令进行安装：
    ```bash
    pip install -e .
    ```
    `-e
`````标志表示以“可编辑”模式安装，这意味着您对源代码的任何更改都会立即生效，无需重新安装。

## 使用方法 (Usage)

2.  **配置实验**
    * 打开 `adv_text_watermark/utils/config.py` 文件。
````    * 根据您的需求调整超参数，例如`EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`, 水印载荷 `WATERMARK
````_PAYLOAD`, 以及三层攻击的融合权重 `W_EXPLICIT`, `W_IMPLICIT`, `W_INTERFERENCE`。

3
````.  **启动训练**
    一切准备就绪后，从项目根目录运行主脚本：
    ```bash
    python experiments/run
````_attack.py
    ```
    训练日志将显示在终端中，包括任务损失、水印损失和各项准确率。最佳模型将根据
````验证集上的水印准确率保存在`saved_models/`目录下。

## 核心概念：三层攻击框架

我们的水印嵌入过程不
````直接修改文本，而是在词嵌入空间中进行。最终的扰动由三层攻击加权融合而成：

* **P<sub>final</sub> =
```` E<sub>orig</sub> + w<sub>1</sub> * Δ<sub>explicit</sub> + w<sub>2</sub> * Δ<sub>implicit</sub> + w<sub>3</sub> * Δ<sub>interference
````</sub>**

其中：
- **E<sub>orig</sub>**: 原始文本的词嵌入。
- **Δ<sub>explicit</sub>**: **显式攻击**
````产生的扰动。它根据水印比特流在指定位置添加固定的、可学习的扰动向量，目标是精确编码信息。
- **
````Δ<sub>implicit</sub>**: **隐式攻击**产生的扰动。它由一个Actor-Critic网络生成，学习一种全局的、动态的扰动策略
````，目标是让水印更隐蔽、更鲁棒。
- **Δ<sub>interference</sub>**: **干扰攻击**产生的扰动。它学习一个通用的、
````与输入无关的扰动场，其唯一目标是“补偿”前两种攻击对主任务性能造成的损失。

## 待办事项 (Future
```` Work)

- [ ] 在更复杂的模型上进行测试，例如 `BERT` 或 `GPT`。
- [ ] 探索更先进的策略
````来选择嵌入水印的位置，而不是固定的前k个位置。
- [ ] 实现针对水印的移除攻击，以测试当前框架的鲁棒性。
````- [ ] 将最终的扰动嵌入转换为可读文本，并使用人工评估和困惑度（Perplexity）来衡量生成文本的质量
````。

## 许可证 (License)

本项目采用 [MIT License](LICENSE) 开源许可证。

