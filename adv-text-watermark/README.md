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


