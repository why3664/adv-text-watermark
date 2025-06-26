# adv_text_watermark/attacks/interference_attack.py

import torch
import torch.nn as nn

# 从配置文件中导入超参数
from utils import config


class InterferenceAttack(nn.Module):
    """
    目标模型干扰攻击 (Interference Attack)。

    该模块学习一个静态的、与具体输入无关的扰动场。
    其唯一目标是在训练过程中，通过梯度下降学习如何修改词嵌入
    以最小化目标模型的任务损失（例如，分类损失）。
    """

    def __init__(self, embedding_dim: int, max_len: int):
        """
        初始化干扰攻击层。

        Args:
            embedding_dim (int): 目标模型词嵌入的维度。
            max_len (int): 文本序列的最大长度。
        """
        super(InterferenceAttack, self).__init__()
        self.embedding_dim = embedding_dim
        self.max_len = max_len

        # --- 核心可学习参数 ---
        # 创建一个可学习的、静态的扰动矩阵。
        # 每一行代表对序列中一个特定位置施加的通用“性能优化”扰动。
        self.interference_vectors = nn.Parameter(
            torch.randn(self.max_len, self.embedding_dim) * 0.01
        )

    def forward(self, embedded_input: torch.Tensor) -> torch.Tensor:
        """
        在前向传播中应用干扰扰动。

        Args:
            embedded_input (torch.Tensor):
                原始的或已被其他层扰动过的词嵌入张量。
                Shape: (batch_size, seq_len, embedding_dim)

        Returns:
            torch.Tensor: 经过干扰扰动增强后的词嵌入张量。
                          Shape: (batch_size, seq_len, embedding_dim)
        """
        # 获取当前批次的实际序列长度
        seq_len = embedded_input.shape[1]

        # 截取与当前序列长度匹配的扰动向量
        # 注意：这里假设 seq_len <= self.max_len
        interference_to_apply = self.interference_vectors[:seq_len, :]

        # 将扰动向量 unsqueeze(0) 以匹配批次维度，然后加到输入上
        # PyTorch的广播机制会自动将其扩展到整个批次
        perturbed_embeddings = embedded_input + interference_to_apply.unsqueeze(0)

        return perturbed_embeddings


# --- 独立测试块 ---
if __name__ == '__main__':
    print("--- Running a self-test for InterferenceAttack ---")

    # 配置参数
    batch_size = config.BATCH_SIZE
    max_len = config.MAX_LEN
    embedding_dim = config.EMBEDDING_DIM

    # 1. 实例化模型
    interference_attacker = InterferenceAttack(
        embedding_dim=embedding_dim,
        max_len=max_len
    ).to(config.DEVICE)
    print("\nModel instantiated:")
    print(interference_attacker)
    print(f"Learnable interference vectors shape: {interference_attacker.interference_vectors.shape}")

    # 2. 创建虚拟输入
    dummy_embeddings = torch.randn(batch_size, max_len, embedding_dim).to(config.DEVICE)
    print(f"\nCreated dummy embedding tensor of shape: {dummy_embeddings.shape}")

    # 3. 执行前向传播
    perturbed_embeddings = interference_attacker(dummy_embeddings.clone())
    print("Forward pass successful!")
    print(f"Output shape: {perturbed_embeddings.shape}")
    assert perturbed_embeddings.shape == dummy_embeddings.shape

    # 4. 验证逻辑
    # 计算原始输入和扰动后输出之间的差异
    diff = torch.sum(dummy_embeddings - perturbed_embeddings).abs().item()
    print(f"\nTotal absolute difference between original and perturbed: {diff:.4f}")
    assert diff > 1e-6, "Embeddings should have been modified by the interference vectors!"
    print("Perturbation successfully applied.")

    print("\n--- Self-test Passed! ---")
