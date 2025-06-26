# adv_text_watermark/attacks/implicit_attack.py

import torch
import torch.nn as nn

# 从配置文件中导入超参数
from utils import config


class Actor(nn.Module):
    """
    演员网络 (Policy Network)

    负责根据输入序列生成一个同样长度的扰动序列。
    它使用一个Bi-LSTM来理解上下文，并为每个位置输出一个扰动向量。
    """

    def __init__(self, embedding_dim: int, hidden_dim: int, perturbation_scale: float = 0.1):
        """
        初始化Actor网络。

        Args:
            embedding_dim (int): 词嵌入的维度。
            hidden_dim (int): Actor内部LSTM的隐藏层维度。
            perturbation_scale (float): 控制扰动大小的缩放因子。
        """
        super(Actor, self).__init__()
        self.perturbation_scale = perturbation_scale

        # Bi-LSTM层，用于捕捉序列上下文
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)

        # 线性层，将LSTM的输出映射回嵌入维度，作为扰动向量
        self.fc = nn.Linear(hidden_dim * 2, embedding_dim)

        # Tanh激活函数，将扰动限制在[-1, 1]之间，再用scale缩放
        self.tanh = nn.Tanh()

    def forward(self, embedded_input: torch.Tensor) -> torch.Tensor:
        """
        生成扰动。

        Args:
            embedded_input (torch.Tensor): 原始词嵌入 (batch_size, seq_len, embedding_dim)

        Returns:
            torch.Tensor: 生成的扰动 (batch_size, seq_len, embedding_dim)
        """
        # embedded_input shape: (batch_size, seq_len, embedding_dim)
        lstm_out, _ = self.lstm(embedded_input)
        # lstm_out shape: (batch_size, seq_len, hidden_dim * 2)

        perturbations = self.fc(lstm_out)
        # perturbations shape: (batch_size, seq_len, embedding_dim)

        # 使用tanh限制范围，并用一个小的缩放因子控制扰动强度
        scaled_perturbations = self.tanh(perturbations) * self.perturbation_scale

        return scaled_perturbations


class Critic(nn.Module):
    """
    评论家网络 (Value Network)

    负责评估当前状态（即原始输入序列）的价值。
    其输出是对最终组合损失（任务损失+水印损失）的一个预测。
    """

    def __init__(self, embedding_dim: int, hidden_dim: int):
        """
        初始化Critic网络。

        Args:
            embedding_dim (int): 词嵌入的维度。
            hidden_dim (int): Critic内部LSTM的隐藏层维度。
        """
        super(Critic, self).__init__()

        # Bi-LSTM层，用于编码状态信息
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)

        # 线性层，将序列信息聚合后，回归到一个标量值（预测的损失）
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, embedded_input: torch.Tensor) -> torch.Tensor:
        """
        预测状态价值。

        Args:
            embedded_input (torch.Tensor): 原始词嵌入 (batch_size, seq_len, embedding_dim)

        Returns:
            torch.Tensor: 预测的价值（损失）(batch_size, 1)
        """
        # embedded_input shape: (batch_size, seq_len, embedding_dim)
        lstm_out, _ = self.lstm(embedded_input)
        # lstm_out shape: (batch_size, seq_len, hidden_dim * 2)

        # 我们使用最后一个时间步的输出来代表整个序列的状态
        last_hidden_state = lstm_out[:, -1, :]
        # last_hidden_state shape: (batch_size, hidden_dim * 2)

        value = self.fc(last_hidden_state)
        # value shape: (batch_size, 1)

        return value


class ImplicitAttack(nn.Module):
    """
    隐式层攻击 (Implicit Layer Attack)。

    封装了Actor和Critic网络。在训练期间，需要分别定义和更新它们的优化器。
    """

    def __init__(self, embedding_dim: int, hidden_dim: int):
        super(ImplicitAttack, self).__init__()
        self.actor = Actor(embedding_dim, hidden_dim)
        self.critic = Critic(embedding_dim, hidden_dim)

    def forward(self, embedded_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        执行隐式攻击。

        Args:
            embedded_input (torch.Tensor): 原始词嵌入 (batch_size, seq_len, embedding_dim)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
            - perturbed_embeddings (torch.Tensor): 施加扰动后的嵌入
            - state_value (torch.Tensor): Critic对当前状态的价值评估
        """
        perturbations = self.actor(embedded_input)
        state_value = self.critic(embedded_input)  # Critic评估的是原始状态

        perturbed_embeddings = embedded_input + perturbations

        return perturbed_embeddings, state_value


# --- 独立测试块 ---
if __name__ == '__main__':
    print("--- Running a self-test for ImplicitAttack (Actor-Critic) ---")

    # 配置参数
    batch_size = config.BATCH_SIZE
    max_len = config.MAX_LEN
    embedding_dim = config.EMBEDDING_DIM
    hidden_dim = 128  # 内部LSTM维度

    # 1. 实例化模型
    implicit_attacker = ImplicitAttack(embedding_dim, hidden_dim).to(config.DEVICE)
    print("\nModel instantiated:")
    print(implicit_attacker)

    # 2. 创建虚拟输入
    dummy_embeddings = torch.randn(batch_size, max_len, embedding_dim).to(config.DEVICE)
    print(f"\nCreated dummy embedding tensor of shape: {dummy_embeddings.shape}")

    # 3. 执行前向传播
    perturbed_embeddings, state_value = implicit_attacker(dummy_embeddings.clone())
    print("Forward pass successful!")
    print(f"Perturbed embeddings output shape: {perturbed_embeddings.shape}")
    print(f"State value output shape: {state_value.shape}")

    # 4. 验证形状和逻辑
    assert perturbed_embeddings.shape == dummy_embeddings.shape
    assert state_value.shape == (batch_size, 1)

    diff = torch.sum(dummy_embeddings - perturbed_embeddings).abs().item()
    print(f"Total absolute difference between original and perturbed: {diff:.4f}")
    assert diff > 1e-6, "Embeddings should have been modified by the actor's perturbations!"
    print("Perturbation successfully applied.")

    print("\n--- Self-test Passed! ---")
