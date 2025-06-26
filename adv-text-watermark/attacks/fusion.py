# adv_text_watermark/attacks/fusion.py

import torch
import torch.nn as nn

# 导入三层攻击模块
from .explicit_attack import ExplicitAttack
from .implicit_attack import ImplicitAttack
from .interference_attack import InterferenceAttack

# 从配置文件中导入超参数
from utils import config


class FusionAttack(nn.Module):
    """
    三层攻击融合模块。

    该模块整合了显式、隐式和干扰攻击，并将它们生成的扰动
    根据预设权重进行加权求和，生成最终的扰动嵌入。
    """

    def __init__(self, embedding_dim: int, watermark_length: int, max_len: int):
        """
        初始化融合模块。

        Args:
            embedding_dim (int): 词嵌入维度。
            watermark_length (int): 水印长度。
            max_len (int): 文本序列最大长度。
        """
        super(FusionAttack, self).__init__()

        # 1. 实例化三层攻击子模块
        self.explicit_attack = ExplicitAttack(embedding_dim, watermark_length)
        # 隐式攻击的内部LSTM隐藏维度可以自行设定，这里用一个常用值
        self.implicit_attack = ImplicitAttack(embedding_dim, hidden_dim=128)
        self.interference_attack = InterferenceAttack(embedding_dim, max_len)

        # 2. 从配置加载融合权重
        self.w_explicit = config.W_EXPLICIT
        self.w_implicit = config.W_IMPLICIT
        self.w_interference = config.W_INTERFERENCE

    def forward(self, embedded_input: torch.Tensor, watermark_payload: torch.Tensor) -> tuple[
        torch.Tensor, torch.Tensor]:
        """
        执行融合攻击的前向传播。

        Args:
            embedded_input (torch.Tensor):
                原始的词嵌入张量 (batch_size, seq_len, embedding_dim)。
            watermark_payload (torch.Tensor):
                要嵌入的水印比特流 (watermark_length,)。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
            - final_embeddings (torch.Tensor): 融合扰动后的最终嵌入。
            - state_value (torch.Tensor): 来自隐式攻击Critic的价值评估。
        """
        # 使用 .clone().detach() 来计算扰动增量，以防止梯度重复计算。
        # Actor/Critic的梯度将在主训练循环中单独计算和传播。
        original_embeddings = embedded_input.clone()

        # 1. 计算各层的扰动增量 (delta)
        # 显式层
        explicit_perturbed = self.explicit_attack(original_embeddings, watermark_payload)
        delta_explicit = explicit_perturbed - original_embeddings

        # 隐式层
        implicit_perturbed, state_value = self.implicit_attack(original_embeddings)
        delta_implicit = implicit_perturbed - original_embeddings

        # 干扰层
        interference_perturbed = self.interference_attack(original_embeddings)
        delta_interference = interference_perturbed - original_embeddings

        # 2. 加权融合扰动
        final_embeddings = (
                original_embeddings +
                self.w_explicit * delta_explicit +
                self.w_implicit * delta_implicit +
                self.w_interference * delta_interference
        )

        return final_embeddings, state_value


def find_nearest_tokens(perturbed_embeddings: torch.Tensor, embedding_matrix: nn.Parameter) -> torch.Tensor:
    """
    将扰动后的连续嵌入向量映射回最近的离散Token ID。

    这是从连续的嵌入空间生成实际对抗文本的关键步骤。

    Args:
        perturbed_embeddings (torch.Tensor):
            经过扰动的嵌入张量 (batch_size, seq_len, embedding_dim)。
        embedding_matrix (nn.Parameter):
            目标模型的完整词嵌入矩阵 (vocab_size, embedding_dim)。

    Returns:
        torch.Tensor: 对应的最近Token ID序列 (batch_size, seq_len)。
    """
    batch_size, seq_len, emb_dim = perturbed_embeddings.shape
    vocab_size = embedding_matrix.shape[0]

    # 将嵌入矩阵扩展以进行批处理计算
    # (vocab_size, emb_dim) -> (1, 1, vocab_size, emb_dim)
    # 这样可以和 (batch_size, seq_len, 1, emb_dim) 进行广播计算
    embedding_matrix_expanded = embedding_matrix.unsqueeze(0).unsqueeze(0)

    # 计算L2距离的平方
    # (batch_size, seq_len, 1, emb_dim) - (1, 1, vocab_size, emb_dim) -> (batch_size, seq_len, vocab_size, emb_dim)
    diff = perturbed_embeddings.unsqueeze(2) - embedding_matrix_expanded
    distances = torch.sum(diff ** 2, dim=3)
    # distances shape: (batch_size, seq_len, vocab_size)

    # 找到每个位置距离最小的token的索引
    nearest_token_ids = torch.argmin(distances, dim=2)
    # nearest_token_ids shape: (batch_size, seq_len)

    return nearest_token_ids


# --- 独立测试块 ---
if __name__ == '__main__':
    print("--- Running a self-test for FusionAttack and find_nearest_tokens ---")

    # 配置参数
    batch_size = config.BATCH_SIZE
    max_len = config.MAX_LEN
    embedding_dim = config.EMBEDDING_DIM
    watermark_payload = torch.tensor(config.WATERMARK_PAYLOAD).to(config.DEVICE)
    vocab_size = config.VOCAB_SIZE

    # 1. 实例化融合模块
    fusion_attacker = FusionAttack(
        embedding_dim=embedding_dim,
        watermark_length=len(watermark_payload),
        max_len=max_len
    ).to(config.DEVICE)
    print("\nFusionAttack module instantiated.")

    # 2. 创建虚拟输入
    dummy_embeddings = torch.randn(batch_size, max_len, embedding_dim).to(config.DEVICE)
    dummy_embedding_matrix = nn.Parameter(torch.randn(vocab_size, embedding_dim).to(config.DEVICE))
    print(
        f"\nCreated dummy inputs: embeddings shape {dummy_embeddings.shape}, matrix shape {dummy_embedding_matrix.shape}")

    # 3. 执行融合攻击前向传播
    final_embeddings, state_value = fusion_attacker(dummy_embeddings.clone(), watermark_payload)
    print("\nFusionAttack forward pass successful!")
    print(f" embeddings shape: {final_embeddings.shape}")
    print(f"State value shape: {state_value.shape}")
    assert final_embeddings.shape == dummy_embeddings.shape
    assert state_value.shape == (batch_size, 1)

    # 4. 测试 find_nearest_tokens 函数
    print("\nTesting find_nearest_tokens function...")
    # 为了验证，我们让一个扰动后的向量正好等于词汇表中的某个词向量
    final_embeddings[0, 5, :] = dummy_embedding_matrix[123, :]

    nearest_ids = find_nearest_tokens(final_embeddings, dummy_embedding_matrix)
    print("find_nearest_tokens execution successful!")
    print(f"Output token IDs shape: {nearest_ids.shape}")
    assert nearest_ids.shape == (batch_size, max_len)
    assert nearest_ids[0, 5].item() == 123, "Nearest token mapping failed verification!"
    print("Token mapping logic verified.")

    print("\n--- Self-test Passed! ---")

