# adv_text_watermark/attacks/explicit_attack.py

import torch
import torch.nn as nn

# 从配置文件中导入超参数
from utils import config


class ExplicitAttack(nn.Module):
    """
    显式层攻击 (Explicit Layer Attack)。

    该模块在词嵌入层面对输入进行微小的、有针对性的扰动，
    以直接编码水印信息。
    """

    def __init__(self, embedding_dim: int, watermark_length: int):
        """
        初始化显式攻击层。

        Args:
            embedding_dim (int): 目标模型词嵌入的维度。扰动向量的维度必须与之匹配。
            watermark_length (int): 水印的长度。
        """
        super(ExplicitAttack, self).__init__()
        self.embedding_dim = embedding_dim
        self.watermark_length = watermark_length

        # --- 核心可学习参数 ---
        # 创建一个可学习的扰动向量矩阵。
        # 每一行代表一个水印比特位为'1'时，要施加的特定扰动。
        # self.perturbation_vectors[k] 是要嵌入第k位水印时使用的扰动向量。
        self.perturbation_vectors = nn.Parameter(
            torch.randn(self.watermark_length, self.embedding_dim) * 0.01
        )

    def forward(self, embedded_input: torch.Tensor, watermark_payload: torch.Tensor) -> torch.Tensor:
        """
        在前向传播中应用显式扰动。

        Args:
            embedded_input (torch.Tensor):
                来自TargetModel的原始词嵌入张量。
                Shape: (batch_size, seq_len, embedding_dim)
            watermark_payload (torch.Tensor):
                要嵌入的真实水印比特流。
                Shape: (watermark_length,)

        Returns:
            torch.Tensor: 经过显式扰动后的词嵌入张量。
                          Shape: (batch_size, seq_len, embedding_dim)
        """
        batch_size, seq_len, _ = embedded_input.shape
        device = embedded_input.device

        # 1. 创建一个与输入同形的零矩阵，用于存放要施加的扰动
        perturbations_to_apply = torch.zeros_like(embedded_input)

        # 2. 根据水印载荷，填充扰动矩阵
        # 我们采用一个简单的策略：将第k个水印比特嵌入到序列的第k个位置。
        # 注意：这里的长度是 min(watermark_length, seq_len)，防止水印比序列还长
        num_bits_to_embed = min(self.watermark_length, seq_len)

        for k in range(num_bits_to_embed):
            # 如果第k个水印比特为1，则在该位置施加扰动
            if watermark_payload[k] == 1:
                # 获取第k个扰动向量
                k_th_perturbation = self.perturbation_vectors[k]
                # 将其应用到批次中所有样本的第k个时间步上
                perturbations_to_apply[:, k, :] = k_th_perturbation

        # 3. 将计算好的扰动加到原始的嵌入上
        perturbed_embeddings = embedded_input + perturbations_to_apply

        return perturbed_embeddings


# --- 独立测试块 ---
if __name__ == '__main__':
    print("--- Running a self-test for ExplicitAttack ---")

    # 配置参数
    batch_size = config.BATCH_SIZE
    max_len = config.MAX_LEN
    embedding_dim = config.EMBEDDING_DIM
    watermark_payload = torch.tensor(config.WATERMARK_PAYLOAD).to(config.DEVICE)

    # 1. 实例化模型
    explicit_attacker = ExplicitAttack(
        embedding_dim=embedding_dim,
        watermark_length=len(watermark_payload)
    ).to(config.DEVICE)
    print("\nModel instantiated:")
    print(explicit_attacker)
    print(f"Learnable perturbation vectors shape: {explicit_attacker.perturbation_vectors.shape}")

    # 2. 创建虚拟输入
    dummy_embeddings = torch.randn(batch_size, max_len, embedding_dim).to(config.DEVICE)
    print(f"\nCreated dummy embedding tensor of shape: {dummy_embeddings.shape}")

    # 3. 执行前向传播
    perturbed_embeddings = explicit_attacker(dummy_embeddings.clone(), watermark_payload)
    print("Forward pass successful!")
    print(f"Output shape: {perturbed_embeddings.shape}")
    assert perturbed_embeddings.shape == dummy_embeddings.shape

    # 4. 验证逻辑
    print("\nVerifying perturbation logic...")
    # 找到第一个为1的水印位
    first_one_bit_idx = -1
    for i, bit in enumerate(watermark_payload):
        if bit == 1:
            first_one_bit_idx = i
            break

    if first_one_bit_idx != -1:
        # 检查被扰动位置的嵌入是否已改变
        original_vec = dummy_embeddings[0, first_one_bit_idx, :]
        perturbed_vec = perturbed_embeddings[0, first_one_bit_idx, :]
        diff = torch.sum(original_vec - perturbed_vec).item()
        print(
            f"Checking position for watermark bit '{watermark_payload[first_one_bit_idx]}' at index {first_one_bit_idx}:")
        assert diff != 0, "Embedding at perturbed position should have changed!"
        print("-> Embedding correctly modified.")

        # 检查未被扰动的位置（水印位为0，或超出水印长度）
        # 找到第一个为0的水印位
        first_zero_bit_idx = -1
        for i, bit in enumerate(watermark_payload):
            if bit == 0:
                first_zero_bit_idx = i
                break

        if first_zero_bit_idx != -1:
            original_vec_zero = dummy_embeddings[0, first_zero_bit_idx, :]
            perturbed_vec_zero = perturbed_embeddings[0, first_zero_bit_idx, :]
            diff_zero = torch.sum(original_vec_zero - perturbed_vec_zero).item()
            print(
                f"Checking position for watermark bit '{watermark_payload[first_zero_bit_idx]}' at index {first_zero_bit_idx}:")
            assert diff_zero == 0, "Embedding at non-perturbed position should be unchanged!"
            print("-> Embedding correctly unchanged.")

    print("\n--- Self-test Passed! ---")