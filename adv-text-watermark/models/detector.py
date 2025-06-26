# adv_text_watermark/models/detector.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# 从配置文件中导入超参数
from utils import config


class Attention(nn.Module):
    """一个简单的加性注意力机制层"""

    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))

    def forward(self, hidden_states):
        # hidden_states: (batch_size, seq_len, hidden_dim)
        energy = torch.tanh(self.attn(hidden_states))
        # energy: (batch_size, seq_len, hidden_dim)

        # 计算注意力分数
        # energy.permute(0, 2, 1) -> (batch_size, hidden_dim, seq_len)
        attn_scores = torch.bmm(energy, self.v.unsqueeze(0).unsqueeze(2).repeat(energy.size(0), 1, 1)).squeeze(2)
        # attn_scores: (batch_size, seq_len)

        return F.softmax(attn_scores, dim=1)


class Detector(nn.Module):
    """
    水印检测模型。

    该模型接收目标模型的LSTM隐藏状态序列作为输入，
    并预测其中是否嵌入了水印，以及水印的内容是什么。
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 watermark_length: int,
                 lstm_layers: int = 1,
                 dropout: float = 0.3,
                 bidirectional: bool = True):
        """
        初始化检测器模型。

        Args:
            input_dim (int): 输入特征的维度。这通常是TargetModel的 (hidden_dim * num_directions)。
            hidden_dim (int): 检测器内部LSTM的隐藏层维度。
            watermark_length (int): 要检测的水印比特流的长度。
            lstm_layers (int): 检测器LSTM的层数。
            dropout (float): Dropout比例。
            bidirectional (bool): 是否使用双向LSTM。
        """
        super(Detector, self).__init__()

        num_directions = 2 if bidirectional else 1

        # 1. 使用LSTM处理来自TargetModel的隐藏状态序列
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        # 2. 注意力机制，用于从序列中聚合信息
        self.attention = Attention(hidden_dim * num_directions)

        # 3. Dropout层
        self.dropout = nn.Dropout(dropout)

        # 4. 输出层，将加权后的向量映射到水印长度
        self.fc = nn.Linear(hidden_dim * num_directions, watermark_length)

    def forward(self, target_model_hiddens: torch.Tensor) -> torch.Tensor:
        """
        定义检测器的前向传播。

        Args:
            target_model_hiddens (torch.Tensor):
                来自TargetModel的LSTM输出张量。
                Shape: (batch_size, seq_len, target_model_hidden_dim * num_directions)

        Returns:
            torch.Tensor: 预测的水印logits。Shape: (batch_size, watermark_length)
        """
        # target_model_hiddens shape: (batch, seq, feature_in)

        # 1. 通过检测器的LSTM
        lstm_out, _ = self.lstm(target_model_hiddens)
        # lstm_out shape: (batch, seq, detector_hidden_dim * num_directions)

        # 2. 计算注意力权重
        attn_weights = self.attention(lstm_out)
        # attn_weights shape: (batch, seq_len)

        # 3. 使用注意力权重对LSTM输出进行加权求和
        # attn_weights.unsqueeze(1) -> (batch, 1, seq_len)
        # torch.bmm -> (batch, 1, detector_hidden_dim * num_directions)
        context_vector = torch.bmm(attn_weights.unsqueeze(1), lstm_out)
        # context_vector.squeeze(1) -> (batch, detector_hidden_dim * num_directions)
        context_vector = context_vector.squeeze(1)

        # 4. Dropout 和 全连接层
        context_vector = self.dropout(context_vector)
        watermark_logits = self.fc(context_vector)
        # watermark_logits shape: (batch, watermark_length)

        return watermark_logits


# --- 独立测试块 ---
if __name__ == '__main__':
    print("--- Running a self-test for Detector Model ---")

    # TargetModel的输出维度
    target_model_output_dim = config.HIDDEN_DIM * 2

    # 实例化Detector
    detector = Detector(
        input_dim=target_model_output_dim,
        hidden_dim=128,  # 检测器可以有自己的隐藏维度
        watermark_length=config.WATERMARK_LENGTH,
    ).to(config.DEVICE)

    print("\nDetector Model Architecture:")
    print(detector)

    # 创建一个虚拟的输入batch，模拟TargetModel的输出
    batch_size = config.BATCH_SIZE
    max_len = config.MAX_LEN
    dummy_input = torch.randn(batch_size, max_len, target_model_output_dim).to(config.DEVICE)
    print(f"\nCreated a dummy input tensor of shape: {dummy_input.shape}")

    # 执行前向传播
    try:
        watermark_logits = detector(dummy_input)
        print("Forward pass successful!")
        print(
            f"Watermark logits output shape: {watermark_logits.shape} (Expected: [{batch_size}, {config.WATERMARK_LENGTH}])")
        assert watermark_logits.shape == (batch_size, config.WATERMARK_LENGTH)
        print("\n--- Self-test Passed! ---")
    except Exception as e:
        print(f"\n--- Self-test Failed! ---")
        print(f"Error: {e}")

