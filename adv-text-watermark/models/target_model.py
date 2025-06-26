# adv_text_watermark/models/target_model.py

import torch
import torch.nn as nn
from typing import Tuple

# 我们将从配置文件中导入超参数，但这只是为了在独立测试此文件时方便。
# 在主流程中，参数会由训练脚本直接传入。
from utils import config


class TargetModel(nn.Module):
    """
    目标文本分类模型 (一个具体的Bi-LSTM实现)。

    这个模型执行主要的文本分类任务（例如，情感分析）。
    我们的攻击目标是在不显著降低其性能的前提下，向其处理的文本中嵌入水印。
    """

    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 hidden_dim: int,
                 num_classes: int,
                 lstm_layers: int,
                 dropout: float,
                 bidirectional: bool = True):
        """
        初始化模型层。

        Args:
            vocab_size (int): 词汇表的大小。
            embedding_dim (int): 词嵌入的维度。
            hidden_dim (int): LSTM 隐藏层的维度。
            num_classes (int): 输出类别的数量。
            lstm_layers (int): LSTM 的层数。
            dropout (float): Dropout 的比例。
            bidirectional (bool): 是否使用双向LSTM。
        """
        super(TargetModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers
        self.num_directions = 2 if bidirectional else 1

        # 1. 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=config.SPECIAL_TOKENS.index('<PAD>'))

        # 2. LSTM层
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,  # 输入和输出张量格式为 (batch, seq, feature)
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0
        )

        # 3. Dropout层
        self.dropout = nn.Dropout(dropout)

        # 4. 全连接输出层
        # 输入维度是 hidden_dim * 2 (因为是双向)
        self.fc = nn.Linear(hidden_dim * self.num_directions, num_classes)

    def forward(self, text: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        定义模型的前向传播。

        Args:
            text (torch.Tensor): 输入的文本张量 (batch_size, seq_len)。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
            - logits (torch.Tensor): 最终的分类预测 logits (batch_size, num_classes)。
            - lstm_output (torch.Tensor): LSTM层的输出，用于后续的水印检测 (batch_size, seq_len, hidden_dim * num_directions)。
        """
        # text shape: (batch_size, seq_len)

        # 1. 嵌入
        embedded = self.embedding(text)
        # embedded shape: (batch_size, seq_len, embedding_dim)

        # 2. LSTM
        # 初始化LSTM的隐藏状态和细胞状态
        h0 = torch.zeros(self.lstm_layers * self.num_directions, text.size(0), self.hidden_dim).to(config.DEVICE)
        c0 = torch.zeros(self.lstm_layers * self.num_directions, text.size(0), self.hidden_dim).to(config.DEVICE)

        # lstm_output 包含了每个时间步的隐藏状态
        # _ (h_n, c_n) 是最后一个时间步的隐藏状态和细胞状态
        lstm_output, _ = self.lstm(embedded, (h0, c0))
        # lstm_output shape: (batch_size, seq_len, hidden_dim * num_directions)

        # 3. 从LSTM输出中提取用于分类的特征
        # 我们将使用最后一个时间步的隐藏状态进行分类
        # 由于 batch_first=True, lstm_output 是 (batch, seq, feature)
        # 我们可以简单地取最后一个时间步的输出，但更稳健的做法是处理填充
        # 这里为了简化，我们先使用最后一个时间步的输出
        # lstm_output[:, -1, :] shape: (batch_size, hidden_dim * num_directions)
        final_hidden_state = lstm_output[:, -1, :]

        # 4. Dropout 和 全连接层
        dropped_out = self.dropout(final_hidden_state)
        logits = self.fc(dropped_out)
        # logits shape: (batch_size, num_classes)

        return logits, lstm_output


# --- 独立测试块 ---
if __name__ == '__main__':
    print("--- Running a self-test for TargetModel ---")

    # 使用config中的参数来实例化模型
    model = TargetModel(
        vocab_size=config.VOCAB_SIZE,
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_classes=config.NUM_CLASSES,
        lstm_layers=config.LSTM_LAYERS,
        dropout=config.DROPOUT
    ).to(config.DEVICE)

    print("\nModel Architecture:")
    print(model)

    # 创建一个虚拟的输入batch
    batch_size = config.BATCH_SIZE
    max_len = config.MAX_LEN
    dummy_input = torch.randint(0, config.VOCAB_SIZE, (batch_size, max_len)).to(config.DEVICE)
    print(f"\nCreated a dummy input tensor of shape: {dummy_input.shape}")

    # 执行前向传播
    try:
        logits, lstm_hidden_states = model(dummy_input)
        print("Forward pass successful!")
        print(f"Logits output shape: {logits.shape} (Expected: [{batch_size}, {config.NUM_CLASSES}])")
        print(
            f"LSTM hidden states output shape: {lstm_hidden_states.shape} (Expected: [{batch_size}, {max_len}, {config.HIDDEN_DIM * 2}])")
        assert logits.shape == (batch_size, config.NUM_CLASSES)
        assert lstm_hidden_states.shape == (batch_size, max_len, config.HIDDEN_DIM * 2)
        print("\n--- Self-test Passed! ---")
    except Exception as e:
        print(f"\n--- Self-test Failed! ---")
        print(f"Error: {e}")

