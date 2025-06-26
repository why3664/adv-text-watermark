# adv_text_watermark/data/watermark_utils.py

import torch
import torch.nn.functional as F
from typing import List

# 从配置文件中导入超参数
from utils import config


def calculate_watermark_loss(detector_logits: torch.Tensor, watermark_payload: torch.Tensor) -> torch.Tensor:
    """
    计算水印损失。

    该函数使用 Binary Cross Entropy with Logits Loss 来衡量
    检测器预测的水印与真实水印之间的差距。

    Args:
        detector_logits (torch.Tensor):
            来自Detector模型的原始输出 (logits)。
            Shape: (batch_size, watermark_length)
        watermark_payload (torch.Tensor):
            要嵌入的真实水印比特流。
            Shape: (watermark_length,)

    Returns:
        torch.Tensor: 一个标量 (scalar) 张量，代表该批次的平均水印损失。
    """
    batch_size = detector_logits.size(0)
    device = detector_logits.device

    # 1. 将水印载荷扩展到与批次大小匹配
    # 目标是让批次中的每个样本都包含相同的水印
    # watermark_payload.float() 转换为浮点型以匹配BCEloss的期望输入类型
    ground_truth_watermark = watermark_payload.float().unsqueeze(0).repeat(batch_size, 1).to(device)
    # ground_truth_watermark shape: (batch_size, watermark_length)

    # 2. 计算损失
    # BCEWithLogitsLoss 在数值上比 Sigmoid + BCELoss 更稳定
    # 它将 Sigmoid 层和 BCELoss 合二为一
    loss = F.binary_cross_entropy_with_logits(detector_logits, ground_truth_watermark)

    return loss


def check_watermark_accuracy(detector_logits: torch.Tensor, watermark_payload: torch.Tensor) -> float:
    """
    检查水印提取的准确率。

    计算在一批样本中，有多少个样本的水印被完美地解码出来。

    Args:
        detector_logits (torch.Tensor):
            来自Detector模型的原始输出 (logits)。
            Shape: (batch_size, watermark_length)
        watermark_payload (torch.Tensor):
            要嵌入的真实水印比特流。
            Shape: (watermark_length,)

    Returns:
        float: 完美匹配的准确率 (0.0 到 1.0 之间)。
    """
    device = detector_logits.device

    # 1. 从logits生成预测结果
    # 应用sigmoid将logits转换为概率，然后以0.5为阈值进行二值化
    predictions = (torch.sigmoid(detector_logits) > 0.5).long()
    # predictions shape: (batch_size, watermark_length)

    # 2. 准备真实标签
    ground_truth_watermark = watermark_payload.long().unsqueeze(0).repeat(predictions.size(0), 1).to(device)

    # 3. 检查每个样本是否完美匹配
    # torch.all(dim=1) 会检查每个样本的所有比特位是否都匹配
    correct_predictions = torch.all(predictions == ground_truth_watermark, dim=1)

    # 4. 计算准确率
    accuracy = correct_predictions.float().mean().item()

    return accuracy


# --- 独立测试块 ---
if __name__ == '__main__':
    print("--- Running a self-test for watermark utils ---")

    batch_size = config.BATCH_SIZE
    watermark_len = config.WATERMARK_LENGTH

    # 1. 创建虚拟的detector输出和水印载荷
    dummy_logits = torch.randn(batch_size, watermark_len).to(config.DEVICE)
    # 模拟一个比较好的预测，让一半的logits与payload符号相同
    payload_tensor = torch.tensor(config.WATERMARK_PAYLOAD).to(config.DEVICE)
    # 将payload从[0, 1]映射到[-1, 1]
    payload_as_sign = (payload_tensor - 0.5) * 2
    for i in range(batch_size // 2):
        dummy_logits[i] = payload_as_sign * torch.rand_like(payload_as_sign) * 2

    print(f"Created dummy logits of shape: {dummy_logits.shape}")
    print(f"Using watermark payload: {config.WATERMARK_PAYLOAD}")

    # 2. 测试损失计算
    try:
        loss = calculate_watermark_loss(dummy_logits, payload_tensor)
        print(f"\nCalculated watermark loss: {loss.item():.4f}")
        assert loss.item() > 0
        print("Loss calculation successful!")
    except Exception as e:
        print(f"Loss calculation failed: {e}")

    # 3. 测试准确率计算
    try:
        accuracy = check_watermark_accuracy(dummy_logits, payload_tensor)
        # 因为我们手动让一半的样本接近完美，所以准确率应该是0.5左右
        print(f"\nCalculated watermark accuracy (perfect match): {accuracy:.4f}")
        assert 0.0 <= accuracy <= 1.0
        print("Accuracy calculation successful!")
        print(f"(Expected accuracy for this test case is approx 0.5, result is {accuracy})")
    except Exception as e:
        print(f"Accuracy calculation failed: {e}")

    print("\n--- Self-test Passed! ---")

