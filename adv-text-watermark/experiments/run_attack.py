# adv_text_watermark/experiments/run_attack.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time

# 导入所有必要的模块
from utils import config
from data.tokenizer import Tokenizer
from data.dataset import TextDataset  # 假设我们有一个标准的TextDataset
from data.watermark_utils import calculate_watermark_loss, check_watermark_accuracy
from models.target_model import TargetModel
from models.detector import Detector
from attacks.fusion import FusionAttack, find_nearest_tokens


def train_epoch(target_model, detector, fusion_attacker, dataloader, optimizers, watermark_payload):
    """一个训练轮次的完整流程"""
    target_model.train()
    detector.train()
    fusion_attacker.train()

    # 从optimizers字典中解包
    optimizer_main = optimizers['main']
    optimizer_actor = optimizers['actor']
    optimizer_critic = optimizers['critic']

    total_task_loss = 0
    total_watermark_loss = 0
    total_accuracy = 0

    start_time = time.time()

    for i, (texts, labels) in enumerate(dataloader):
        texts, labels = texts.to(config.DEVICE), labels.to(config.DEVICE)

        # --- 核心训练步骤 ---

        # 1. 获取原始嵌入
        original_embeddings = target_model.embedding(texts)

        # 2. 通过融合攻击模块生成扰动嵌入和Critic的评估值
        perturbed_embeddings, state_value = fusion_attacker(original_embeddings, watermark_payload)

        # 3. 目标模型前向传播 (使用被扰动过的嵌入)
        # 注意：这里我们需要修改TargetModel的forward，使其能接受嵌入作为输入
        # 为了简化，我们直接将扰动后的嵌入送入LSTM层
        lstm_output, _ = target_model.lstm(perturbed_embeddings)
        final_hidden_state = lstm_output[:, -1, :]
        logits = target_model.fc(target_model.dropout(final_hidden_state))

        # 4. 检测器前向传播
        detector_logits = detector(lstm_output.detach())  # 从主计算图中分离，防止梯度干扰

        # 5. 计算损失
        task_loss_fn = nn.CrossEntropyLoss()
        task_loss = task_loss_fn(logits, labels)

        watermark_loss = calculate_watermark_loss(detector_logits, watermark_payload)

        total_loss = task_loss + config.GAMMA * watermark_loss

        # --- Actor-Critic 优化 ---
        # a. 更新 Critic
        optimizer_critic.zero_grad()
        # Critic的目标是预测total_loss
        critic_loss_fn = nn.MSELoss()
        # 使用 .detach() 来确保这部分的梯度只用于更新Critic
        critic_loss = critic_loss_fn(state_value, total_loss.detach().unsqueeze(1))
        critic_loss.backward()
        optimizer_critic.step()

        # b. 更新 Actor
        optimizer_actor.zero_grad()
        # Actor的目标是最小化Critic的评估值
        actor_loss = state_value.mean()
        actor_loss.backward()
        optimizer_actor.step()

        # --- 主模型优化 ---
        optimizer_main.zero_grad()
        # 主优化器负责更新 TargetModel, Detector, 和 FusionAttack 中非AC的部分
        total_loss.backward()
        optimizer_main.step()

        # --- 记录统计数据 ---
        total_task_loss += task_loss.item()
        total_watermark_loss += watermark_loss.item()
        total_accuracy += check_watermark_accuracy(detector_logits, watermark_payload)

        if (i + 1) % config.LOG_INTERVAL == 0:
            elapsed = time.time() - start_time
            print(f"  Batch {i + 1}/{len(dataloader)} | "
                  f"Task Loss: {task_loss.item():.4f} | "
                  f"WM Loss: {watermark_loss.item():.4f} | "
                  f"Time: {elapsed:.2f}s")
            start_time = time.time()

    avg_task_loss = total_task_loss / len(dataloader)
    avg_wm_loss = total_watermark_loss / len(dataloader)
    avg_wm_accuracy = total_accuracy / len(dataloader)

    return avg_task_loss, avg_wm_loss, avg_wm_accuracy


def evaluate(target_model, detector, fusion_attacker, dataloader, watermark_payload):
    """评估模型性能和水印效果"""
    target_model.eval()
    detector.eval()
    fusion_attacker.eval()

    total_task_loss = 0
    total_wm_accuracy = 0
    correct_task_preds = 0
    total_samples = 0

    task_loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for texts, labels in dataloader:
            texts, labels = texts.to(config.DEVICE), labels.to(config.DEVICE)

            original_embeddings = target_model.embedding(texts)
            perturbed_embeddings, _ = fusion_attacker(original_embeddings, watermark_payload)

            lstm_output, _ = target_model.lstm(perturbed_embeddings)
            final_hidden_state = lstm_output[:, -1, :]
            logits = target_model.fc(target_model.dropout(final_hidden_state))

            detector_logits = detector(lstm_output)

            total_task_loss += task_loss_fn(logits, labels).item()
            total_wm_accuracy += check_watermark_accuracy(detector_logits, watermark_payload)

            preds = torch.argmax(logits, dim=1)
            correct_task_preds += (preds == labels).sum().item()
            total_samples += labels.size(0)

    avg_task_loss = total_task_loss / len(dataloader)
    task_accuracy = correct_task_preds / total_samples
    avg_wm_accuracy = total_wm_accuracy / len(dataloader)

    return avg_task_loss, task_accuracy, avg_wm_accuracy


def main():
    """主执行函数"""
    print("--- Starting Adversarial Text Watermarking Experiment ---")

    # 1. 准备数据和分词器 (此处为伪代码，需要替换为真实数据)
    print("1. Loading data and tokenizer...")



    # ------------------ 替换从这里开始 ------------------
    # 假设您的文本数据放在项目根目录的 data/ 文件夹下
    # 您需要准备 train.csv 和 val.csv 文件
    # 每个csv文件应包含两列：'text' 和 'label'

    from data.dataset import TextDataset  # 确保导入
    from data.tokenizer import Tokenizer  # 确保导入

    print("1. Loading data and building tokenizer...")
    # 实例化分词器
    tokenizer = Tokenizer(config.MIN_FREQ)

    # 从训练数据构建词汇表
    # 您需要提供您的训练数据文件路径
    train_file_path = 'data/train.csv'  # <-- 替换为您的训练文件名
    tokenizer.fit_on_file(train_file_path)

    # 动态更新配置文件中的词汇表大小
    config.VOCAB_SIZE = tokenizer.vocab_size
    print(f"Vocabulary Size: {config.VOCAB_SIZE}")

    # 创建数据集实例
    val_file_path = 'data/val.csv'  # <-- 替换为您的验证文件名
    train_dataset = TextDataset(train_file_path, tokenizer, max_len=config.MAX_LEN)
    val_dataset = TextDataset(val_file_path, tokenizer, max_len=config.MAX_LEN)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)

    # 2. 初始化模型
    print("2. Initializing models...")
    target_model = TargetModel(vocab_size=config.VOCAB_SIZE, embedding_dim=config.EMBEDDING_DIM,
                               hidden_dim=config.HIDDEN_DIM, num_classes=config.NUM_CLASSES,
                               lstm_layers=config.LSTM_LAYERS, dropout=config.DROPOUT).to(config.DEVICE)
    detector = Detector(input_dim=config.HIDDEN_DIM * 2, hidden_dim=128, watermark_length=config.WATERMARK_LENGTH).to(
        config.DEVICE)
    fusion_attacker = FusionAttack(embedding_dim=config.EMBEDDING_DIM, watermark_length=config.WATERMARK_LENGTH,
                                   max_len=config.MAX_LEN).to(config.DEVICE)

    # 3. 设置优化器
    print("3. Setting up optimizers...")
    # 主优化器，负责更新目标模型、检测器和攻击模块（除AC外）
    main_params = list(target_model.parameters()) + list(detector.parameters()) + \
                  list(fusion_attacker.explicit_attack.parameters()) + \
                  list(fusion_attacker.interference_attack.parameters())
    optimizer_main = optim.Adam(main_params, lr=config.LEARNING_RATE)

    # Actor-Critic 的独立优化器
    optimizer_actor = optim.Adam(fusion_attacker.implicit_attack.actor.parameters(), lr=config.LR_ACTOR)
    optimizer_critic = optim.Adam(fusion_attacker.implicit_attack.critic.parameters(), lr=config.LR_CRITIC)

    optimizers = {
        'main': optimizer_main,
        'actor': optimizer_actor,
        'critic': optimizer_critic
    }

    watermark_payload = torch.tensor(config.WATERMARK_PAYLOAD).to(config.DEVICE)
    best_wm_acc = 0.0

    # 4. 训练与评估循环
    print("\n--- Starting Training ---")
    for epoch in range(1, config.EPOCHS + 1):
        print(f"\nEpoch {epoch}/{config.EPOCHS}")

        train_task_loss, train_wm_loss, train_wm_acc = train_epoch(
            target_model, detector, fusion_attacker, train_loader, optimizers, watermark_payload
        )
        print(f"Epoch {epoch} Summary (Train):")
        print(
            f"  Avg Task Loss: {train_task_loss:.4f} | Avg WM Loss: {train_wm_loss:.4f} | Avg WM Accuracy: {train_wm_acc:.4f}")

        val_task_loss, val_task_acc, val_wm_acc = evaluate(
            target_model, detector, fusion_attacker, val_loader, watermark_payload
        )
        print(f"Epoch {epoch} Summary (Validation):")
        print(f"  Task Loss: {val_task_loss:.4f} | Task Accuracy: {val_task_acc:.4f} | WM Accuracy: {val_wm_acc:.4f}")

        # 保存最佳模型 (基于水印准确率)
        if val_wm_acc > best_wm_acc:
            best_wm_acc = val_wm_acc
            print(f"  New best watermark accuracy! Saving model...")
            torch.save(target_model.state_dict(), os.path.join(config.MODEL_SAVE_DIR, 'best_target_model.pt'))
            torch.save(detector.state_dict(), os.path.join(config.MODEL_SAVE_DIR, 'best_detector.pt'))
            torch.save(fusion_attacker.state_dict(), os.path.join(config.MODEL_SAVE_DIR, 'best_fusion_attacker.pt'))

    print("\n--- Training Finished ---")


if __name__ == '__main__':
    main()

