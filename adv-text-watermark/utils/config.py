# adv_text_watermark/utils/config.py

import os
import torch

# --- 路径配置 (Path Configuration) ---
# 使用 os.path.abspath(__file__) 获取当前文件的绝对路径
# os.path.dirname() 用于获取该路径的目录
# 两次 dirname() 就可以得到项目的根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 基于根目录构建其他核心目录的路径
DATA_DIR = os.path.join(BASE_DIR, 'data/')
MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'saved_models/')
LOG_DIR = os.path.join(BASE_DIR, 'logs/')
RESULTS_DIR = os.path.join(BASE_DIR, 'results/')

# 确保这些目录存在
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# 预训练模型或数据集的特定文件路径 (根据需要修改)
# 例如: TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'your_dataset/train.csv')
# 例如: VOCAB_PATH = os.path.join(DATA_DIR, 'vocab.json')


# --- 设备配置 (Device Configuration) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# --- 数据与分词器超参数 (Data & Tokenizer Hyperparameters) ---
MAX_LEN = 128          # 文本序列最大长度
BATCH_SIZE = 64        # 批量大小
MIN_FREQ = 2           # 构建词汇表时字符的最小频率
SPECIAL_TOKENS = ['<PAD>', '<UNK>'] # 特殊Token


# --- 目标模型超参数 (Target Model Hyperparameters) ---
# 假设我们使用一个简单的Bi-LSTM进行文本分类
# VOCAB_SIZE 将在分词器构建后动态确定，这里只是一个占位符
VOCAB_SIZE = 5000      # 词汇表大小 (将在运行时被覆盖)
EMBEDDING_DIM = 128    # 词嵌入维度
HIDDEN_DIM = 256       # LSTM隐藏层维度
LSTM_LAYERS = 2        # LSTM层数
NUM_CLASSES = 2        # 目标任务的类别数量 (例如，情感分析中的正面/负面)
DROPOUT = 0.5          # Dropout比例


# --- 水印参数 (Watermark Parameters) ---
WATERMARK_PAYLOAD = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] # 想要嵌入的水印比特流
WATERMARK_LENGTH = len(WATERMARK_PAYLOAD)           # 水印长度
GAMMA = 0.1            # 水印损失在总损失中的平衡因子 (loss = task_loss + GAMMA * watermark_loss)


# --- 攻击模块超参数 (Attack Module Hyperparameters) ---
# 隐式层攻击 (Actor-Critic)
LR_ACTOR = 1e-4        # Actor (策略网络) 的学习率
LR_CRITIC = 1e-3       # Critic (价值网络) 的学习率
DISCOUNT_FACTOR = 0.99 # RL中的折扣因子 (gamma)

# 攻击融合权重 (Fusion Weights)
# 这些权重用于融合三层攻击的结果
W_EXPLICIT = 1.0       # 显式攻击的权重
W_IMPLICIT = 1.0       # 隐式攻击的权重
W_INTERFERENCE = 0.5   # 目标模型干扰的权重


# --- 训练与评估超参数 (Training & Evaluation Hyperparameters) ---
EPOCHS = 20                     # 总训练轮次
LEARNING_RATE = 1e-3            # 目标模型或整体攻击的初始学习率
LOG_INTERVAL = 50               # 每隔多少个batch打印一次训练日志
