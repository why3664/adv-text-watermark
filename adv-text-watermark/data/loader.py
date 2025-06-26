# adv_text_watermark/data/loader.py

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple

# 假设CharTokenizer与此文件在同一目录下或已正确安装
from .tokenizer import CharTokenizer


class TextDataset(Dataset):
    """
    文本数据集的PyTorch Dataset封装。

    继承自 torch.utils.data.Dataset，用于被 DataLoader 加载。
    """

    def __init__(self, texts: List[str], labels: List[int], tokenizer: CharTokenizer, max_len: int):
        """
        初始化数据集。

        Args:
            texts (List[str]): 所有文本数据的列表。
            labels (List[int]): 对应每个文本的标签列表。
            tokenizer (CharTokenizer): 已经实例化并构建好词汇表的分词器。
            max_len (int): 文本编码后的最大长度，会进行截断或填充。
        """
        if len(texts) != len(labels):
            raise ValueError("The number of texts and labels must be the same.")

        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_id = self.tokenizer.pad_token_id
        if self.pad_id == -1:
            raise ValueError("<PAD> token not found in the tokenizer's vocabulary.")

    def __len__(self) -> int:
        """返回数据集中的样本总数"""
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        获取单个样本。

        Args:
            idx (int): 样本索引。

        Returns:
            Tuple[torch.Tensor, torch.Tensor, int]:
            - text_tensor: 编码和填充/截断后的文本张量 (shape: [max_len])
            - label_tensor: 标签的张量 (shape: [])
            - length: 原始文本（编码后）的实际长度
        """
        text = self.texts[idx]
        label = self.labels[idx]

        # 1. 编码
        encoded_text = self.tokenizer.encode(text)
        original_len = len(encoded_text)

        # 2. 截断或填充
        if len(encoded_text) > self.max_len:
            encoded_text = encoded_text[:self.max_len]
        else:
            padding_needed = self.max_len - len(encoded_text)
            encoded_text.extend([self.pad_id] * padding_needed)

        # 3. 转换为Tensor
        text_tensor = torch.tensor(encoded_text, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return text_tensor, label_tensor, min(original_len, self.max_len)


def get_dataloader(
        texts: List[str],
        labels: List[int],
        tokenizer: CharTokenizer,
        max_len: int,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 0
) -> DataLoader:
    """
    创建一个DataLoader实例。

    Args:
        texts (List[str]): 文本列表。
        labels (List[int]): 标签列表。
        tokenizer (CharTokenizer): 分词器实例。
        max_len (int): 最大序列长度。
        batch_size (int): 批量大小。
        shuffle (bool): 是否在每个epoch开始时打乱数据。
        num_workers (int): 用于数据加载的子进程数。

    Returns:
        DataLoader: PyTorch数据加载器。
    """
    dataset = TextDataset(
        texts=texts,
        labels=labels,
        tokenizer=tokenizer,
        max_len=max_len
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

    return dataloader

