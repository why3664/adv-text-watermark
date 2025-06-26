# adv_text_watermark/data/tokenizer.py

import json
from collections import Counter
from typing import List, Dict


class CharTokenizer:
    """
    字符级分词器 (Character-level Tokenizer)

    功能:
    1. 从文本语料库中构建词汇表。
    2. 将文本字符串编码 (encode) 为整数ID序列。
    3. 将整数ID序列解码 (decode) 回文本字符串。
    """

    def __init__(self, texts: List[str] = None, min_freq: int = 1, special_tokens: List[str] = None):
        """
        初始化分词器。

        Args:
            texts (List[str]): 用于构建词汇表的文本列表。
            min_freq (int): 字符进入词汇表的最小频率。
            special_tokens (List[str]): 特殊标记列表，如 ['<PAD>', '<UNK>', '<SOS>', '<EOS>']。
                                       '<PAD>' (padding) 和 '<UNK>' (unknown) 是强烈建议的。
        """
        if special_tokens is None:
            # 定义默认的特殊tokens
            self.special_tokens = ['<PAD>', '<UNK>']
        else:
            self.special_tokens = special_tokens

        self.char2idx: Dict[str, int] = {}
        self.idx2char: Dict[int, str] = {}

        if texts:
            self._build_vocab(texts, min_freq)

    def _build_vocab(self, texts: List[str], min_freq: int):
        """
        根据文本语料构建词汇表。

        Args:
            texts (List[str]): 文本列表。
            min_freq (int): 最小词频阈值。
        """
        # 1. 添加特殊tokens到词汇表
        for token in self.special_tokens:
            if token not in self.char2idx:
                idx = len(self.char2idx)
                self.char2idx[token] = idx
                self.idx2char[idx] = token

        # 2. 统计所有字符频率
        char_counts = Counter("".join(texts))

        # 3. 添加满足最小频率的字符到词汇表
        for char, count in char_counts.items():
            if count >= min_freq and char not in self.char2idx:
                idx = len(self.char2idx)
                self.char2idx[char] = idx
                self.idx2char[idx] = char

    @property
    def vocab_size(self) -> int:
        """返回词汇表大小"""
        return len(self.char2idx)

    @property
    def unk_token_id(self) -> int:
        """返回未知token的ID"""
        return self.char2idx.get('<UNK>', -1)

    @property
    def pad_token_id(self) -> int:
        """返回填充token的ID"""
        return self.char2idx.get('<PAD>', -1)

    def encode(self, text: str) -> List[int]:
        """
        将单个字符串编码为ID列表。

        Args:
            text (str): 输入字符串。

        Returns:
            List[int]: 编码后的ID列表。
        """
        # 如果<UNK>不存在，则遇到未知字符会抛出异常
        unk_id = self.unk_token_id
        if unk_id == -1:
            return [self.char2idx[char] for char in text]
        return [self.char2idx.get(char, unk_id) for char in text]

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        将ID列表解码为字符串。

        Args:
            ids (List[int]): ID列表。
            skip_special_tokens (bool): 是否在解码时跳过特殊tokens。

        Returns:
            str: 解码后的字符串。
        """
        tokens = []
        for i in ids:
            char = self.idx2char.get(i)
            if char:
                if skip_special_tokens and char in self.special_tokens:
                    continue
                tokens.append(char)
        return "".join(tokens)

    def save_vocab(self, file_path: str):
        """
        将词汇表保存到文件。

        Args:
            file_path (str): 保存路径 (建议为.json格式)。
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.char2idx, f, ensure_ascii=False, indent=4)
        print(f"Vocabulary of size {self.vocab_size} saved to {file_path}")

    def load_vocab(self, file_path: str):
        """
        从文件加载词汇表。

        Args:
            file_path (str): 词汇表文件路径。
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            self.char2idx = json.load(f)

        # 重建 idx2char 映射
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}
        print(f"Vocabulary of size {self.vocab_size} loaded from {file_path}")

