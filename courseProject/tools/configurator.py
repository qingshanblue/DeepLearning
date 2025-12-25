# 延迟类型注解检查
from __future__ import annotations

# 主要计算
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# 数据加载
import os
import csv
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit

# 其他辅助
from typing import cast
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# 用户实现
from dataloader import MyDataset


class MyParams:
    def __init__(
        self,
        num_classes: int = 58,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        num_epochs: int = 32,
        accumulation_steps: int = 1,
        seed: int = 114514,
    ) -> None:
        # 定义训练参数
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.accumulation_steps = accumulation_steps
        # 初始化随机种子,初始化运行设备
        self.seed = seed
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            torch.cuda.manual_seed(self.seed)
        else:
            self.device = torch.device("cpu")
        # 定义数据迭代器参数
        root_dir = "./data"
        images_size = (64, 64)
        train_proportion = 0.7
        valid_proportion = 0.2
        batch_size = 128
        num_workers = 6
        prefetch_factor = 2
        # 加载数据
        chinaTrafficSignData = MyDataset(root_dir=root_dir)
        self.train_loader, self.valid_loader, self.test_loader = (
            chinaTrafficSignData.getDataLoaders(
                image_size=images_size,
                train_proportion=train_proportion,
                valid_proportion=valid_proportion,
                batch_size=batch_size,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                persistent_workers=True,
                pin_memory=(self.device.type == "cuda"),
                seed=self.seed,
            )
        )
