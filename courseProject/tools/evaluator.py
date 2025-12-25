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


def evaluation(
    model: nn.Module,
    loss: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
):
    model.eval()
    with torch.no_grad():
        total_nums = 0
        data_loss_value, data_acc_value = 0, 0
        tqdm_batch = tqdm(
            data_loader,
            desc="Eval",
            leave=False,
            position=1,
            mininterval=1,
            maxinterval=10,
            smoothing=0.1,
        )
        for features, labels in tqdm_batch:
            features, labels = features.to(device), labels.to(device)
            predict = model(features)
            loss_value = loss(predict, labels)

            total_nums += len(labels)
            data_loss_value += loss_value.item() * len(labels)
            data_acc_value += (predict.argmax(dim=1) == labels).sum().item()
            tqdm_batch.set_postfix_str(f"Current Batch Loss={loss_value.item():.4f}")
        data_loss_value /= total_nums
        data_acc_value /= total_nums
        return data_loss_value, data_acc_value
