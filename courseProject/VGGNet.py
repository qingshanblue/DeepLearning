from __future__ import annotations  # 延迟注解解析

# 主要计算
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd

# 数据加载
import os
import csv
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# from sklearn.model_selection import StratifiedShuffleSplit

# 其他辅助
from typing import cast
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


class VGGNet:
    class Model(nn.Module):
        def __init__(
            self,
            chns_in: int,
            chns_base: int,
            feats_base: int,
            nums_classes: int,
            dropout_rate: float,
            ker_size: int,
            padding: int = 0,
            stride: int = 1,
        ) -> None:
            super().__init__()
            self.chns_in = chns_in
            self.chns_base = chns_base
            self.feats_base = feats_base
            self.nums_classes = nums_classes
            self.dropout_rate = dropout_rate
            self.ker_size = ker_size
            self.padding = padding
            self.stride = stride

            self.conv = nn.Sequential(
                # cov1
                nn.Conv2d(
                    chns_in,
                    chns_base * 1,
                    kernel_size=ker_size,
                    padding=self.padding,
                    stride=self.stride,
                ),
                nn.BatchNorm2d(chns_base * 1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                # cov2
                nn.Conv2d(
                    chns_base * 1,
                    chns_base * 2,
                    kernel_size=ker_size,
                    padding=self.padding,
                    stride=self.stride,
                ),
                nn.BatchNorm2d(chns_base * 2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                # conv3
                nn.Conv2d(
                    chns_base * 2,
                    chns_base * 4,
                    kernel_size=ker_size,
                    padding=self.padding,
                    stride=self.stride,
                ),
                nn.BatchNorm2d(chns_base * 4),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                # conv4
                nn.Conv2d(
                    chns_base * 4,
                    chns_base * 8,
                    kernel_size=ker_size,
                    padding=self.padding,
                    stride=self.stride,
                ),
                nn.BatchNorm2d(chns_base * 8),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
            )
            self.pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)))
            self.fc = nn.Sequential(
                # fc1
                nn.Linear(self.chns_base * 8, self.feats_base),
                # nn.BatchNorm1d(self.feats_base),
                nn.ReLU(inplace=True),
                nn.Dropout(p=self.dropout_rate),
                # # fc2
                # nn.Linear(self.feats_base, self.feats_base),
                # nn.BatchNorm1d(self.feats_base),
                # nn.ReLU(inplace=True),
                # nn.Dropout(p=self.dropout_rate),
                # fc3
                nn.Linear(self.feats_base, self.nums_classes),
            )
            self._initialize_weights()

        def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

        def forward(self, X: torch.Tensor) -> torch.Tensor:
            X = self.conv(X)
            X = self.pool(X)
            X = torch.flatten(X, 1)
            X = self.fc(X)
            return X

    class Loss:
        def __init__(self) -> None:
            self.criterion = nn.CrossEntropyLoss()

        def calc(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
            return self.criterion(y_pred, y_true)

        def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
            return self.calc(y_pred, y_true)

    class Optimizer:
        def __init__(
            self, model: VGGNet.Model, lr: float = 0.01, weight_decay: float = 0.01
        ) -> None:
            self.optimizer = optim.AdamW(
                model.parameters(), lr, weight_decay=weight_decay
            )

        def step(self) -> None:
            self.optimizer.step()

        def zero_grad(self) -> None:
            self.optimizer.zero_grad()

    @staticmethod
    def evaluation(
        model: VGGNet.Model,
        loss: Loss,
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
                tqdm_batch.set_postfix_str(
                    f"Current Batch Loss={loss_value.item():.4f}"
                )
            data_loss_value /= total_nums
            data_acc_value /= total_nums
            return data_loss_value, data_acc_value

    @staticmethod
    def train(
        model: VGGNet.Model,
        loss: Loss,
        optimizer: Optimizer,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        nnum_epoches: int,
        device: torch.device,
        accumulation_steps: int,
    ):
        print(f"batch积累倍率:{accumulation_steps}")
        train_loss_list, train_acc_list = [], []
        valid_loss_list, valid_acc_list = [], []
        tqdm_epoch = tqdm(
            range(nnum_epoches),
            desc="Epochs",
            leave=True,
            position=0,
            mininterval=1,
            maxinterval=10,
            smoothing=0.1,
        )
        for epoch in tqdm_epoch:
            model.train()
            total_nums = 0
            train_loss, train_acc = 0.0, 0.0
            # 在循环开始前清零
            optimizer.zero_grad()
            tqdm_batch = tqdm(
                train_loader,
                desc=f"Train",
                leave=False,
                position=1,
                mininterval=1,
                maxinterval=10,
                smoothing=0.1,
            )
            for i, (images, labels) in enumerate(tqdm_batch):
                images, labels = images.to(device), labels.to(device)
                # 前向计算
                predict = model(images)
                raw_loss = loss(predict, labels)  # 保存原始 loss 用于统计
                # 1. 梯度缩放与反向传播
                loss_accumulated = raw_loss / accumulation_steps
                loss_accumulated.backward()
                # 2. 达到步数更新梯度
                if (i + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                # 3. 统计数据使用原始 loss (raw_loss)
                total_nums += len(labels)
                train_loss += raw_loss.item() * len(
                    labels
                )  # 这里用 raw_loss 保证统计准确
                train_acc += (predict.argmax(dim=1) == labels).sum().item()
                # 更新当前batch信息
                tqdm_batch.set_postfix_str(f"Current Batch Loss={raw_loss.item():.4f}")
            # 4. 处理 Epoch 末尾未达步数的残余梯度
            if len(train_loader) % accumulation_steps != 0:
                optimizer.step()
                optimizer.zero_grad()
            # 计算平均值并输出返回
            train_loss /= total_nums
            train_acc /= total_nums
            valid_loss, valid_acc = VGGNet.evaluation(model, loss, valid_loader, device)
            tqdm_epoch.set_postfix_str(
                f"TrainLoss={train_loss:.4f} ValidLoss={valid_loss:.4f}   TrainAcc={train_acc:.4f} ValidAcc={valid_acc:.4f}",
            )
            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            valid_loss_list.append(valid_loss)
            valid_acc_list.append(valid_acc)
        return train_loss_list, train_acc_list, valid_loss_list, valid_acc_list
