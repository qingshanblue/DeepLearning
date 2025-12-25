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
from configurator import MyParams
from evaluator import evaluation
from nets.residualNet import ResidualNet
from nets.AlexNet import AlexNet
from nets.VGGNet import VGGNet


def train(  # TODO 实现PR曲线和AP、mAP
    model: nn.Module,
    loss: nn.Module,
    optimizer: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    num_epoches: int,
    device: torch.device,
    accumulation_steps: int,
):
    print(f"batch积累倍率:{accumulation_steps}")
    train_loss_list, train_acc_list = [], []
    valid_loss_list, valid_acc_list = [], []
    tqdm_epoch = tqdm(
        range(num_epoches),
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
            train_loss += raw_loss.item() * len(labels)  # 这里用 raw_loss 保证统计准确
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
        valid_loss, valid_acc = evaluation(model, loss, valid_loader, device)
        tqdm_epoch.set_postfix_str(
            f"TrainLoss={train_loss:.4f} ValidLoss={valid_loss:.4f}   TrainAcc={train_acc:.4f} ValidAcc={valid_acc:.4f}",
        )
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        valid_loss_list.append(valid_loss)
        valid_acc_list.append(valid_acc)
    return train_loss_list, train_acc_list, valid_loss_list, valid_acc_list


# NOTE 1 ResidualNet:
def train_residualNet(
    params: MyParams,
) -> tuple[list[float], list[float], list[float], list[float]]:
    print("正在进行:训练ResidualNet模型...")
    # 创建模型
    residualNet = ResidualNet()
    model_residualNet = residualNet.Model(num_classes=params.num_classes).to(
        device=params.device
    )
    loss_residualNet = residualNet.Loss()
    optimizer_residualNet = residualNet.Optimizer(
        model=model_residualNet,
        lr=params.learning_rate,
        weight_decay=params.weight_decay,
    )
    # 训练
    (
        train_loss_residualNet,
        train_acc_residualNet,
        valid_loss_residualNet,
        valid_acc_residualNet,
    ) = train(
        model=model_residualNet,
        loss=loss_residualNet,
        optimizer=optimizer_residualNet,
        train_loader=params.train_loader,
        valid_loader=params.valid_loader,
        num_epoches=params.num_epochs,
        device=params.device,
        accumulation_steps=params.accumulation_steps,
    )
    # 绘制结果图
    figure_residualNet, axes_residualNet = plt.subplots(1, 2, figsize=(10, 5))
    axes_residualNet[0].plot(train_loss_residualNet, label="train_loss")
    axes_residualNet[0].plot(valid_loss_residualNet, label="valid_loss")
    axes_residualNet[0].set_xlabel("epoch")
    axes_residualNet[0].set_ylabel("loss")
    axes_residualNet[0].legend()
    axes_residualNet[1].plot(train_acc_residualNet, label="train_acc")
    axes_residualNet[1].plot(valid_acc_residualNet, label="valid_acc")
    axes_residualNet[1].set_xlabel("epoch")
    axes_residualNet[1].set_ylabel("accuracy")
    axes_residualNet[1].legend()
    figure_residualNet.suptitle("ResidualNet Training and Validation Metrics")
    # 返回
    return (
        train_loss_residualNet,
        train_acc_residualNet,
        valid_loss_residualNet,
        valid_acc_residualNet,
    )


# NOTE 2 AlexNet:
def train_alexNet(
    params: MyParams,
) -> tuple[list[float], list[float], list[float], list[float]]:
    print("正在进行:训练AlexNet模型...")
    # 创建模型
    model_alexNet = AlexNet.Model(
        chns_in=3,
        chns_mid=[96, 256, 384, 384],
        ker_size=[11, 5, 3, 3],
        padding=[2, 2, 1, 1],
        stride=[4, 1, 1, 1],
        poolKer_size=[3, 3],
        poolStride=[2, 2],
        feats_mid=[4096, 4096],
        feats_out=58,
        dropout_rate=[0.4, 0.4],
    ).to(params.device)
    loss_alexNet = AlexNet.Loss()
    optimizer_alexNet = AlexNet.Optimizer(
        model=model_alexNet, lr=params.learning_rate, weight_decay=params.weight_decay
    )
    # 训练
    train_loss_alexNet, train_acc_alexNet, valid_loss_alexNet, valid_acc_alexNet = (
        train(
            model=model_alexNet,
            loss=loss_alexNet,
            optimizer=optimizer_alexNet,
            train_loader=params.train_loader,
            valid_loader=params.valid_loader,
            num_epoches=params.num_epochs,
            device=params.device,
            accumulation_steps=params.accumulation_steps,
        )
    )
    # 绘制结果图
    figure_alexNet, axes_alexNet = plt.subplots(1, 2, figsize=(10, 5))
    axes_alexNet[0].plot(train_loss_alexNet, label="train_loss")
    axes_alexNet[0].plot(valid_loss_alexNet, label="valid_loss")
    axes_alexNet[0].set_xlabel("epoch")
    axes_alexNet[0].set_ylabel("loss")
    axes_alexNet[0].legend()
    axes_alexNet[1].plot(train_acc_alexNet, label="train_acc")
    axes_alexNet[1].plot(valid_acc_alexNet, label="valid_acc")
    axes_alexNet[1].set_xlabel("epoch")
    axes_alexNet[1].set_ylabel("accuracy")
    axes_alexNet[1].legend()
    figure_alexNet.suptitle("AlexNet Training and Validation Metrics")
    # 返回
    return train_loss_alexNet, train_acc_alexNet, valid_loss_alexNet, valid_acc_alexNet

    # NOTE 3 VGGNet:


def train_VGGNet(
    params: MyParams,
) -> tuple[list[float], list[float], list[float], list[float]]:
    print("正在进行:训练VGGNet模型...")
    # 创建模型
    model_VGGNet = VGGNet.Model(
        chns_in=3,
        chns_base=64,
        feats_base=1024,
        dropout_rate=0.5,
        ker_size=3,
        nums_classes=params.num_classes,
        padding=1,
        stride=1,
    ).to(params.device)
    loss_VGGNet = VGGNet.Loss()
    optimizer_VGGNet = VGGNet.Optimizer(
        model=model_VGGNet, lr=params.learning_rate, weight_decay=params.weight_decay
    )
    # 训练
    train_loss_VGGNet, train_acc_VGGNet, valid_loss_VGGNet, valid_acc_VGGNet = train(
        model=model_VGGNet,
        loss=loss_VGGNet,
        optimizer=optimizer_VGGNet,
        train_loader=params.train_loader,
        valid_loader=params.valid_loader,
        num_epoches=params.num_epochs,
        device=params.device,
        accumulation_steps=params.accumulation_steps,
    )
    # 绘制结果图
    figure_VGGNet, axes_VGGNet = plt.subplots(1, 2, figsize=(10, 5))
    axes_VGGNet[0].plot(train_loss_VGGNet, label="train_loss")
    axes_VGGNet[0].plot(valid_loss_VGGNet, label="valid_loss")
    axes_VGGNet[0].set_xlabel("epoch")
    axes_VGGNet[0].set_ylabel("loss")
    axes_VGGNet[0].legend()
    axes_VGGNet[1].plot(train_acc_VGGNet, label="train_acc")
    axes_VGGNet[1].plot(valid_acc_VGGNet, label="valid_acc")
    axes_VGGNet[1].set_xlabel("epoch")
    axes_VGGNet[1].set_ylabel("accuracy")
    axes_VGGNet[1].legend()
    figure_VGGNet.suptitle("VGGNet Training and Validation Metrics")
    # 返回
    return train_loss_VGGNet, train_acc_VGGNet, valid_loss_VGGNet, valid_acc_VGGNet
