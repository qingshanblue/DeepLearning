# 延迟类型注解检查
from __future__ import annotations

# 主要计算
import torch
import torch.nn as nn

# 数据加载
import os
from torch.utils.data import DataLoader

# 其他辅助
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# 用户实现
from tools.configurator import Configurator
from tools.evaluator import validate
from nets.net import Net


def train(
    model: Net.Model,
    loss: Net.Loss,
    optimizer: Net.Optimizer,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    num_epoches: int,
    device: torch.device,
    accumulation_steps: int,
    model_name: str = "model",  # 传入模型名称，用于保存对应的pth文件
) -> tuple[list[float], list[float], list[float], list[float]]:
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
    best_acc = 0.0
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
        tqdm_batch.close()
        # 4. 处理 Epoch 末尾未达步数的残余梯度
        if len(train_loader) % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()
        # 计算平均值并输出返回
        train_loss /= total_nums
        train_acc /= total_nums
        valid_loss, valid_acc = validate(model, loss, valid_loader, device)
        tqdm_epoch.set_postfix_str(
            f"TrainLoss={train_loss:.4f} ValidLoss={valid_loss:.4f}   TrainAcc={train_acc:.4f} ValidAcc={valid_acc:.4f}",
        )
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        valid_loss_list.append(valid_loss)
        valid_acc_list.append(valid_acc)
        # 保存最佳模型
        if valid_acc > best_acc:
            best_acc = valid_acc
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/best_{model_name}.pth")
            tqdm_epoch.write(
                f"Epoch {epoch}: 新最佳模型准确度 = {best_acc:.4f} 已保存!"
            )
    tqdm_epoch.close()
    return train_loss_list, train_acc_list, valid_loss_list, valid_acc_list


def run_train(net: Net, configurator: Configurator, model_name: str) -> dict:
    print(f"正在进行:训练{model_name}模型...")
    # 训练
    (
        train_loss_template,
        train_acc_template,
        valid_loss_template,
        valid_acc_template,
    ) = train(
        model=net.model,
        loss=net.loss,
        optimizer=net.optimizer,
        train_loader=configurator.train_loader,
        valid_loader=configurator.valid_loader,
        num_epoches=configurator.num_epochs,
        device=configurator.device,
        accumulation_steps=configurator.accumulation_steps,
        model_name=model_name,
    )
    # 绘制结果图
    figure, axes = plt.subplots(1, 2, figsize=(10, 5))
    figure.suptitle(f"{model_name} Training and Validation Results")
    axes[0].plot(train_loss_template, label="train_loss")
    axes[0].plot(valid_loss_template, label="valid_loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[1].plot(train_acc_template, label="train_acc")
    axes[1].plot(valid_acc_template, label="valid_acc")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].legend()
    # 返回
    return {
        "train_loss": train_loss_template,
        "train_acc": train_acc_template,
        "valid_loss": valid_loss_template,
        "valid_acc": valid_acc_template,
        "net": net,
    }
