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
    net: Net,
    configurator: Configurator,
) -> tuple[list[float], list[float], list[float], list[float]]:
    train_loss_list, train_acc_list = [], []
    valid_loss_list, valid_acc_list = [], []
    tqdm_epoch = tqdm(
        range(configurator.num_epochs),
        desc="Epochs",
        leave=True,
        position=0,
        mininterval=1,
        maxinterval=10,
        smoothing=0.1,
    )
    best_acc = 0.0
    for epoch in tqdm_epoch:
        net.model.train()
        total_nums = 0
        train_loss, train_acc = 0.0, 0.0
        # 在循环开始前清零
        net.optimizer.zero_grad()
        tqdm_batch = tqdm(
            configurator.train_loader,
            desc=f"Train",
            leave=False,
            position=1,
            mininterval=1,
            maxinterval=10,
            smoothing=0.1,
        )
        for i, (images, labels) in enumerate(tqdm_batch):
            images, labels = images.to(configurator.device), labels.to(
                configurator.device
            )
            # 前向计算
            predict = net.model(images)
            raw_loss = net.loss(predict, labels)  # 保存原始 loss 用于统计
            # 1. 梯度缩放与反向传播
            loss_accumulated = (
                raw_loss / configurator.accumulation_steps
            )  # 缩放loss以实现梯度累积
            loss_accumulated.backward()
            # 2. 达到步数更新梯度
            if (i + 1) % configurator.accumulation_steps == 0:
                net.optimizer.step()  # 累积步数达到时更新参数
                net.optimizer.zero_grad()  # 清零梯度准备下一轮累积
            # 3. 统计数据使用原始 loss (raw_loss)
            total_nums += len(labels)
            train_loss += raw_loss.item() * len(labels)  # 使用原始loss保证统计准确性
            train_acc += (predict.argmax(dim=1) == labels).sum().item()
            # 更新当前batch信息
            tqdm_batch.set_postfix_str(f"Current Batch Loss={raw_loss.item():.4f}")
        tqdm_batch.close()
        # 4. 处理 Epoch 末尾未达步数的残余梯度
        if len(configurator.train_loader) % configurator.accumulation_steps != 0:
            net.optimizer.step()  # 处理最后未达到累积步数的残余梯度
            net.optimizer.zero_grad()
        # 计算平均值并输出返回
        train_loss /= total_nums
        train_acc /= total_nums
        valid_loss, valid_acc = validate(
            model=net.model,
            loss=net.loss,
            data_loader=configurator.valid_loader,
            device=configurator.device,
        )
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
            os.makedirs(  # 确保checkpoints目录存在
                f"checkpoints/{net.name}/{configurator.desc}", exist_ok=True
            )
            torch.save(
                net.model.state_dict(),
                f"checkpoints/{net.name}/{configurator.desc}/best_model.pth",
            )
            tqdm_epoch.write(
                f"Epoch {epoch}: 新最佳模型准确度 = {best_acc:.4f} 已保存!"
            )
    tqdm_epoch.close()
    return train_loss_list, train_acc_list, valid_loss_list, valid_acc_list


def run_train(net: Net, configurator: Configurator) -> dict:
    print(f"正在进行:训练{net.name}模型...")
    # 执行模型训练，获取训练和验证过程中的损失值和准确率
    (
        train_loss_template,
        train_acc_template,
        valid_loss_template,
        valid_acc_template,
    ) = train(
        net=net,
        configurator=configurator,
    )
    # 创建子图用于可视化训练过程，左侧显示损失曲线，右侧显示准确率曲线
    figure, axes = plt.subplots(1, 2, figsize=(10, 5))
    figure.suptitle(f"{net.name} Training and Validation Results")
    # 绘制训练和验证损失曲线，便于观察模型是否过拟合
    axes[0].plot(train_loss_template, label="train_loss")
    axes[0].plot(valid_loss_template, label="valid_loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    # 绘制训练和验证准确率曲线，评估模型性能变化趋势
    axes[1].plot(train_acc_template, label="train_acc")
    axes[1].plot(valid_acc_template, label="valid_acc")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].legend()
    # 保存图片
    os.makedirs(  # 确保figures目录存在
        f"figures/{net.name}/{configurator.desc}", exist_ok=True
    )
    figure.savefig(f"figures/{net.name}/{configurator.desc}/train&valid_loss&acc.png")
    # 返回训练指标和训练后的模型，便于后续评估或保存
    return {
        "train_loss": train_loss_template,
        "train_acc": train_acc_template,
        "valid_loss": valid_loss_template,
        "valid_acc": valid_acc_template,
        "net": net,
    }
