# 延迟类型注解检查
from __future__ import annotations

# 计算
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import seaborn as sns

# 数据
import os
from torch.utils.data import DataLoader
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
)
from sklearn.preprocessing import label_binarize

# 辅助
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# 用户
from tools.configurator import Configurator
from nets.net import Net


# 验证函数，用于评估模型在验证集上的表现
def validate(
    model: Net.Model,
    loss: Net.Loss,
    data_loader: DataLoader,
    device: torch.device,
):
    # 将模型设置为评估模式
    model.eval()
    with torch.no_grad():
        total_nums = 0
        data_loss_value, data_acc_value = 0, 0
        # 使用tqdm显示进度条
        tqdm_batch = tqdm(
            data_loader,
            desc="Eval ",
            leave=False,
            position=1,
            mininterval=1,
            maxinterval=10,
            smoothing=0.1,
        )
        for features, labels in tqdm_batch:
            features, labels = features.to(device), labels.to(device)
            # 获取模型预测
            predict = model(features)
            # 计算损失
            loss_value = loss(predict, labels)

            total_nums += len(labels)
            data_loss_value += loss_value.item() * len(labels)
            data_acc_value += (predict.argmax(dim=1) == labels).sum().item()
            # 更新进度条描述
            tqdm_batch.set_postfix_str(f"Current Batch Loss={loss_value.item():.4f}")
        tqdm_batch.close()
        # 计算平均损失和准确率
        data_loss_value /= total_nums
        data_acc_value /= total_nums
        return data_loss_value, data_acc_value


# 可视化函数，用于对比显示不同模型的具体图片的预测结果
def visualize_results(
    nets: list[Net], configurator: Configurator, num_samples: int = 4
) -> None:
    # 1. 拿到数据
    images, labels = next(iter(configurator.test_loader))
    images, labels = images[:num_samples], labels[:num_samples]

    # 2. 绘图设置 (行: 样本, 列: 模型数量)
    figure, axes = plt.subplots(num_samples, len(nets), figsize=(15, 12), squeeze=False)

    for i in range(num_samples):
        img = images[i].permute(1, 2, 0).cpu().numpy()
        # 反归一化，否则图片颜色是错的
        img = np.clip(img * 0.5 + 0.5, 0, 1)

        # 搬移图片到显存参与后续计算
        input_img = images[i].unsqueeze(0).to(configurator.device)
        for j, net_obj in enumerate(nets):
            # 搬移模型到显存参与后续计算
            net_obj.model.to(device=configurator.device)
            net_obj.model.eval()
            with torch.no_grad():
                pred = net_obj.model(input_img)
                pred_idx = pred.argmax(dim=1).item()

            ax = axes[i, j]
            ax.imshow(img)

            # 预测正确用绿色标题，错误用红色
            color = "green" if pred_idx == labels[i].item() else "red"
            ax.set_title(
                f"Model: {net_obj.__class__.__name__}\nPred: {pred_idx} | GT: {labels[i].item()}",
                color=color,
                fontsize=9,
            )
            ax.axis("off")
            # 腾出显存
            net_obj.model.to(device="cpu")
    # 保存图片
    listName = [f"{i.name}" for i in nets]
    os.makedirs(  # 确保figures目录存在
        f"figures/compare/{configurator.desc}", exist_ok=True
    )
    figure.savefig(f"figures/compare/{configurator.desc}/{listName}.png")


def plot_confusion_matrix(all_labels, all_preds, save_path, model_name, num_classes=58):
    cm = confusion_matrix(
        all_labels,
        all_preds,
        labels=range(num_classes),
        normalize="true",  # 加上归一化，方便观察
    )

    plt.figure(figsize=(22, 18))  # 大画布，容纳 58 个标签

    # 绘制热力图
    sns.heatmap(
        cm,
        annot=False,
        cmap="Blues",
        xticklabels=[str(i) for i in range(num_classes)],  # 转换为字符串列表
        yticklabels=[str(i) for i in range(num_classes)],  # 转换为字符串列表
    )
    plt.title(
        f"Normalized Confusion Matrix: {model_name} (Full 58 Classes)", fontsize=20
    )
    plt.xlabel("Predicted Labels", fontsize=15)
    plt.ylabel("True Labels", fontsize=15)

    plt.savefig(save_path, bbox_inches="tight")


def test_full_performance(net: Net, configurator: Configurator) -> tuple[
    float,
    dict[int, float],
    np.ndarray,
    np.ndarray,
    float,
    float,
    np.ndarray,
    np.ndarray,
]:
    # 设置模型为评估模式，关闭dropout和batchnorm的训练行为
    net.model.eval()
    all_probs = []
    all_labels = []
    # 禁用梯度计算以节省内存和加速推理
    with torch.no_grad():
        batch_tqdm = tqdm(
            configurator.test_loader,
            desc="mertics",
            mininterval=1,
            maxinterval=10,
            smoothing=0.1,
            position=0,
        )
        for images, labels in batch_tqdm:
            images = images.to(configurator.device)
            outputs = net.model(images)
            # 模型输出是logits，需要通过softmax转换为概率分布
            # 因为原模型只在loss计算时使用softmax，推理时需要手动转换
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
        batch_tqdm.close()
    # 将所有batch的预测概率和真实标签合并为完整的数据集
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    # 将标签转换为one-hot编码格式，便于计算每个类别的AP
    y_one_hot = np.asarray(
        label_binarize(labels, classes=range(configurator.num_classes))
    )
    # 计算每个类别的平均精度(AP)
    all_aps = {}
    for i in range(configurator.num_classes):
        # 只计算测试集中实际存在的类别的AP，避免除零错误
        if np.sum(y_one_hot[:, i]) > 0:
            ap_i = average_precision_score(y_one_hot[:, i], probs[:, i])
            all_aps[i] = float(ap_i)
    # 计算所有类别的平均精度均值(mAP)，使用macro平均方式
    mAP = float(np.mean(list(all_aps.values())))
    # 计算micro平均的精确率-召回率曲线
    # 将所有类别视为一个二分类问题来计算PR曲线
    precision, recall, _ = precision_recall_curve(y_one_hot.ravel(), probs.ravel())
    # 验证模型在测试集上的loss和准确率
    loss, acc = validate(
        model=net.model,
        loss=net.loss,
        data_loader=configurator.test_loader,
        device=configurator.device,
    )
    return mAP, all_aps, precision, recall, loss, acc, labels, probs


def run_test(net: Net, configurator: Configurator) -> dict:
    # 测试测试集
    try:
        # 加载训练过程中保存的最佳模型权重进行评估
        net.model.load_state_dict(
            torch.load(
                f"checkpoints/{net.name}/{configurator.desc}/best_model.pth",
                weights_only=True,
            )
        )
    except FileNotFoundError:
        print("未找到最佳模型文件，请先训练模型。")
        return {}
    # 在测试集上计算完整的性能指标，包括mAP、各类别AP、精确率、召回率等
    mAP, APs, precision, recall, loss, acc, labels, probs = test_full_performance(
        net=net,
        configurator=configurator,
    )
    # 绘PR图
    figure, axes = plt.subplots(1, 2, figsize=(20, 7))
    figure.suptitle(
        f"Performance Test: {net.name}\nmAP:{mAP:.4f}, loss:{loss:.4f}, acc:{acc:.4f}",
        fontsize=16,
    )
    # 绘制 PR 曲线 - 展示模型在不同召回率下的精确率表现
    axes[0].plot(recall, precision, label=f"mAP={mAP:.4f}", color="blue")
    axes[0].set_title(f"Precision-Recall Curve", fontsize=14)
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].legend(loc="lower left")
    axes[0].grid(True, linestyle="--", alpha=0.7)
    # 绘制 APs 柱状图 - 直观展示每个类别的平均精度
    classes = list(APs.keys())
    ap_values = list(APs.values())
    axes[1].bar(classes, ap_values, color="skyblue", edgecolor="navy")
    axes[1].set_title(f"Average Precision per Class", fontsize=14)
    axes[1].set_xlabel("Class ID")
    axes[1].set_ylabel("AP Score")
    # 保存图片
    os.makedirs(  # 确保figures目录存在
        f"figures/{net.name}/{configurator.desc}", exist_ok=True
    )
    figure.savefig(f"figures/{net.name}/{configurator.desc}/PR&AP.png")
    # 输出关键性能指标供用户快速查看模型表现
    print(f"测试集的: mAP:{mAP:.4f}, loss:{loss:.4f}, acc:{acc:.4f}")

    # 绘制混淆矩阵
    pred_labels = np.argmax(probs, axis=1)
    plot_confusion_matrix(
        all_labels=labels,
        all_preds=pred_labels,
        save_path=f"figures/{net.name}/{configurator.desc}/confusionMatrix.png",
        model_name=net.name,
    )
    return {
        "mAP": mAP,
        "APs": APs,
        "P": precision,
        "R": recall,
        "loss": loss,
        "acc": acc,
    }
