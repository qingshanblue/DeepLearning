# 延迟类型注解检查
from __future__ import annotations

# 主要计算
import torch
import torch.nn as nn
import numpy as np

# 数据加载
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

# 其他辅助
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# 用户实现
from tools.configurator import Configurator
from nets.net import Net


def validate(
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
            desc="Eval ",
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
        tqdm_batch.close()
        data_loss_value /= total_nums
        data_acc_value /= total_nums
        return data_loss_value, data_acc_value


def visualize_results(
    nets: list[Net], configurator: Configurator, num_samples: int = 4
) -> None:
    """
    nets: 传入一个列表 [vgg_instance, res_instance, alex_instance]
    """
    # 1. 拿到数据
    images, labels = next(iter(configurator.test_loader))
    images, labels = images[:num_samples], labels[:num_samples]

    # 2. 绘图设置 (行: 样本, 列: 模型数量)
    fig, axes = plt.subplots(num_samples, len(nets), figsize=(15, 12), squeeze=False)

    for i in range(num_samples):
        img = images[i].permute(1, 2, 0).cpu().numpy()
        # 反归一化，否则图片颜色是错的
        img = np.clip(img * 0.5 + 0.5, 0, 1)

        for j, net_obj in enumerate(nets):
            net_obj.model.eval()
            with torch.no_grad():
                pred = net_obj.model(images[i].unsqueeze(0).to(configurator.device))
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


def test_full_performance(
    net: Net, data_loader: DataLoader, device: torch.device, num_classes=58
) -> tuple[float, dict[int, float], np.ndarray, np.ndarray, float, float]:
    net.model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        batch_tqdm = tqdm(
            data_loader,
            desc="mertics",
            mininterval=1,
            maxinterval=10,
            smoothing=0.1,
            position=0,
        )
        for images, labels in batch_tqdm:
            images = images.to(device)
            outputs = net.model(images)
            # 原模型训练只在loss里计算softmax，这里补充使用将输出转为 0~1 的概率
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
        batch_tqdm.close()
    # 将多个 batch 的数据拼接成大的矩阵
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    # 二值化
    y_one_hot = np.asarray(label_binarize(labels, classes=range(num_classes)))
    # 计算每一类的具体 AP
    all_aps = {}
    for i in range(num_classes):
        # 确保测试集中有这个类，否则 AP 无意义
        if np.sum(y_one_hot[:, i]) > 0:
            ap_i = average_precision_score(y_one_hot[:, i], probs[:, i])
            all_aps[i] = float(ap_i)
    # mAP(macro)
    mAP = float(np.mean(list(all_aps.values())))
    # 计算PR(micro)
    precision, recall, _ = precision_recall_curve(y_one_hot.ravel(), probs.ravel())
    loss, acc = validate(
        model=net.model, data_loader=data_loader, loss=net.loss, device=device
    )
    return mAP, all_aps, precision, recall, loss, acc


def run_test(net: Net, configurator: Configurator, model_name: str) -> dict:
    # 测试测试集
    try:
        net.model.load_state_dict(
            torch.load(f"checkpoints/best_{model_name}.pth", weights_only=True)
        )
    except FileNotFoundError:
        print("未找到最佳模型文件，请先训练模型。")
        return {}
    mAP, APs, precision, recall, loss, acc = test_full_performance(
        net=net,
        data_loader=configurator.test_loader,
        device=configurator.device,
        num_classes=configurator.num_classes,
    )
    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(20, 7), squeeze=False)
    fig.suptitle(f"Performance Evaluation: {model_name}", fontsize=16)
    # 绘制 PR 曲线
    axes[0].plot(recall, precision, label=f"mAP={mAP:.4f}", color="blue")
    axes[0].set_title(f"Precision-Recall Curve", fontsize=14)
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].legend(loc="lower left")
    axes[0].grid(True, linestyle="--", alpha=0.7)
    # 绘制 APs 柱状图
    classes = list(APs.keys())
    ap_values = list(APs.values())
    axes[1].bar(classes, ap_values, color="skyblue", edgecolor="navy")
    axes[1].set_title(f"Average Precision per Class", fontsize=14)
    axes[1].set_xlabel("Class ID")
    axes[1].set_ylabel("AP Score")
    # 输出mAP、loss、acc
    print(f"测试集的: mAP:{mAP:.4f}, loss:{loss:.4f}, acc:{acc:.4f}")
    return {
        "mAP": mAP,
        "APs": APs,
        "P": precision,
        "R": recall,
        "loss": loss,
        "acc": acc,
    }
