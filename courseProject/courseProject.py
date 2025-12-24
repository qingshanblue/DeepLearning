# 主要计算
import torch
import torch.nn as nn
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
import tqdm.auto as tqdm
import matplotlib.pyplot as plt


class MyDataset(Dataset):
    def __init__(self, root_dir: str) -> None:
        # 初始化图片和标签的csv路径
        self.root_dir = root_dir
        self.img_dir = os.path.join(self.root_dir, "images")
        self.label_path = os.path.join(self.root_dir, "annotations.csv")

        # 初始化图片名称列表和样本(图片名称:str,编号:int)列表
        self.samples: list[tuple[str, int]] = []
        with open(self.label_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                file_name = row["file_name"]
                label = int(row["category"])
                self.samples.append((file_name, label))

    def __len__(self) -> int:
        return len(self.samples)  # 返回样本列表长度

    # 根据索引获取样本
    def __getitem__(self, index: int) -> tuple[Image.Image, torch.Tensor]:
        img_name, label = self.samples[index]

        # 获取图片路径并打开对应路径
        img_path = os.path.join(self.img_dir, img_name)
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            raise IOError(f"Error opening image: {img_path}")

        # 执行变换
        # image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

    # 创建自定义划分子集类，用于灵活使用transform
    class TransformSubset(Dataset):
        def __init__(
            self,
            dataset: Dataset,
            indices: list[int],
            transform: transforms.Compose | None = None,
        ) -> None:
            self.dataset = dataset
            self.indices = indices
            self.transform = transform

        def __len__(self) -> int:
            return len(self.indices)

        def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
            real_idx = self.indices[idx]
            image, label = self.dataset[real_idx]
            if self.transform:
                image = self.transform(image)
            return image, label

    def getDataLoaders(
        self,
        image_size: tuple[int, int] = (64, 64),
        train_proportion: float = 0.7,
        valid_proportion: float = 0.2,
        batch_size: int = 32,
        num_workers: int = 6,
        pin_memory: bool = True,
        prefetch_factor: int = 8,
        persistent_workers: bool = True,
        seed: int = 114514,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        # 定义 transform
        train_transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.RandomRotation(10),
                # transforms.RandomHorizontalFlip,  # 因为是交通信号标志，不做水平翻转增强
                transforms.ToTensor(),  # 图片经过 ToTensor() 后，像素值被映射到 [0,1]
                transforms.Normalize(
                    mean=[0.5] * 3, std=[0.5] * 3
                ),  # 归一化:output = (input-mean)/std;把数据分布中心化（均值变为 0);把数据方差缩放到相近尺度;有利于梯度下降收敛快、稳定
            ]
        )
        validTest_transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),  # 同上
            ]
        )

        # 计算子集索引
        full_dataset = self
        # 0:准备索引和标签
        indices = np.arange(len(full_dataset))
        # labels = np.array(
        #     [label for _, label in full_dataset.samples]
        # )
        # 因为数据集中标签9的类别只有1个数据，无法使用分层抽样，所以使用随机抽样
        # # 0
        # # 1：full -> train + temp
        # sss1 = StratifiedShuffleSplit(
        #     n_splits=1, test_size=(1 - train_proportion), random_state=seed
        # )
        # train_idx, temp_idx = next(sss1.split(indices, labels))
        # # 2：temp -> valid + test
        # temp_labels = labels[temp_idx]
        # temp_valid_size = valid_proportion / (1 - train_proportion)
        # sss2 = StratifiedShuffleSplit(
        #     n_splits=1, test_size=(1 - temp_valid_size), random_state=seed
        # )
        # valid_idx, test_idx = next(sss2.split(temp_idx, temp_labels))
        np.random.seed(seed)
        np.random.shuffle(indices)
        train_idx = list(indices[: int(len(indices) * train_proportion)])
        valid_idx = list(
            indices[
                int(len(indices) * train_proportion) : int(
                    len(indices) * (train_proportion + valid_proportion)
                )
            ]
        )
        test_idx = list(
            indices[int(len(indices) * (train_proportion + valid_proportion)) :]
        )

        # 创建子集
        train_dataset = MyDataset.TransformSubset(
            full_dataset,
            train_idx,
            transform=train_transform,
        )
        valid_dataset = MyDataset.TransformSubset(
            full_dataset,
            valid_idx,
            transform=validTest_transform,
        )
        test_dataset = MyDataset.TransformSubset(
            full_dataset,
            test_idx,
            transform=validTest_transform,
        )

        # 创建数据迭代器对象
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )

        print(
            f"总数据数：{len(full_dataset)}, 训练集大小：{len(train_dataset)}, 验证集大小：{len(valid_dataset)}, 测试集大小：{len(test_dataset)}"
        )

        return train_loader, valid_loader, test_loader


if __name__ == "__main__":
    # 初始化随机种子，保证每次运行代码时数据集的划分一致
    seed = 114514
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed(seed)
    else:
        device = torch.device("cpu")
    # 加载数据
    chinaTrafficSignData = MyDataset(root_dir="./courseProject/data")
    train_loader, valid_loader, test_loader = chinaTrafficSignData.getDataLoaders(
        image_size=(64, 64),
        train_proportion=0.7,
        valid_proportion=0.2,
        batch_size=128,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
        prefetch_factor=2,
        persistent_workers=True,
        seed=seed,
    )
    from residualNet import ResidualNet

    residualNet = ResidualNet()
    model_residualNet = residualNet.Model(nums_classes=58).to(device=device)
    loss_residualNet = residualNet.Loss()
    optimizer_residualNet = residualNet.Optimizer(
        model=model_residualNet, lr=1e-3, weight_decay=1e-4
    )
    (
        train_loss_residualNet,
        train_acc_residualNet,
        valid_loss_residualNet,
        valid_acc_residualNet,
    ) = residualNet.train(
        model=model_residualNet,
        loss=loss_residualNet,
        optimizer=optimizer_residualNet,
        train_loader=train_loader,
        valid_loader=valid_loader,
        num_epoches=32,
        device=device,
        accumulation_steps=1,
    )
    figure, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].plot(train_loss_residualNet, label="train_loss")
    axes[0].plot(valid_loss_residualNet, label="valid_loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[1].plot(train_acc_residualNet, label="train_acc")
    axes[1].plot(valid_acc_residualNet, label="valid_acc")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].legend()
    plt.tight_layout()
    plt.show()
