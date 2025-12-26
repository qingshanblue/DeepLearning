# 延迟类型注解检查
from __future__ import annotations

# 主要计算
import torch
import torch.nn as nn
import torch.optim as optim

# 用户实现
from nets.net import Net


class VGGNet(Net):
    class Model(Net.Model):
        def __init__(
            self,
            chns_in: int = 3,
            num_classes: int = 58,
            # VGG 经典的通道增长序列: 64 -> 128 -> 256 -> 512
            chns: list[int] = [64, 128, 256, 512],
            feats_mid: int = 4096,  # 保持和 AlexNet 一致的 4096 维
            dropout_rate: float = 0.5,
        ) -> None:
            super().__init__()

            # 卷积特征提取部分: 4个VGG块，每块包含2层卷积+1层池化
            # 每个块的通道数按chns列表递增，逐步提取更高级的特征
            self.conv = nn.Sequential(
                # Block 1: 64通道 (2层卷积)
                nn.Conv2d(chns_in, chns[0], 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(chns[0], chns[0], 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                # Block 2: 128通道 (2层卷积)
                nn.Conv2d(chns[0], chns[1], 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(chns[1], chns[1], 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                # Block 3: 256通道 (2层卷积)
                nn.Conv2d(chns[1], chns[2], 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(chns[2], chns[2], 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                # Block 4: 512通道 (2层卷积)
                nn.Conv2d(chns[2], chns[3], 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(chns[3], chns[3], 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
            )

            # 自适应平均池化: 统一输出尺寸为7x7，便于连接全连接层
            # 无论输入图像尺寸如何变化，都能保证后续FC层输入维度一致
            self.pool = nn.AdaptiveAvgPool2d((7, 7))

            # 全连接分类器: 3层结构(2个隐藏层+1个输出层)
            # 使用较大的中间维度(4096)来增强模型表达能力
            self.fc = nn.Sequential(
                # 512 * 7 * 7 = 25088 维输入
                nn.Linear(chns[3] * 7 * 7, feats_mid),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),  # 防止过拟合
                nn.Linear(feats_mid, feats_mid),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),  # 防止过拟合
                nn.Linear(feats_mid, num_classes),
            )
            self._initialize_weights()

        def _initialize_weights(self):
            # 权重初始化: 根据不同层类型采用不同的初始化策略
            # 卷积层使用Kaiming初始化，适合ReLU激活函数
            # 全连接层使用较小标准差的正态分布初始化
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
            # 前向传播流程: 卷积特征提取 -> 自适应池化 -> 展平 -> 全连接分类
            X = self.conv(X)
            X = self.pool(X)
            X = torch.flatten(X, 1)  # 展平为(batch_size, -1)格式
            X = self.fc(X)
            return X

    class Loss(Net.Loss):
        def __init__(self) -> None:
            super().__init__()
            # 分类任务使用交叉熵损失函数
            self.criterion = nn.CrossEntropyLoss()

        def calc(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
            # 计算预测值与真实标签之间的交叉熵损失
            return self.criterion(y_pred, y_true)

        def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
            return self.calc(y_pred, y_true)

    class Optimizer(Net.Optimizer):
        def __init__(
            self, model: VGGNet.Model, lr: float = 0.01, weight_decay: float = 0.01
        ) -> None:
            super().__init__(model=model, lr=lr, weight_decay=weight_decay)
            # 使用AdamW优化器: 结合Adam的优点和权重衰减正则化
            self.optimizer = optim.AdamW(
                model.parameters(), lr, weight_decay=weight_decay
            )

        def step(self) -> None:
            # 执行一步参数更新
            self.optimizer.step()

        def zero_grad(self, set_to_none: bool = False) -> None:
            # 清零梯度缓存，准备下一轮反向传播
            self.optimizer.zero_grad(set_to_none=set_to_none)
