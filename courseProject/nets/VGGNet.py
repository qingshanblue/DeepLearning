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

            # 统一使用 7x7 或 6x6 的 AdaptivePool，确保能接上巨型 FC 层
            self.pool = nn.AdaptiveAvgPool2d((7, 7))

            self.fc = nn.Sequential(
                # 512 * 7 * 7 = 25088 维输入
                nn.Linear(chns[3] * 7 * 7, feats_mid),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
                nn.Linear(feats_mid, feats_mid),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
                nn.Linear(feats_mid, num_classes),
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

    class Loss(Net.Loss):
        def __init__(self) -> None:
            super().__init__()
            self.criterion = nn.CrossEntropyLoss()

        def calc(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
            return self.criterion(y_pred, y_true)

        def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
            return self.calc(y_pred, y_true)

    class Optimizer(Net.Optimizer):
        def __init__(
            self, model: VGGNet.Model, lr: float = 0.01, weight_decay: float = 0.01
        ) -> None:
            super().__init__(model=model, lr=lr, weight_decay=weight_decay)
            self.optimizer = optim.AdamW(
                model.parameters(), lr, weight_decay=weight_decay
            )

        def step(self) -> None:
            self.optimizer.step()

        def zero_grad(self, set_to_none: bool = False) -> None:
            self.optimizer.zero_grad(set_to_none=set_to_none)
