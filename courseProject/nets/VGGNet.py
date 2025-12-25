# 延迟类型注解检查
from __future__ import annotations

# 主要计算
import torch
import torch.nn as nn
import torch.optim as optim


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

    class Loss(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.criterion = nn.CrossEntropyLoss()

        def calc(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
            return self.criterion(y_pred, y_true)

        def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
            return self.calc(y_pred, y_true)

    class Optimizer(nn.Module):
        def __init__(
            self, model: VGGNet.Model, lr: float = 0.01, weight_decay: float = 0.01
        ) -> None:
            super().__init__()
            self.optimizer = optim.AdamW(
                model.parameters(), lr, weight_decay=weight_decay
            )

        def step(self) -> None:
            self.optimizer.step()

        def zero_grad(self, set_to_none: bool = False) -> None:
            self.optimizer.zero_grad()
