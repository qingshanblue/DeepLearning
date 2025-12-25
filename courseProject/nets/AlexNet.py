# 延迟类型注解检查
from __future__ import annotations

# 主要计算
import torch
import torch.nn as nn
import torch.optim as optim


# AlexNet
class AlexNet:
    class Model(nn.Module):
        def __init__(
            self,
            chns_in: int,
            chns_mid: list[int],
            ker_size: list[int],
            padding: list[int],
            stride: list[int],
            poolKer_size: list[int],
            poolStride: list[int],
            feats_mid: list[int],
            feats_out: int,
            dropout_rate: list[float],
        ) -> None:
            super().__init__()
            self.chns_in = chns_in
            self.chns_mid = chns_mid
            self.ker_size = ker_size
            self.padding = padding
            self.stride = stride
            self.poolKer_size = poolKer_size
            self.poolStride = poolStride
            self.feats_mid = feats_mid
            self.feats_out = feats_out
            self.dropout_rate = dropout_rate

            self.conv = nn.Sequential(
                # cov1
                nn.Conv2d(
                    in_channels=self.chns_in,
                    out_channels=self.chns_mid[0],
                    kernel_size=self.ker_size[0],
                    padding=self.padding[0],
                    stride=self.stride[0],
                ),
                nn.BatchNorm2d(chns_mid[0]),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(self.poolKer_size[0], self.poolStride[0]),
                # cov2
                nn.Conv2d(
                    in_channels=self.chns_mid[0],
                    out_channels=self.chns_mid[1],
                    kernel_size=self.ker_size[1],
                    padding=self.padding[1],
                    stride=self.stride[1],
                ),
                nn.BatchNorm2d(chns_mid[1]),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(self.poolKer_size[1], self.poolStride[1]),
                # cov3
                nn.Conv2d(
                    in_channels=self.chns_mid[1],
                    out_channels=self.chns_mid[2],
                    kernel_size=self.ker_size[2],
                    padding=self.padding[2],
                    stride=self.stride[2],
                ),
                nn.BatchNorm2d(chns_mid[2]),
                nn.ReLU(inplace=True),
                # nn.MaxPool2d(self.poolKer_size[2], self.poolStride[2]),
                # cov4
                nn.Conv2d(
                    in_channels=self.chns_mid[2],
                    out_channels=self.chns_mid[3],
                    kernel_size=self.ker_size[3],
                    padding=self.padding[3],
                    stride=self.stride[3],
                ),
                nn.BatchNorm2d(chns_mid[3]),
                nn.ReLU(inplace=True),
                # nn.MaxPool2d(self.poolKer_size[3], self.poolStride[3]),
            )
            self.pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)))
            self.fc = nn.Sequential(
                # fc1
                nn.Linear(self.chns_mid[3], self.feats_mid[0]),
                nn.BatchNorm1d(self.feats_mid[0]),
                nn.ReLU(inplace=True),
                nn.Dropout(p=self.dropout_rate[0]),
                # fc2
                nn.Linear(self.feats_mid[0], self.feats_mid[1]),
                nn.BatchNorm1d(self.feats_mid[1]),
                nn.ReLU(inplace=True),
                nn.Dropout(p=self.dropout_rate[1]),
                # fc3
                nn.Linear(self.feats_mid[1], self.feats_out),
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
            self, model: AlexNet.Model, lr: float = 0.01, weight_decay: float = 0.01
        ) -> None:
            super().__init__()
            self.optimizer = optim.AdamW(
                model.parameters(), lr, weight_decay=weight_decay
            )

        def step(self) -> None:
            self.optimizer.step()

        def zero_grad(self, set_to_none: bool = False) -> None:
            self.optimizer.zero_grad()
