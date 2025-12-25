# 延迟类型注解检查
from __future__ import annotations

# 主要计算
import torch
import torch.nn as nn
import torch.optim as optim


# ResidualNet
class ResidualNet:
    class Model(nn.Module):
        def __init__(
            self,
            # chns_in: int = 3,
            # chns_base: int,
            # feats_base: int,
            num_classes: int,
            # dropout_rate: float,
            # ker_size: int,
            # padding: int = 0,
            # stride: int = 1,
        ) -> None:
            super().__init__()
            # self.chns_in = chns_in
            # self.chns_base = chns_base
            # self.feats_base = feats_base
            self.num_classes = num_classes
            # self.dropout_rate = dropout_rate
            # self.ker_size = ker_size
            # self.padding = padding
            # self.stride = stride

            self.stem = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )
            self.layer1 = nn.Sequential(
                ResidualNet.Model.ResidualBlock(64, 64),
                ResidualNet.Model.ResidualBlock(64, 64),
            )
            self.layer2 = nn.Sequential(
                ResidualNet.Model.ResidualBlock(64, 128, stride=2),
                ResidualNet.Model.ResidualBlock(128, 128),
            )
            self.layer3 = nn.Sequential(
                ResidualNet.Model.ResidualBlock(128, 256, stride=2),
                ResidualNet.Model.ResidualBlock(256, 256),
            )
            self.layer4 = nn.Sequential(
                ResidualNet.Model.ResidualBlock(256, 512, stride=2),
                ResidualNet.Model.ResidualBlock(512, 512),
            )
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512, num_classes)

            # 权重初始化
            self._initialize_weights()

        def forward(self, x):
            x = self.stem(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.pool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x

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

        class ResidualBlock(nn.Module):
            def __init__(self, in_channels, out_channels, stride=1):
                super().__init__()

                self.conv1 = nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    bias=False,
                )
                self.bn1 = nn.BatchNorm2d(out_channels)
                self.relu = nn.ReLU(inplace=True)

                self.conv2 = nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                )
                self.bn2 = nn.BatchNorm2d(out_channels)

                if stride != 1 or in_channels != out_channels:
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(
                            in_channels,
                            out_channels,
                            kernel_size=1,
                            stride=stride,
                            bias=False,
                        ),
                        nn.BatchNorm2d(out_channels),
                    )
                else:
                    self.shortcut = nn.Identity()

            def forward(self, x):
                out = self.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                out += self.shortcut(x)
                out = self.relu(out)
                return out

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
            self, model: ResidualNet.Model, lr: float = 0.01, weight_decay: float = 0.01
        ) -> None:
            super().__init__()
            self.optimizer = optim.AdamW(
                model.parameters(), lr, weight_decay=weight_decay
            )

        def step(self) -> None:
            self.optimizer.step()

        def zero_grad(self, set_to_none: bool = False) -> None:
            self.optimizer.zero_grad()
