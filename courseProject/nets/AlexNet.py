# 延迟类型注解检查
from __future__ import annotations

# 主要计算
import torch
import torch.nn as nn
import torch.optim as optim

# 用户实现
from nets.net import Net


# AlexNet
class AlexNet(Net):
    class Model(Net.Model):
        def __init__(
            self,
            chns_in: int = 3,           # 输入通道数，如RGB图像为3
            chns_mid: list[int] = [96, 256, 384, 384],  # 各卷积层输出通道数
            ker_size: list[int] = [11, 5, 3, 3],        # 各卷积层卷积核大小
            padding: list[int] = [2, 2, 1, 1],          # 各卷积层填充大小
            stride: list[int] = [4, 1, 1, 1],           # 各卷积层步长
            poolKer_size: list[int] = [3, 3],           # 池化层卷积核大小
            poolStride: list[int] = [2, 2],             # 池化层步长
            feats_mid: list[int] = [4096, 4096],        # 全连接层中间特征数
            feats_out: int = 58,                        # 输出类别数
            dropout_rate: list[float] = [0.4, 0.4],     # Dropout比率
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

            # 卷积层部分：5个卷积层 + 激活函数 + 池化层
            self.conv = nn.Sequential(
                # cov1: 第一个卷积层，大幅下采样
                nn.Conv2d(
                    in_channels=self.chns_in,
                    out_channels=self.chns_mid[0],
                    kernel_size=self.ker_size[0],
                    padding=self.padding[0],
                    stride=self.stride[0],
                ),
                # nn.BatchNorm2d(chns_mid[0]),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(self.poolKer_size[0], self.poolStride[0]),
                # cov2: 第二个卷积层，进一步提取特征
                nn.Conv2d(
                    in_channels=self.chns_mid[0],
                    out_channels=self.chns_mid[1],
                    kernel_size=self.ker_size[1],
                    padding=self.padding[1],
                    stride=self.stride[1],
                ),
                # nn.BatchNorm2d(chns_mid[1]),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(self.poolKer_size[1], self.poolStride[1]),
                # cov3: 第三个卷积层，加深网络深度
                nn.Conv2d(
                    in_channels=self.chns_mid[1],
                    out_channels=self.chns_mid[2],
                    kernel_size=self.ker_size[2],
                    padding=self.padding[2],
                    stride=self.stride[2],
                ),
                # nn.BatchNorm2d(chns_mid[2]),
                nn.ReLU(inplace=True),
                # nn.MaxPool2d(self.poolKer_size[2], self.poolStride[2]),
                # cov4: 第四个卷积层，保持特征图尺寸
                nn.Conv2d(
                    in_channels=self.chns_mid[2],
                    out_channels=self.chns_mid[3],
                    kernel_size=self.ker_size[3],
                    padding=self.padding[3],
                    stride=self.stride[3],
                ),
                # nn.BatchNorm2d(chns_mid[3]),
                nn.ReLU(inplace=True),
                # nn.MaxPool2d(self.poolKer_size[3], self.poolStride[3]),
            )
            # 自适应平均池化，将特征图统一为6x6大小
            self.pool = nn.Sequential(nn.AdaptiveAvgPool2d((6, 6)))
            # 全连接层部分：3个全连接层 + Dropout
            self.fc = nn.Sequential(
                # fc1: 第一个全连接层，大幅降维
                nn.Linear(self.chns_mid[3] * 6 * 6, self.feats_mid[0]),
                # nn.BatchNorm1d(self.feats_mid[0]),
                nn.ReLU(inplace=True),
                nn.Dropout(p=self.dropout_rate[0]),
                # fc2: 第二个全连接层，进一步特征提取
                nn.Linear(self.feats_mid[0], self.feats_mid[1]),
                # nn.BatchNorm1d(self.feats_mid[1]),
                nn.ReLU(inplace=True),
                nn.Dropout(p=self.dropout_rate[1]),
                # fc3: 输出层，映射到类别数
                nn.Linear(self.feats_mid[1], self.feats_out),
            )
            # 初始化网络权重
            self._initialize_weights()

        def _initialize_weights(self):
            # 遍历所有模块，根据不同类型进行权重初始化
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    # 卷积层使用Kaiming初始化，适合ReLU激活函数
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                    # 批归一化层初始化为1和0，保持输入分布
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    # 全连接层使用正态分布初始化
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

        def forward(self, X: torch.Tensor) -> torch.Tensor:
            # 前向传播过程：卷积 -> 池化 -> 展平 -> 全连接
            X = self.conv(X)
            X = self.pool(X)
            X = torch.flatten(X, 1)  # 展平为一维向量
            X = self.fc(X)
            return X

    class Loss(Net.Loss):
        def __init__(self) -> None:
            super().__init__()
            # 使用交叉熵损失函数，适用于多分类任务
            self.criterion = nn.CrossEntropyLoss()

        def calc(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
            # 计算预测值与真实值之间的损失
            return self.criterion(y_pred, y_true)

        def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
            # 使类实例可以像函数一样调用
            return self.calc(y_pred, y_true)

    class Optimizer(Net.Optimizer):
        def __init__(
            self, model: AlexNet.Model, lr: float = 0.01, weight_decay: float = 0.01
        ) -> None:
            super().__init__(model=model, lr=lr, weight_decay=weight_decay)
            # 使用AdamW优化器，具有较好的收敛性和泛化能力
            self.optimizer = optim.AdamW(
                model.parameters(), lr, weight_decay=weight_decay
            )

        def step(self) -> None:
            # 执行一次优化步骤
            self.optimizer.step()

        def zero_grad(self, set_to_none: bool = False) -> None:
            # 清零梯度，准备下一次反向传播
            self.optimizer.zero_grad(set_to_none=set_to_none)