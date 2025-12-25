# 延迟类型注解检查
from __future__ import annotations

# 主要计算
import torch
import torch.nn as nn
import torch.optim as optim

# 用户实现
from tools.configurator import Configurator


# 空网络，用于类型注解：供不同网络继承，以实现训练、测试函数的抽象
class Net:
    def __init__(self, configurator: Configurator):
        self.model = self.Model().to(configurator.device)
        self.loss = self.Loss()
        self.optimizer = self.Optimizer(
            model=self.model,
            lr=configurator.learning_rate,
            weight_decay=configurator.weight_decay,
        )

    class Model(nn.Module):
        pass

    class Loss(nn.Module):
        pass

    class Optimizer:
        def __init__(self, model: Net.Model, lr: float, weight_decay: float):
            pass

        def step(self):
            pass

        def zero_grad(self):
            pass

        pass
