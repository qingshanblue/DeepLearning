# 延迟类型注解检查
from __future__ import annotations

# 计算
import torch
import torch.nn as nn
import torch.optim as optim

# 用户
from tools.configurator import Configurator


# 网络模板，用于类型注解：供不同网络继承，以实现训练、测试函数的抽象
class Net:
    def __init__(self, configurator: Configurator):
        # 将模型移动到指定设备（GPU/CPU）以优化计算性能
        self.model = self.Model().to(configurator.device)
        # 初始化损失函数，用于计算模型预测与真实标签的差异
        self.loss = self.Loss()
        # 配置优化器，包含权重衰减以防止过拟合
        self.optimizer = self.Optimizer(
            model=self.model,
            lr=configurator.learning_rate,
            weight_decay=configurator.weight_decay,
        )
        # 记录模型名称以供路径用于保存
        self.name = self.__class__.__name__
        # 记录模型参数量以对比
        self.yields = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    class Model(nn.Module):
        # 抽象模型类，子类必须实现具体的网络结构
        pass

    class Loss(nn.Module):
        # 抽象损失函数类，子类需定义具体的损失计算方式
        pass

    class Optimizer:
        def __init__(self, model: Net.Model, lr: float, weight_decay: float):
            # 初始化优化器参数，学习率控制更新步长，权重衰减防止过拟合
            pass

        def step(self):
            # 执行一步参数更新，根据计算得到的梯度调整模型权重
            pass

        def zero_grad(self):
            # 清零梯度，避免梯度累积影响下次更新
            pass

        pass
