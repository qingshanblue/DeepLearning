# 延迟类型注解检查
from __future__ import annotations

# 计算
import torch

# 用户
from tools.dataloader import MyDataLoader


class Configurator:
    def __init__(
        self,
        num_classes: int = 58,
        image_size: tuple[int, int] = (64, 64),
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        num_epochs: int = 32,
        batch_size: int = 128,
        accumulation_steps: int = 1,
        seed: int = 114514,
    ) -> None:
        # 设置模型训练的核心超参数
        self.num_classes = num_classes
        self.image_size = image_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.accumulation_steps = accumulation_steps
        self.desc = f"imgSz{image_size}_lr{learning_rate}_wd{weight_decay}_ep{num_epochs}_bcSz{batch_size}_acs{accumulation_steps}"

        # 固定随机种子确保实验可复现性  注：不知道是哪里还有随机种子没设置，跑起来好像还是有点不一样
        self.seed = seed
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            # 优先使用GPU加速训练
            self.device = torch.device("cuda")
            torch.cuda.manual_seed(self.seed)
        else:
            # GPU不可用时回退到CPU
            self.device = torch.device("cpu")

        # 配置数据相关参数
        root_dir = "./courseProject/data"
        images_size = self.image_size  # 统一调整图像尺寸以适应模型输入
        train_proportion = 0.7  # 70%数据用于训练
        valid_proportion = 0.2  # 20%数据用于验证，剩余10%用于测试
        batch_size = self.batch_size
        num_workers = 6  # 多进程加载数据提高效率
        prefetch_factor = 2  # 预取数据减少IO等待时间

        # 创建数据集并按比例分割为训练/验证/测试集
        chinaTrafficSignData = MyDataLoader(root_dir=root_dir)
        self.train_loader, self.valid_loader, self.test_loader = (
            chinaTrafficSignData.getDataLoaders(
                image_size=images_size,
                train_proportion=train_proportion,
                valid_proportion=valid_proportion,
                batch_size=batch_size,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                persistent_workers=True,  # 保持worker进程存活避免重复创建开销
                pin_memory=(
                    self.device.type == "cuda"
                ),  # GPU训练时启用内存固定加速传输
                seed=self.seed,  # 确保数据分割的随机性一致
            )
        )
