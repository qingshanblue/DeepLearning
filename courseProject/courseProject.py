# 计算
import torch

# 辅助
import matplotlib.pyplot as plt

# 用户
from tools.configurator import Configurator
from tools.trainer import run_train
from tools.evaluator import run_test, visualize_results
from nets.net import Net
from nets.residualNet import ResidualNet
from nets.AlexNet import AlexNet
from nets.VGGNet import VGGNet


def TrainOrTest(net: Net, configurator: Configurator, testOnly: bool = False) -> None:
    print(f"{net.name}模型参数量:{net.yields/1000000:.4f}M")
    if not testOnly:
        run_train(
            net=net,
            configurator=configurator,
        )
    # 无论是否训练，都执行测试流程以评估模型性能
    run_test(
        net=net,
        configurator=configurator,
    )
    # 腾出显存
    net.model.to("cpu")


# 程序入口，主函数：
if __name__ == "__main__":
    mode = []
    # 交互：选择执行的操作
    print(
        f"请输入想要执行的操作:\n\
        确认:y, 取消:n\n\
        1x:执行对应序号模型的评估\n\
        0: 执行所有训练并评估\n\
        1: 训练ResidualNet\n\
        2: 训练AlexNet\n\
        3: 训练VGGNet"
    )
    while True:  # 多次输入配置
        try:
            value = input("请输入：").strip()
            if value in ["0", "10"]:
                # 选择0或10时清空模式列表，确保只执行所有模型
                mode.clear()
                mode.append(int(value))
            elif value in ["1", "2", "3", "11", "12", "13"]:
                # 检查是否已选择该模型或已选择全部执行模式，避免重复训练
                if (value not in mode) and (0 not in mode) and (10 not in mode):
                    mode.append(int(value))
            elif value in ["N", "n"]:
                # 取消选择，清空模式列表
                mode.clear()
            elif value in ["Y", "y"]:
                # 确认选择，退出输入循环
                break
            else:
                print("输入超出范围")
        except:
            print("输入的字符无效")
        mode.sort()  # 排序以保证后面保存图片命名时一致
        print(f"您选择了模式 {mode}")

    # 训练、评估模型
    # 配置训练参数，包括类别数、学习率、权重衰减等
    configurator0 = Configurator(
        num_classes=58,
        image_size=(64, 64),
        learning_rate=1e-3,
        weight_decay=1e-4,
        num_epochs=32,
        batch_size=128,
        accumulation_steps=1,
        seed=114514,
    )
    configurator1 = Configurator(
        num_classes=58,
        image_size=(32, 32),
        learning_rate=1e-3,
        weight_decay=1e-4,
        num_epochs=32,
        batch_size=128,
        accumulation_steps=1,
        seed=114514,
    )
    configurator = configurator0
    # 保存使用了哪些模型的列表
    net_list = []
    for m in mode:  # 根据输入执行操作
        # 判断是否仅评估模式（10+表示仅评估）
        testOnly = True if m >= 10 else False
        match m:
            case 0 | 10:
                # 执行所有模型的训练或评估
                residNet = ResidualNet(configurator=configurator)
                net_list.append(residNet)
                TrainOrTest(net=residNet, configurator=configurator, testOnly=testOnly)
                alexNet = AlexNet(configurator=configurator)
                net_list.append(alexNet)
                TrainOrTest(net=alexNet, configurator=configurator, testOnly=testOnly)
                vggNet = VGGNet(configurator=configurator)
                net_list.append(vggNet)
                TrainOrTest(net=vggNet, configurator=configurator, testOnly=testOnly)
                break
            case 1 | 11:
                # 执行ResidualNet的训练或评估
                residNet = ResidualNet(configurator=configurator)
                net_list.append(residNet)
                TrainOrTest(net=residNet, configurator=configurator, testOnly=testOnly)
            case 2 | 12:
                # 执行AlexNet的训练或评估
                alexNet = AlexNet(configurator=configurator)
                net_list.append(alexNet)
                TrainOrTest(net=alexNet, configurator=configurator, testOnly=testOnly)
            case 3 | 13:
                # 执行VGGNet的训练或评估
                vggNet = VGGNet(configurator=configurator)
                net_list.append(vggNet)
                TrainOrTest(net=vggNet, configurator=configurator, testOnly=testOnly)
    if net_list:
        # 可视化训练结果，展示样本预测效果
        visualize_results(nets=net_list, configurator=configurator, num_samples=4)
    # 绘制结果图
    if mode:
        plt.tight_layout()
        plt.show()
        plt.close("all")
        torch.cuda.empty_cache()
