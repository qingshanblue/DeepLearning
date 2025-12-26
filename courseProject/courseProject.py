# 其他辅助
import matplotlib.pyplot as plt

# 用户实现
from tools.configurator import Configurator
from tools.trainer import run_train
from tools.evaluator import run_test, visualize_results
from nets.net import Net
from nets.residualNet import ResidualNet
from nets.AlexNet import AlexNet
from nets.VGGNet import VGGNet


def go_residualNet() -> Net:
    residual = ResidualNet(configurator=configurator)
    run_train(
        net=residual,
        configurator=configurator,
        model_name="residualNet",
    )
    run_test(
        net=residual,
        configurator=configurator,
        model_name="residualNet",
    )
    return residual


def go_alexNet() -> Net:
    alexNet = AlexNet(configurator=configurator)
    run_train(
        net=alexNet,
        configurator=configurator,
        model_name="alexNet",
    )
    run_test(
        net=alexNet,
        configurator=configurator,
        model_name="alexNet",
    )
    return alexNet


def go_vggNet() -> Net:
    vggNet = VGGNet(configurator=configurator)
    run_train(
        net=vggNet,
        configurator=configurator,
        model_name="vggNet",
    )
    run_test(
        net=vggNet,
        configurator=configurator,
        model_name="vggNet",
    )
    return vggNet


# 程序入口，主函数：
if __name__ == "__main__":
    mode = []
    # 交互：选择执行的操作
    print(
        f"请输入想要执行的操作:\n\
        可多次输入,y确认,n取消,大小写不限\n\
        0: 执行所有训练\n\
        1: 训练ResidualNet\n\
        2: 训练AlexNet\n\
        3: 训练VGGNet"
    )
    while True:
        try:
            value = input("请输入：").strip()
            if value == "0":
                mode.clear()
                mode.append(value)
            elif value in ["1", "2", "3"]:
                if value not in mode:  # 防止重复训练同一模型
                    mode.append(value)
            elif value in ["N", "n"]:
                mode.clear()
                break
            elif value in ["Y", "y"]:
                break
            else:
                print("输入超出范围")
        except:
            print("输入的字符无效")
        print(f"您选择了模式 {mode}")

    # 训练、评估模型
    configurator = Configurator(
        num_classes=58,
        learning_rate=1e-3,
        weight_decay=1e-4,
        num_epochs=32,
        accumulation_steps=1,
        seed=114514,
    )
    net_list = []
    for m in mode:
        match m:
            case "0":
                net_list.append(go_residualNet())
                net_list.append(go_alexNet())
                net_list.append(go_vggNet())
                break
            case "1":
                net_list.append(go_residualNet())
            case "2":
                net_list.append(go_alexNet())
            case "3":
                net_list.append(go_vggNet())
    if net_list:
        visualize_results(nets=net_list, configurator=configurator, num_samples=4)
    # 绘制结果图
    if mode:
        plt.tight_layout()
        plt.show()
