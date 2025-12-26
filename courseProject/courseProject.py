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


def go_residualNet(configurator: Configurator, testOnly: bool = False) -> Net:
    residual = ResidualNet(configurator=configurator)
    if not testOnly:
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


def go_alexNet(configurator: Configurator, testOnly: bool = False) -> Net:
    alexNet = AlexNet(configurator=configurator)
    if not testOnly:
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


def go_vggNet(configurator: Configurator, testOnly: bool = False) -> Net:
    vggNet = VGGNet(configurator=configurator)
    if not testOnly:
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
        确认:y, 取消:n\n\
        1x:执行对应序号模型的评估\n\
        0: 执行所有训练并评估\n\
        1: 训练ResidualNet\n\
        2: 训练AlexNet\n\
        3: 训练VGGNet"
    )
    while True:
        try:
            value = input("请输入：").strip()
            if value in ["0", "10"]:
                mode.clear()
                mode.append(int(value))
            elif value in ["1", "2", "3", "11", "12", "13"]:
                if (
                    (value not in mode) and (0 not in mode) and (10 not in mode)
                ):  # 防止重复训练同一模型
                    mode.append(int(value))
            elif value in ["N", "n"]:
                mode.clear()
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
        testOnly = True if m >= 10 else False
        match m:
            case 0 | 10:
                net_list.append(
                    go_residualNet(configurator=configurator, testOnly=testOnly)
                )
                net_list.append(
                    go_alexNet(configurator=configurator, testOnly=testOnly)
                )
                net_list.append(go_vggNet(configurator=configurator, testOnly=testOnly))
                break
            case 1 | 11:
                net_list.append(
                    go_residualNet(configurator=configurator, testOnly=testOnly)
                )
            case 2 | 12:
                net_list.append(
                    go_alexNet(configurator=configurator, testOnly=testOnly)
                )
            case 3 | 13:
                net_list.append(go_vggNet(configurator=configurator, testOnly=testOnly))
    if net_list:
        visualize_results(nets=net_list, configurator=configurator, num_samples=4)
    # 绘制结果图
    if mode:
        plt.tight_layout()
        plt.show()
