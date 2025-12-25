# 其他辅助
import matplotlib.pyplot as plt

# 用户实现
from tools.configurator import MyParams
from tools.trainer import train_residualNet, train_alexNet, train_VGGNet

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
                break
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

    # 训练模型
    params = MyParams(
        num_classes=58,
        learning_rate=1e-3,
        weight_decay=1e-4,
        num_epochs=32,
        accumulation_steps=1,
        seed=114514,
    )
    for m in mode:
        match m:
            case "0":
                train_residualNet(params=params)
                train_alexNet(params=params)
                train_VGGNet(params=params)
                break
            case "1":
                train_residualNet(params=params)
            case "2":
                train_alexNet(params=params)
            case "3":
                train_VGGNet(params=params)
    # 绘制结果图
    if mode:
        plt.tight_layout()
        plt.show()
