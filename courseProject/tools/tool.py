import csv
import os
from collections import Counter
import matplotlib.pyplot as plt


def plot_category_distribution(root_dir: str) -> None:
    """
    遍历 annotations.csv 中的所有类别，统计每个类别的图片数量并绘图显示。

    Args:
        root_dir (str): 数据集根目录路径，包含 images 文件夹和 annotations.csv 文件。
    """
    label_path = os.path.join(root_dir, "annotations.csv")

    # 读取标签文件并统计类别分布
    categories = []
    with open(label_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            categories.append(int(row["category"]))

    # 统计每个类别的样本数
    category_counts = Counter(categories)

    # 提取类别和对应的数量
    labels = list(category_counts.keys())
    counts = list(category_counts.values())

    # 绘制柱状图
    plt.figure(figsize=(12, 6))
    bars = plt.bar(labels, counts, color="skyblue")
    plt.xlabel("Category")
    plt.ylabel("Number of Images")
    plt.title("Image Distribution per Category")
    plt.xticks(rotation=45)

    # 在每个柱子上显示具体数值
    for bar, count in zip(bars, counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(count),
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_category_distribution("./data/")
    # import torch

    # # 获取 CUDA 版本字符串
    # import torch
    # print(torch.version.cuda)
    # print(f"完整版本信息: {torch.__version__}")