import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# 使用全局阈值处理图像
def process(image):
    # _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # binary_image = cv2.bitwise_not(binary_image)

    # 计算平均灰度值
    mean = np.mean(image)
    print("mean:", mean)
    # 迭代三次
    for i in range(3):
        # 计算两个类的均值
        mean1 = np.mean(image[image > mean])
        mean2 = np.mean(image[image <= mean])
        # 更新全局阈值
        mean = (mean1 + mean2) / 2
        print("mean:", mean)
    binary_image = np.zeros_like(image)
    binary_image[image > mean] = 255

    return binary_image




current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

image_path = r"pic\\Fig1038(a)(noisy_fingerprint).tif"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 处理图像
binary_image = process(image)

# 保存处理后的图像
cv2.imwrite("result\\2-1.jpg", binary_image)
