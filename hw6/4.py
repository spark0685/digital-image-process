import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# 使用全局阈值处理图像
def process_1(image):
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

# 使用Otsu算法处理图像
def process_2(image):
    # 计算直方图
    pixel_counts = np.zeros(256)
    for pixel in image.flatten():
        pixel_counts[pixel] += 1
    # 归一化
    pixel_counts = pixel_counts / pixel_counts.sum()

    # 计算累计和
    s = np.zeros(256)
    s[0] = pixel_counts[0]
    for i in range(1, 256):
        s[i] = s[i - 1] + pixel_counts[i]
    
    # 计算累计均值
    m = np.zeros(256)
    m[0] = 0
    for i in range(1, 256):
        m[i] = m[i - 1] + i * pixel_counts[i]
    
    # 计算类间方差
    sigma = np.zeros(256)
    for i in range(256):
        if s[i] == 0 or s[i] == 1:
            sigma[i] = 0
        else:
            m1 = m[i] / s[i]
            m2 = (m[255] - m[i]) / (1 - s[i])
            sigma[i] = s[i] * (1 - s[i]) * (m1 - m2) ** 2
    
    # 找到最大类间方差对应的阈值
    threshold = np.argmax(sigma)
    print("threshold:", threshold)
    binary_image = np.zeros_like(image)
    binary_image[image > threshold] = 255

    return binary_image

# 使用Otsu算法处理分块阈值图像
def process_3(image):
    # 获取图像的高度和宽度
    height, width = image.shape

    # 计算每块的高度和宽度
    block_height = height // 2
    block_width = width // 3

    binary_image = np.zeros_like(image)

    # 对每个块进行阈值处理
    for i in range(2):
        for j in range(3):
            # 获取当前块的坐标
            start_row = i * block_height
            end_row = (i + 1) * block_height
            start_col = j * block_width
            end_col = (j + 1) * block_width

            # 提取当前块
            block = image[start_row:end_row, start_col:end_col]

            # # 使用Otsu算法获取当前块的阈值
            # _, threshold_value = cv2.threshold(block, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # # 对当前块进行阈值处理
            # _, binary_block = cv2.threshold(block, threshold_value, 255, cv2.THRESH_BINARY)

            binary_block = process_2(block)

            # 将处理后的块放回二值化图像中
            binary_image[start_row:end_row, start_col:end_col] = binary_block

    return binary_image

# 主程序部分
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

# 图像路径
image_path = r"pic\\Fig1046(a)(septagon_noisy_shaded).tif"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 处理图像
result_1 = process_1(image)
result_2 = process_2(image)
result_3 = process_3(image)

# 保存处理后的图像
cv2.imwrite("result\\4-1.jpg", result_1)
cv2.imwrite("result\\4-2.jpg", result_2)
cv2.imwrite("result\\4-3.jpg", result_3)



