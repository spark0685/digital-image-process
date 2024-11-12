import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


def process(image):
    # 验证图像是否只有0和255
    unique_values = np.unique(image)
    if not np.array_equal(unique_values, [0, 255]):
        raise ValueError("图像不只包含0和255两个灰度值")
    
    # 进行腐蚀操作
    kernel = np.ones((51, 1), np.uint8)
    eroded_image = cv2.erode(image, kernel, iterations=1)

    # 进行开操作
    kernel = np.ones((51, 1), np.uint8)
    opened_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    # 进行重建开操作
    kernel = np.ones((3, 3), np.uint8)
    previous_image = eroded_image
    num = 0
    while True:
        reconstructed_image = cv2.dilate(previous_image, kernel, iterations=1)
        reconstructed_image = np.minimum(reconstructed_image, image)
        if np.array_equal(reconstructed_image, previous_image):
            break
        previous_image = reconstructed_image
        num += 1
    print(f"迭代次数: {num}")
    
    return eroded_image, opened_image, reconstructed_image


current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
image_path = r"pic\\Fig0929(a)(text_image).tif"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 处理图像并生成直方图
try:
    eroded_image, opened_image, reconstructed_image = process(image)
    # 保存处理后的图像
    cv2.imwrite("result\\1-1.jpg", image)
    cv2.imwrite("result\\1-2.jpg", eroded_image)
    cv2.imwrite("result\\1-3.jpg", opened_image)
    cv2.imwrite("result\\1-4.jpg", reconstructed_image)
except ValueError as e:
    print(f"Error: {e}")
