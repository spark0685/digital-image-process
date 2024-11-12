import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


def process(image):
    # 验证图像是否只有0和255
    unique_values = np.unique(image)
    if not np.array_equal(unique_values, [0, 255]):
        raise ValueError("图像不只包含0和255两个灰度值")
    

    # 标记图像
    markers = np.zeros_like(image)
    markers[0, :] = image[0, :]
    markers[-1, :] = image[-1, :]
    markers[:, 0] = image[:, 0]
    markers[:, -1] = image[:, -1]

    # 进行重建开操作
    kernel = np.ones((3, 3), np.uint8)
    previous_image = markers
    num = 0
    while True:
        reconstructed_image = cv2.dilate(previous_image, kernel, iterations=1)
        reconstructed_image = np.minimum(reconstructed_image, image)
        if np.array_equal(reconstructed_image, previous_image):
            break
        previous_image = reconstructed_image
        num += 1
    print(f"迭代次数: {num}")

    cleared_image = image - reconstructed_image

    return markers, reconstructed_image, cleared_image


current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
image_path = r"pic\\Fig0929(a)(text_image).tif"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 处理图像并生成直方图
try:
    markers, reconstructed_image, cleared_image = process(image)
    cv2.imwrite("result\\3-1.jpg", image)  # 保存原图像
    cv2.imwrite("result\\3-2.jpg", markers)  # 保存标记图像
    cv2.imwrite("result\\3-3.jpg", reconstructed_image)  # 保存重建图像
    cv2.imwrite("result\\3-4.jpg", cleared_image)  # 保存去除背景后的图像

except ValueError as e:
    print(f"Error: {e}")
