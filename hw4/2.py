import cv2
import os
import numpy as np
import pywt


# 对图像进行二维快速小波变换
def process(image):
    # 进行二维小波变换
    coeffs2 = pywt.wavedec2(image, "haar", level=2)  # 使用 Haar 小波，分解到第 2 层

    # 近似系数
    cA = coeffs2[0]

    # 细节系数
    cH = coeffs2[1][0]  # 水平细节
    cV = coeffs2[1][1]  # 垂直细节
    cD = coeffs2[1][2]  # 对角细节

    # 将子图调整为相同大小（可以使用 resize 或 crop）
    size = (cA.shape[1], cA.shape[0])  # 取近似系数的大小

    cA_resized = cv2.resize(cA, size)
    cH_resized = cv2.resize(cH, size)
    cV_resized = cv2.resize(cV, size)
    cD_resized = cv2.resize(cD, size)

    # 创建一个新画布，将四个子图拼接在一起
    top_row = np.hstack((cA_resized, cH_resized))  # 上排：近似系数 和 水平细节
    bottom_row = np.hstack((cV_resized, cD_resized))  # 下排：垂直细节 和 对角细节
    reconstructed_image = np.vstack((top_row, bottom_row))  # 将上排和下排拼接

    # 确保图像的数据类型与输入图像一致
    reconstructed_image = np.clip(reconstructed_image, 0, 255).astype(np.uint8)

    return reconstructed_image


# Set the working directory and load the image
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
image_path = r"test images\\demo-2.tif"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Process the image to get the final result
result = process(image)

# Save the result
cv2.imwrite("result\\result2.jpg", result)
