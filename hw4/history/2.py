import cv2
import os
import numpy as np
import pywt


def process(image):
    # 进行二维小波变换
    # coeffs2 = pywt.wavedec2(image, "haar", level=2)

    # 使用 db1 小波进行二维小波变换
    coeffs2 = pywt.wavedec2(image, "db4", level=2)

    # 近似系数
    cA = coeffs2[0]

    # 细节系数
    cH = coeffs2[1][0]  # 水平细节
    cV = coeffs2[1][1]  # 垂直细节
    cD = coeffs2[1][2]  # 对角细节

    # 保存近似系数和细节系数
    cv2.imwrite("result\\image.jpg", image)
    cv2.imwrite("result\\cA.jpg", cA)
    cv2.imwrite("result\\cH.jpg", cH)
    cv2.imwrite("result\\cV.jpg", cV)
    cv2.imwrite("result\\cD.jpg", cD)

    # 归一化系数到 0-255 范围
    cA_norm = cv2.normalize(cA, None, 0, 255, cv2.NORM_MINMAX)
    cH_norm = cv2.normalize(cH, None, 0, 255, cv2.NORM_MINMAX)
    cV_norm = cv2.normalize(cV, None, 0, 255, cv2.NORM_MINMAX)
    cD_norm = cv2.normalize(cD, None, 0, 255, cv2.NORM_MINMAX)

    # 将子图调整为相同大小（保持近似系数的大小）
    size = (cA.shape[1], cA.shape[0])

    # 使用归一化的系数
    cA_resized = cv2.resize(cA_norm, size, interpolation=cv2.INTER_LINEAR )
    cH_resized = cv2.resize(cH_norm, size, interpolation=cv2.INTER_LINEAR )
    cV_resized = cv2.resize(cV_norm, size, interpolation=cv2.INTER_LINEAR )
    cD_resized = cv2.resize(cD_norm, size, interpolation=cv2.INTER_LINEAR )


    # 创建一个新画布，将四个子图拼接在一起
    top_row = np.hstack((cA_resized, cH_resized))
    bottom_row = np.hstack((cV_resized, cD_resized))
    reconstructed_image = np.vstack((top_row, bottom_row))

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
