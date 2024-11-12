import cv2
import os
import numpy as np

# Daubechies小波的低通和高通滤波器系数
def daubechies_wavelet_filters():
    # db1 小波 (Haar小波)
    # low_pass = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])
    # high_pass = np.array([-1 / np.sqrt(2), 1 / np.sqrt(2)])
    g0 = np.array([0.0332, -0.0126, -0.0992, 0.2979, 0.8037, 0.4976, -0.0296, -0.0758])
    g1 = np.array([0.0758, 0.0296, -0.4976, 0.8037, -0.2979, -0.0992, 0.0126, 0.0332])
    h0 = np.array([0.0758, -0.0296, -0.4976, -0.8037, -0.2979, 0.0992, 0.0296, -0.0758])
    h1 = np.array([0.0332, 0.0126, -0.0992, -0.2979, 0.8037, -0.4976, -0.0296, 0.0758])
    
    return h0, h1, g0, g1

# 卷积函数
def convolve(signal1, signal2):
    # 获取输入信号的长度
    len_signal1 = len(signal1)
    len_signal2 = len(signal2)
    
    # 计算卷积结果的长度
    output_length = len_signal1 + len_signal2 - 1
    output = np.zeros(output_length)
    
    # 进行卷积操作
    for i in range(len_signal1):
        for j in range(len_signal2):
            output[i + j] += signal1[i] * signal2[j]
    
    return output

# 一维小波变换
def discrete_wavelet_transform_1d(signal):
    # 获取Daubechies小波的滤波器系数
    low_pass, high_pass, _, _ = daubechies_wavelet_filters()
    
    # 低频和高频系数
    # cA = np.convolve(signal, low_pass, mode='full')[::2]
    # cD = np.convolve(signal, high_pass, mode='full')[::2]

    cA = convolve(signal, low_pass)[::2]
    cD = convolve(signal, high_pass)[::2]
    
    return cA, cD

# 二维小波变换
def discrete_wavelet_transform_2d(image):
    # 对图像的每一行进行一维小波变换
    rows_transformed_0 = np.zeros_like(image, dtype=np.float64)
    rows_transformed_1 = np.zeros_like(image, dtype=np.float64)
    cA_rows = []
    cD_rows = []
    
    for i in range(image.shape[0]):
        cA, cD = discrete_wavelet_transform_1d(image[i, :])
        cA_rows.append(cA)
        cD_rows.append(cD)
        rows_transformed_0[i, :len(cA)] = cA  # 保存低频系数
        rows_transformed_1[i, :len(cD)] = cD  # 保存高频系数
    
    cA_rows = np.array(cA_rows)
    cD_rows = np.array(cD_rows)
    
    # 对每一列进行一维小波变换
    cA_ = []
    cV_ = []
    
    for i in range(len(cA_rows[0])):
        cA, cV = discrete_wavelet_transform_1d(rows_transformed_0[:, i])
        cA_.append(cA)
        cV_.append(cV)
    cA_final = np.array(cA_).T
    cV_final = np.array(cV_).T

    cH_ = []
    cD_ = []

    for i in range(len(cD_rows[0])):
        cH, cD = discrete_wavelet_transform_1d(rows_transformed_1[:, i])
        cH_.append(cH)
        cD_.append(cD)

    cH_final = np.array(cH_).T
    cD_final = np.array(cD_).T
    
    return cA_final, cV_final, cH_final, cD_final

def concatenate_images(i1, i2, i3, i4):
    top_row = np.hstack((i1, i2))
    bottom_row = np.hstack((i3, i4))
    return np.vstack((top_row, bottom_row))


def process(image, level=1):
    # 进行二维小波变换
    cA, cV, cH, cD = discrete_wavelet_transform_2d(image)

    # 归一化系数到 0-255 范围
    cA_norm = cv2.normalize(cA, None, 0, 255, cv2.NORM_MINMAX)
    cH_norm = cv2.normalize(cH, None, 0, 255, cv2.NORM_MINMAX)
    cV_norm = cv2.normalize(cV, None, 0, 255, cv2.NORM_MINMAX)
    cD_norm = cv2.normalize(cD, None, 0, 255, cv2.NORM_MINMAX)

    # 将子图调整为相同大小（保持近似系数的大小）
    size = (cA.shape[1], cA.shape[0])

    # 使用归一化的系数并调整大小
    cA_resized = cA_norm
    cD_resized = cD_norm
    cH_resized = cH_norm
    cV_resized = cV_norm

    if level > 1:
        # 递归调用 process 函数
        cA_resized = process(cA_resized, level - 1)
    
    cA_resized = cv2.resize(cA_resized, size, interpolation=cv2.INTER_LANCZOS4)

    reconstructed_image = concatenate_images(cA_resized, cV_resized, cH_resized, cD_resized)

    return reconstructed_image

# 设置工作目录并加载图像
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
image_path = r"test images\\demo-2.tif"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 处理图像以获取最终结果
result1 = process(image, 1)
result2 = process(image, 2)
result3 = process(image, 3)

# 保存结果
cv2.imwrite("result\\2-1.jpg", result1)
cv2.imwrite("result\\2-2.jpg", result2)
cv2.imwrite("result\\2-3.jpg", result3)

