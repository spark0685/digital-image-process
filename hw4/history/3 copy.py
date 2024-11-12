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
    
    ##低频和高频系数
    cA = np.convolve(signal, low_pass, mode='same')[::2]
    cD = np.convolve(signal, high_pass, mode='same')[::2]

    # cA = convolve(signal, low_pass)[::2]
    # cD = convolve(signal, high_pass)[::2]
    
    
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

# 一维逆小波变换
def inverse_wavelet_transform_1d(cA, cD):
    # 获取Daubechies小波的滤波器系数
    _, _, low_pass, high_pass = daubechies_wavelet_filters()
    
    # 对cA和cD进行上采样
    upsampled_cA = np.zeros(2*len(cA))
    upsampled_cA[::2] = cA
    upsampled_cD = np.zeros(2*len(cD))
    upsampled_cD[::2] = cD

    # 逆小波变换
    reconstructed_signal = np.convolve(upsampled_cA, low_pass, mode='same') + np.convolve(upsampled_cD, high_pass, mode='same')

    return reconstructed_signal


# 逆小波变换
def inverse_wavelet_transform(cA, cV, cH, cD):
    # 获取图像的形状
    n_rows, n_cols = cA.shape
    
    # 初始化重建的行
    rows_transformed_0 = np.zeros((n_rows * 2, n_cols))
    rows_transformed_1 = np.zeros((n_rows * 2, n_cols))
    
    # 对每一列进行逆小波变换
    for i in range(n_cols):
        rows_transformed_0[:, i] = inverse_wavelet_transform_1d(cA[:, i], cV[:, i])
        rows_transformed_1[:, i] = inverse_wavelet_transform_1d(cH[:, i], cD[:, i])
    
    # 初始化最终重建的图像
    reconstructed_image = np.zeros((n_rows * 2, n_cols * 2))
    
    # 对每一行进行逆小波变换
    for i in range(n_rows * 2):
        reconstructed_image[i, :] = inverse_wavelet_transform_1d(rows_transformed_0[i, :], rows_transformed_1[i, :])
    
    return reconstructed_image


def process(image, level=1):
    cA, cV, cH, cD = discrete_wavelet_transform_2d(image)

    cA2, cV2, cH2, cD2 = discrete_wavelet_transform_2d(cA)

    # #在二尺度下将 cA 设为全黑
    cA2 = np.zeros_like(cA2)
    # cV2 = np.zeros_like(cV2)
    # cV = np.zeros_like(cV)

    # 逆小波变换
    reconstructed_cA = inverse_wavelet_transform(cA2, cV2, cH2, cD2)

    reconstructed_cA = np.zeros_like(reconstructed_cA)

    # #对reconstructed_cA进行resize
    # reconstructed_cA = cv2.resize(reconstructed_cA, (cA.shape[0], cA.shape[1]), interpolation=cv2.INTER_LINEAR)

    # 进行逆小波变换
    reconstructed_image = inverse_wavelet_transform(reconstructed_cA, cV, cH, cD)


    # #对reconstructed_image进行归一化
    reconstructed_image = cv2.normalize(reconstructed_image, None, 0, 255, cv2.NORM_MINMAX)

    return reconstructed_image

# 设置工作目录并加载图像
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
image_path = r"test images\\demo-2.tif"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 处理图像以获取最终结果
result1 = process(image, 1)
# result2 = process(image, 2)  # 在二尺度下，cA会变为全黑
# result3 = process(image, 3)

# 保存结果
cv2.imwrite("result2\\3-1.jpg", result1)

cv2.imwrite("result2\\image.jpg", image)
# cv2.imwrite("result\\2-3.jpg", result3)
