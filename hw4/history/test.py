import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置字体为 SimHei
rcParams['font.family'] = 'SimHei'

# Daubechies小波的高通滤波器系数
def daubechies_wavelet_filters():
    # db4
    high_pass = np.array([1, 1, 1])
    return high_pass

# 一维小波变换
def discrete_wavelet_transform_1d(signal):
    high_pass = daubechies_wavelet_filters()
    cD = np.convolve(signal, high_pass, mode='same')
    return cD

# 测试信号
signal = np.array([1, 2, 3, 4, 5, 6, 7, 8])

# 进行小波变换
cD = discrete_wavelet_transform_1d(signal)

# 输出结果
print("原始信号:", signal)
print("高频系数 cD:", cD)

# 可视化结果
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.stem(signal)
plt.title('原始信号')
plt.subplot(1, 2, 2)
plt.stem(cD)
plt.title('高频系数 cD')
plt.show()
