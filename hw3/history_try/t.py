import numpy as np
import imageio.v2 as imageio
from numba import njit, prange
from scipy.fft import fft, ifft, fftfreq
import matplotlib.pyplot as plt

@njit(parallel=True)
def radon_transform(image):
    size = image.shape
    round_size = int(np.sqrt(size[0]**2 + size[1]**2))
    result = np.zeros((180, round_size), dtype=np.int64)

    for angle_ in prange(180):
        angle = np.deg2rad(angle_)
        cos = np.cos(angle)
        sin = np.sin(angle)

        #统计不同的rou的取点数量
        num = np.zeros(round_size, dtype=np.int64)

        for x in range(size[0]):
            for y in range(size[1]):
                x_ = y - size[1] / 2
                y_ = -x + size[0] / 2
                rou = x_ * cos + y_ * sin
                if rou > 0:
                    rou = int(rou + 0.5)  # 四舍五入
                else:
                    rou = int(rou - 0.5)
                if np.abs(rou) > round_size // 2:
                    continue
                result[angle_][rou + round_size // 2 - 1] += image[x][y]
                num[rou + round_size // 2 - 1] += 1
        
        #除以num
        for i in range(round_size):
            if num[i] != 0:
                result[angle_][i] = result[angle_][i] // num[i]

    # 使用缩放将结果转换到 0 到 255
    min_val = np.min(result)
    max_val = np.max(result)

    if max_val > min_val:  # 确保不除以零
        result_scaled = (result - min_val) / (max_val - min_val) * 255
    else:
        result_scaled = np.zeros_like(result, dtype=np.float64)

    result_new = result_scaled.astype(np.uint8)
    return result_new

# # 创建滤波器
# def create_filter(size):
#     # 使用 Ram-Lak 滤波器：|f|
#     filter_ = np.abs(fftfreq(size))

#     return filter_

# #创建一个Hann窗滤波器乘以Ram-Lak滤波器
# def create_filter(size):
#     filter_ = np.abs(fftfreq(size))
#     filter_ = filter_ * 0.5 * (1 - np.cos(2 * np.pi * np.arange(size) / size))
#     return filter_

#创建一个Hamming窗滤波器，其中窗的原点位于0
def create_filter(size):
    filter_ = np.abs(fftfreq(size))
    hamming = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(size) / size)
    # 将Hamming窗口的原点移动到0
    hamming = np.roll(hamming, size // 2)
    filter_ = filter_ * hamming
    return filter_



# 对投影数据进行滤波并展示傅里叶变换和滤波器
def apply_filter(projections, show_fft=False):
    filtered_projections = np.zeros_like(projections, dtype=np.float64)
    filter_ = create_filter(projections.shape[1])

    # 可视化滤波器
    if show_fft:
        plt.figure(figsize=(6, 4))
        plt.plot(filter_)
        plt.title('Filter (Ram-Lak)')
        plt.xlabel('Frequency')
        plt.ylabel('Magnitude')
        plt.grid(True)
        plt.show()

    for i in range(projections.shape[0]):
        # 对每个角度的投影数据应用傅里叶变换
        projection = projections[i]
        projection_fft = fft(projection)

        if show_fft:
            # 可视化原始傅里叶变换的幅度谱
            plt.figure(figsize=(10, 6))
            plt.subplot(2, 2, 1)
            plt.plot(projection)
            plt.title(f"Projection {i + 1} in time domain")

            plt.subplot(2, 2, 2)
            plt.plot(np.abs(projection_fft))
            plt.title(f"Projection {i + 1} in frequency domain (FFT magnitude)")

        # 应用滤波器
        filtered_projection_fft = projection_fft * filter_

        if show_fft:
            # 可视化滤波后的傅里叶变换幅度谱
            plt.subplot(2, 2, 3)
            plt.plot(np.abs(filtered_projection_fft))
            plt.title(f"Filtered Projection {i + 1} in frequency domain (Filtered FFT magnitude)")

            # 可视化滤波后的投影数据在时域中的表现
            plt.subplot(2, 2, 4)
            plt.plot(np.real(ifft(filtered_projection_fft)))
            plt.title(f"Filtered Projection {i + 1} in time domain (After IFFT)")

            plt.tight_layout()
            plt.show()

        # 反变换回时域
        filtered_projections[i] = np.real(ifft(filtered_projection_fft))
    
    return filtered_projections

# 示例代码
if __name__ == '__main__':
    # 假设 projections 是shepp-logan_phantom_transform.png的投影数据
    filename1 = 'pic/Fig0539(a)(vertical_rectangle)'
    filename2 = 'pic/Fig0539(c)(shepp-logan_phantom)'

    # 读取TIFF图像
    image1 = imageio.imread(f'{filename1}.tif')
    image2 = imageio.imread(f'{filename2}.tif')

    # Radon变换
    projections = radon_transform(image1)
    projection2 = radon_transform(image2)

    # 对投影数据进行滤波并展示傅里叶变换和滤波器
    filtered_projections = apply_filter(projections, show_fft=True)
