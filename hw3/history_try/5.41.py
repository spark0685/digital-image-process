import numpy as np
import imageio.v2 as imageio
from numba import njit, prange
from scipy.fft import fft, ifft, fftshift, fftfreq

@njit(parallel=True)
def radon_transform(image):
    size = image.shape
    round_size = int(np.sqrt(size[0]**2 + size[1]**2))
    result = np.zeros((180, round_size), dtype=np.int64)

    for angle_ in prange(180):
        angle = np.deg2rad(angle_)
        cos = np.cos(angle)
        sin = np.sin(angle)

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

    # 使用缩放将结果转换到 0 到 255
    min_val = np.min(result)
    max_val = np.max(result)

    if max_val > min_val:  # 确保不除以零
        result_scaled = (result - min_val) / (max_val - min_val) * 255
    else:
        result_scaled = np.zeros_like(result, dtype=np.float64)

    result_new = result_scaled.astype(np.uint8)
    return result_new


# def create_filter(size):
#     # 生成Ram-Lak滤波器，使用rou的绝对值
#     filter_ = np.abs(np.fft.fftfreq(size))

#     return filter_

# # 创建一个什么都不干的滤波器
# def create_filter(size):
#     return np.ones(size, dtype=np.float64)

#创建一个Hamming窗滤波器，其中窗的原点位于0
def create_filter(size):
    filter_ = np.abs(fftfreq(size))
    hamming = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(size) / size)
    # 将Hamming窗口的原点移动到0
    hamming = np.roll(hamming, size // 2)
    filter_ = filter_ * hamming
    return filter_

# 对投影数据进行滤波
def apply_filter(projections):
    filtered_projections = np.zeros_like(projections, dtype=np.float64)
    filter_ = create_filter(projections.shape[1])

    for i in range(projections.shape[0]):
        # 对每个角度的投影数据应用傅里叶变换
        projection = projections[i]
        projection_fft = fft(projection)
        # 应用滤波器
        filtered_projection_fft = projection_fft * filter_
        # 反变换回时域
        filtered_projections[i] = np.real(ifft(filtered_projection_fft))
    
    return filtered_projections

# 逆变换 (反投影)
@njit(parallel=True)
def inverse_radon_transform(image, filtered_projections):
    size = image.shape
    result = np.zeros(size, dtype=np.float64)

    for x in range(size[0]):
        for y in range(size[1]):
            x_ = y - size[1] / 2
            y_ = -x + size[0] / 2
            for angle_ in range(180):
                angle = np.deg2rad(angle_)
                cos = np.cos(angle)
                sin = np.sin(angle)
                rou = x_ * cos + y_ * sin
                if rou > 0:
                    rou = int(rou + 0.5)
                else:
                    rou = int(rou - 0.5)
                if np.abs(rou) > filtered_projections.shape[1] // 2:
                    continue
                result[x][y] += filtered_projections[angle_][rou + filtered_projections.shape[1] // 2 - 1]

    # 使用缩放将结果转换到 0 到 255
    min_val = np.min(result)
    max_val = np.max(result)

    if max_val > min_val:  # 确保不除以零
        result_scaled = (result - min_val) / (max_val - min_val) * 255
    else:
        result_scaled = np.zeros_like(result, dtype=np.float64)

    result = result_scaled.astype(np.uint8)
    return result

# 主函数
if __name__ == '__main__':
    filename1 = 'pic/Fig0539(a)(vertical_rectangle)'
    filename2 = 'pic/Fig0539(c)(shepp-logan_phantom)'

    # 读取TIFF图像
    image1 = imageio.imread(f'{filename1}.tif')
    image2 = imageio.imread(f'{filename2}.tif')

    # Radon变换
    projection1 = radon_transform(image1)
    projection2 = radon_transform(image2)

    # 滤波
    filtered_projection1 = apply_filter(projection1)
    filtered_projection2 = apply_filter(projection2)

    # 保存滤波后的投影结果
    imageio.imwrite('result/vertical_rectangle_filtered.png', filtered_projection1.astype(np.uint8))
    imageio.imwrite('result/shepp_logan_filtered.png', filtered_projection2.astype(np.uint8))

    # 反投影重建图像
    reconstructed1 = inverse_radon_transform(image1, filtered_projection1)
    reconstructed2 = inverse_radon_transform(image2, filtered_projection2)

    # 保存重建的图像
    imageio.imwrite('result/vertical_rectangle_reconstructed.png', reconstructed1)
    imageio.imwrite('result/shepp_logan_reconstructed.png', reconstructed2)

    print('Finish')
