import numpy as np
import imageio.v2 as imageio
from numba import njit, prange
from scipy.fft import fft, ifft, fftfreq

# Radon变换
def transform(image):
    size = image.shape
    round_size = int(np.sqrt(size[0]**2 + size[1]**2))
    result = np.zeros((180, round_size), dtype=int)

    for angle_ in range(180):
        for round in range(-round_size//2, round_size//2 + 1):
            if np.abs(angle_ - 90) > 45:
                angle = np.deg2rad(angle_)
                x = np.arange(size[0])
                y_ = -x + size[0] / 2

                x_ = round / np.cos(angle) - y_ * np.tan(angle)
                y = (x_ + size[1] / 2).astype(int)

                valid_mask = (y >= 0) & (y < size[1])
                result[angle_][round + (round_size // 2) - 1] += image[x[valid_mask], y[valid_mask]].sum()
            else:
                angle = np.deg2rad(angle_)
                y = np.arange(size[1])
                x_ = (y - size[1] / 2).astype(int)
                y_ = round / np.sin(angle) - x_ / np.tan(angle)
                x = (-y_ + size[0] / 2).astype(int)
                valid_mask = (x >= 0) & (x < size[0])
                    
                result[angle_][round + (round_size // 2) - 1] += image[x[valid_mask], y[valid_mask]].sum()

    # 结果缩放
    min_val = np.min(result)
    max_val = np.max(result)

    if max_val > min_val:  # 确保不除以零
        result_scaled = (result - min_val) / (max_val - min_val) * 255
    else:
        result_scaled = np.zeros_like(result, dtype=np.float64)

    result_new = result_scaled.astype(np.uint8)
    return result_new

# 创建滤波器（可选汉明窗）
def create_filter(size, use_hamming=True):
    filter_ = np.abs(fftfreq(size))
    if use_hamming:
        hamming = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(size) / size)
        hamming = np.roll(hamming, size // 2)
        filter_ = filter_ * hamming
    return filter_

# 应用滤波器
def apply_filter(projections, use_hamming=True):
    filtered_projections = np.zeros_like(projections, dtype=np.float64)
    filter_ = create_filter(projections.shape[1], use_hamming)

    for i in range(projections.shape[0]):
        projection = projections[i]
        projection_fft = fft(projection)
        filtered_projection_fft = projection_fft * filter_
        filtered_projections[i] = np.real(ifft(filtered_projection_fft))
    
    return filtered_projections

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
                rou = int(round(rou))
                
                if np.abs(rou) > filtered_projections.shape[1] // 2:
                    continue
                result[x][y] += filtered_projections[angle_][rou + filtered_projections.shape[1] // 2 - 1]

    min_val = np.min(result)
    max_val = np.max(result)

    if max_val > min_val:
        result_scaled = (result - min_val) / (max_val - min_val) * 255
    else:
        result_scaled = np.zeros_like(result, dtype=np.float64)

    result = result_scaled.astype(np.uint8)
    return result

# 主函数
if __name__ == '__main__':
    filename1 = 'pic/Fig0539(a)(vertical_rectangle)'
    filename2 = 'pic/Fig0539(c)(shepp-logan_phantom)'

    image1 = imageio.imread(f'{filename1}.tif')
    image2 = imageio.imread(f'{filename2}.tif')

    # Radon变换
    projection1 = transform(image1)
    projection2 = transform(image2)

    # 滤波（带汉明窗）
    filtered_projection1_hamming = apply_filter(projection1, use_hamming=True)
    filtered_projection2_hamming = apply_filter(projection2, use_hamming=True)

    # 滤波（不带汉明窗）
    filtered_projection1_no_hamming = apply_filter(projection1, use_hamming=False)
    filtered_projection2_no_hamming = apply_filter(projection2, use_hamming=False)

    # 保存带汉明窗的滤波结果
    imageio.imwrite('result_3_4/vertical_rectangle_filtered_hamming.png', filtered_projection1_hamming.astype(np.uint8))
    imageio.imwrite('result_3_4/shepp_logan_filtered_hamming.png', filtered_projection2_hamming.astype(np.uint8))

    # 保存不带汉明窗的滤波结果
    imageio.imwrite('result_3_4/vertical_rectangle_filtered_no_hamming.png', filtered_projection1_no_hamming.astype(np.uint8))
    imageio.imwrite('result_3_4/shepp_logan_filtered_no_hamming.png', filtered_projection2_no_hamming.astype(np.uint8))

    # 反投影重建图像
    reconstructed1_hamming = inverse_radon_transform(image1, filtered_projection1_hamming)
    reconstructed2_hamming = inverse_radon_transform(image2, filtered_projection2_hamming)

    reconstructed1_no_hamming = inverse_radon_transform(image1, filtered_projection1_no_hamming)
    reconstructed2_no_hamming = inverse_radon_transform(image2, filtered_projection2_no_hamming)

    # 保存重建的图像
    imageio.imwrite('result_3_4/vertical_rectangle_reconstructed_hamming.png', reconstructed1_hamming)
    imageio.imwrite('result_3_4/shepp_logan_reconstructed_hamming.png', reconstructed2_hamming)

    imageio.imwrite('result_3_4/vertical_rectangle_reconstructed_no_hamming.png', reconstructed1_no_hamming)
    imageio.imwrite('result_3_4/shepp_logan_reconstructed_no_hamming.png', reconstructed2_no_hamming)

    print('Finish')
