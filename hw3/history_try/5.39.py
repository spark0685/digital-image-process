import numpy as np
import imageio.v2 as imageio
from numba import njit, prange

@njit(parallel=True)
def transform(image):
    size = image.shape
    round_size = int(np.sqrt(size[0]**2 + size[1]**2))
    result = np.zeros((180, round_size), dtype=np.int64)

    for angle_ in range(180):
        print(f'angle: {angle_}')
        angel = np.deg2rad(angle_)
        cos = np.cos(angel)
        sin = np.sin(angel)

        for x in range(size[0]):
            for y in range(size[1]):
                x_ = y - size[1] / 2
                y_ = -x + size[0] / 2
                rou = int(x_ * cos + y_ * sin + 0.5)  # 四舍五入
                if rou > np.abs(round_size // 2):
                    continue
                result[angle_][rou + round_size // 2 - 1] += image[x][y]

    # 使用缩放将结果转换到 0 到 255
    min_val = np.min(result)
    max_val = np.max(result)

    # 线性缩放
    if max_val > min_val:  # 确保不除以零
        result_scaled = (result - min_val) / (max_val - min_val) * 255
    else:
        result_scaled = np.zeros_like(result, dtype=np.float64)

    result_new = result_scaled.astype(np.uint8)
    return result_new

# 主函数
if __name__ == '__main__':
    filename1 = 'pic\Fig0539(a)(vertical_rectangle)'
    filename2 = 'pic\Fig0539(c)(shepp-logan_phantom)'

    # 读取TIFF图像
    image1 = imageio.imread(f'{filename1}.tif')
    image2 = imageio.imread(f'{filename2}.tif')
    
    result1 = transform(image1)
    result2 = transform(image2)



    # 保存图片
    imageio.imwrite('result\\vertical_rectangle_transform.png', result1)
    imageio.imwrite('result\\shepp-logan_phantom_transform.png', result2)
