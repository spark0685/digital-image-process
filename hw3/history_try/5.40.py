import numpy as np
import imageio.v2 as imageio
from numba import njit, prange

@njit(parallel=True)
def transform(image):
    size = image.shape
    round_size = int(np.sqrt(size[0]**2 + size[1]**2))
    result = np.zeros((180, round_size), dtype=np.int64)

    for angle_ in prange(180):
        angel = np.deg2rad(angle_)
        cos = np.cos(angel)
        sin = np.sin(angel)

        for x in range(size[0]):
            for y in range(size[1]):
                x_ = y - size[1] / 2
                y_ = -x + size[0] / 2
                rou = x_ * cos + y_ * sin
                if rou > 0:
                    rou = int(x_ * cos + y_ * sin + 0.5)  # 四舍五入
                else:
                    rou = int(x_ * cos + y_ * sin - 0.5)
                if np.abs(rou) > round_size // 2:
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

# 逆变换
@njit(parallel=True)
def transform2(image, projection):
    size = image.shape
    # 创建变换后的结果
    result = np.zeros(size, dtype=np.int64)

    for x in range(size[0]):
        print(x)
        for y in range(size[1]):
            x_ = y - size[1] / 2
            y_ = -x + size[0] / 2
            for angle_ in range(180):
                angel = np.deg2rad(angle_)
                cos = np.cos(angel)
                sin = np.sin(angel)
                rou = x_ * cos + y_ * sin
                if rou > 0:
                    rou = int(x_ * cos + y_ * sin + 0.5)
                else:
                    rou = int(x_ * cos + y_ * sin - 0.5)
                if np.abs(rou) > projection.shape[1] // 2:
                    continue
                result[x][y] += projection[angle_][rou + projection.shape[1] // 2 - 1]

    # 使用缩放将结果转换到 0 到 255
    min_val = np.min(result)
    max_val = np.max(result)

    # 线性缩放
    if max_val > min_val:  # 确保不除以零
        result_scaled = (result - min_val) / (max_val - min_val) * 255
    else:
        result_scaled = np.zeros_like(result, dtype=np.float64)

    result = result_scaled.astype(np.uint8)
    return result
    


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

    image3 = transform2(image1, result1)
    image4 = transform2(image2, result2)

    # 保存结果
    imageio.imwrite('result\\vertical_rectangle_transform2.png', image3)
    imageio.imwrite('result\\shepp-logan_phantom_transform2.png', image4)

    print('Finish')
