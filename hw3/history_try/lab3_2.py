import numpy as np

import imageio


def transform(image):
    size = image.shape
    round_size = int(np.sqrt(size[0]**2 + size[1]**2))
    result = np.zeros((180, round_size), dtype=int)

    # 计算图像的前缀和
    img_sum = image.copy()
    for i in range(1, size[0]):
        img_sum[i] += img_sum[i - 1]

    for angle_ in range(180):
        angel = np.deg2rad(angle_)
        cos = np.cos(angel)
        sin = np.sin(angel)
        print(angle_)

        # 并行处理
        for x in range(size[0]):
            for y in range(size[1]):
                x_ = y - size[1] / 2
                y_ = -x + size[0] / 2
                rou = int(x_ * cos + y_ * sin + 0.5)  # 四舍五入
                if rou > np.abs(round_size // 2):
                    continue
                result[angle_][rou + round_size // 2 - 1] += image[x][y]

    # 确保结果数组在合适的范围内，并转换为 uint8
    return result

# 主函数
if __name__ == '__main__':    
    filename1 = 'Fig0539(a)(vertical_rectangle)'

    filename2 = 'Fig0539(c)(shepp-logan_phantom)'

    # 读取TIFF图像
    image1 = imageio.imread(f'{filename1}.tif')
    image2 = imageio.imread(f'{filename2}.tif')
    result1 = transform(image1)
    # result2 = transform(image2)

    # 保存图片
    imageio.imwrite(f'{filename1}_transform.png', result1)
    # imageio.imwrite(f'{filename2}_transform.png', result2)
