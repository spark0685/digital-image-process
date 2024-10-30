import numpy as np
import math

# 计算双三次插值的权重
def bicubic_kernel(x):
    x = abs(x)
    if x <= 1:
        return 1 - 2 * (x ** 2) + (x ** 3)
    elif x < 2:
        return 4 - 8 * x + 5 * (x ** 2) - (x ** 3)
    else:
        return 0
    
def bicubic(img_1, n):
    # 假设 img_1 是已加载的图像 (使用 NumPy 或 OpenCV 加载)
    h_1, w_1, c = img_1.shape  # 获取图像尺寸
    h_2 = int(h_1 * n)  # 计算缩放后的高度
    w_2 = int(w_1 * n)  # 计算缩放后的宽度

    # 创建一个大小为 h_2 x w_2 的空图像 (黑色，灰度图)
    img_2 = np.zeros((h_2, w_2, 3), dtype=np.uint8)

    # 下面写双三次插值实现代码
    for i in range(h_2):
        for j in range(w_2):
            # 计算目标像素在原图中的位置
            x = i / n
            y = j / n

            # 确定原图中的16个邻近像素的坐标
            x0 = int(math.floor(x))
            x1 = min(x0 + 1, h_1 - 1)
            y0 = int(math.floor(y))
            y1 = min(y0 + 1, w_1 - 1)

            # 计算在原图中相对位置
            dx = x - x0
            dy = y - y0

            # 对每个颜色通道进行插值
            for k in range(c):
                # 计算双三次插值的结果
                value = 0
                for ii in range(-1, 3):
                    for jj in range(-1, 3):
                        value += img_1[min(max(x0 + ii, 0), h_1 - 1), min(max(y0 + jj, 0), w_1 - 1), k] * bicubic_kernel(ii - dx) * bicubic_kernel(dy - jj)

                img_2[i, j, k] = np.clip(value, 0, 255)
           




    return img_2



