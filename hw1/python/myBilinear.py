import numpy as np

def bilinear(img_1, n):
    # 假设 img_1 是已加载的图像 (使用 NumPy 或 OpenCV 加载)
    h_1, w_1, _ = img_1.shape  # 获取图像尺寸
    h_2 = int(h_1 * n)  # 计算缩放后的高度
    w_2 = int(w_1 * n)  # 计算缩放后的宽度

    # 创建一个大小为 h_2 x w_2 的空图像 (黑色，灰度图)
    img_2 = np.zeros((h_2, w_2, 3), dtype=np.uint8)

    # 下面写双线性插值实现代码
    for i in range(h_2):
        for j in range(w_2):
            x = i / n
            y = j / n
            x1 = int(x)
            y1 = int(y)
            x2 = min(x1 + 1, h_1 - 1)
            y2 = min(y1 + 1, w_1 - 1)
            a = x - x1
            b = y - y1
            img_2[i, j] = (1 - a) * (1 - b) * img_1[x1, y1] + a * (1 - b) * img_1[x2, y1] + (1 - a) * b * img_1[x1, y2] + a * b * img_1[x2, y2]

    return img_2
