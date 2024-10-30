import numpy as np

def nearest(img_1, n):
    # 假设 img_1 是已加载的图像 (使用 NumPy 或 OpenCV 加载)
    h_1, w_1, _ = img_1.shape  # 获取图像尺寸
    h_2 = int(h_1 * n)  # 计算缩放后的高度
    w_2 = int(w_1 * n)  # 计算缩放后的宽度

    # 创建一个大小为 h_2 x w_2 的空图像 
    img_2 = np.zeros((h_2, w_2, 3), dtype=np.uint8)

    # 下面写最近邻插值实现代码
    for i in range(h_2):
        for j in range(w_2):
            x = int(i / n)
            y = int(j / n)

            img_2[i, j] = img_1[x, y]
    return img_2

