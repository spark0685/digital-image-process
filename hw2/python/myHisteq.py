import numpy as np
import cv2

import matplotlib.pyplot as plt


def Histeq(image, n):

    total = image.shape[0] * image.shape[1]
    hist, bin = np.histogram(image.flatten(), bins=256, range=[0, 256])

    print(hist)
    # 绘制直方图
    plt.figure(figsize=(10,5))
    plt.title("灰度图像直方图")
    plt.xlabel("像素值")
    plt.ylabel("频数")
    plt.bar(bin[:-1], hist, width=1, edgecolor='black')
    plt.xlim([0,256])
    plt.show()

    Pr = hist / total

    out_img = np.copy(image)
    sumk = 0.
    for i in range(256):
        sumk += Pr[i]
        out_img[image == i] = np.round((n - 1) * sumk) *(256 / n)
    return out_img
