import numpy as np
import cv2


def Average(image, kernel_size=3, iterations=5):
    h, w = image.shape[:2]
    # 把图片转换为单通道灰度图
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)

    result = image.copy()

    for _ in range(iterations):

        for i in range(kernel_size // 2, h - kernel_size // 2):
            for j in range(kernel_size // 2, w - kernel_size // 2):
                neighbors = image[
                    i - kernel_size // 2 : i + kernel_size // 2 + 1,
                    j - kernel_size // 2 : j + kernel_size // 2 + 1,
                ]
                result[i, j] = np.sum(neighbors * kernel)

        image = result.copy()

    return result
