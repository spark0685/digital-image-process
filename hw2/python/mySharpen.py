import cv2
import numpy as np


def Sharpen(image):
    h, w = image.shape[:2]
    
    # 把图片转换为单通道灰度图
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], np.float32)

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)

    #中间为1，周围为0的kernel
    #kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], np.float32)

    result = image.copy()
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            neighbors = image[i - 1 : i + 2, j - 1 : j + 2]
            temp = np.sum(neighbors * kernel)
            if temp > 255:
                temp = 255
            elif temp < 0:
                temp = 0
            result[i, j] = temp
    return result
