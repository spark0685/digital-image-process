import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


def threshold_process(image, percentile=72):
    # 计算图像的特定百分位值
    threshold_value = np.percentile(image, percentile)
    
    # 对图片做阈值处理，使用计算得到的百分位值作为阈值
    _, thresholded_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    
    return thresholded_image


def process(image):
    # 对图片做阈值处理
    threshold_image = threshold_process(image)

    # 使用半径40的圆盘做顶帽变换
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (80, 80))  # 半径40的圆盘
    
    open_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    # # 顶帽变换: 原始图像减去开操作的结果
    tophat_image = image + 100 - open_image

    # open_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    # tophat_image = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)

    thresholded_tophat_image = threshold_process(tophat_image)
    
    return threshold_image, open_image, tophat_image, thresholded_tophat_image

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
image_path = r"pic\\Fig0940(a)(rice_image_with_intensity_gradient).tif"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 处理图像
threshold_image, open_image, tophat_image, thresholded_tophat_image = process(image)

# 保存结果
cv2.imwrite('result\\4-1.jpg', image)
cv2.imwrite('result\\4-2.jpg', threshold_image)
cv2.imwrite('result\\4-3.jpg', open_image)
cv2.imwrite('result\\4-4.jpg', tophat_image)
cv2.imwrite('result\\4-5.jpg', thresholded_tophat_image)


