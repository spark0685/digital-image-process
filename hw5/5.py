import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def process(image):
    # 使用半径为5的形态学滤波器做平滑
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    smooth_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    # 对图片使用不同半径的圆盘做开操作，统计表面区域差值
    open_images = []
    for i in range(1, 8):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (i * 10, i * 10))
        open_image = cv2.morphologyEx(smooth_image, cv2.MORPH_OPEN, kernel)
        open_images.append(open_image)

    
    # 返回半径为10，20，25，30的圆盘开操作结果
    return smooth_image, open_images


current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
image_path = r"pic\\Fig0941(a)(wood_dowels).tif"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 处理图像
smooth_image, open_images = process(image)

# 保存原图和半径为10，20，25，30的圆盘开操作结果
cv2.imwrite('result\\5-1.jpg', image)
cv2.imwrite('result\\5-2.jpg', smooth_image)
cv2.imwrite(f'result\\5-3.jpg', open_images[1])
cv2.imwrite(f'result\\5-4.jpg', open_images[3])
cv2.imwrite(f'result\\5-5.jpg', open_images[4])
cv2.imwrite(f'result\\5-6.jpg', open_images[5])



