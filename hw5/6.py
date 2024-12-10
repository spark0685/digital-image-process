import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def process(image):
    # 使用半径为30的圆盘做闭操作
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (60, 60))
    close_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    # 使用半径为60的圆盘做开操作
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (120, 120))
    open_image = cv2.morphologyEx(close_image, cv2.MORPH_OPEN, kernel)

    # 使用大小3*3的结构元对图像执行形态学梯度操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gradient_image = cv2.morphologyEx(open_image, cv2.MORPH_GRADIENT, kernel)

    # 把边界叠加到原图，gradient_image中非0的像素值为边界，在结果中为255
    boundary_image = np.where(gradient_image != 0, 255, 0).astype(np.uint8)
    boundary_overlay = cv2.addWeighted(image, 1, boundary_image, 1, 0)

    return close_image, open_image, gradient_image, boundary_overlay


current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
image_path = r"pic\\Fig0943(a)(dark_blobs_on_light_background).tif"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 处理图像
close_image, open_image, gradient_image, boundary_image = process(image)

# 保存结果
cv2.imwrite('result\\6-1.jpg', image)
cv2.imwrite('result\\6-2.jpg', close_image)
cv2.imwrite('result\\6-3.jpg', open_image)
cv2.imwrite('result\\6-4.jpg', gradient_image)
cv2.imwrite('result\\6-5.jpg', boundary_image)




