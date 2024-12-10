import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def process(image):
    # 使用半径为5的形态学滤波器做平滑
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    smooth_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    # 对图片使用不同半径的圆盘做开操作，计算表面区域的像素值之和的差值
    open_images = {}
    surface_diffs = []  # 用于存储每个半径之间的像素和差值
    prev_sum = np.sum(smooth_image)  # 初始平滑图像的像素和
    
    for radius in range(10, 41):  # 半径从10到40的整数
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2, radius * 2))
        open_image = cv2.morphologyEx(smooth_image, cv2.MORPH_OPEN, kernel)
        open_images[radius] = open_image

        # 计算当前开操作结果的像素值之和
        current_sum = np.sum(open_image)
        
        # 计算与前一个半径图像之间的像素值之和的差
        surface_diff = prev_sum - current_sum
        surface_diffs.append(surface_diff)

        # 更新 prev_sum 为当前的像素值之和
        prev_sum = current_sum
    
    return smooth_image, open_images, surface_diffs

# 设置路径
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
image_path = r"pic\\Fig0941(a)(wood_dowels).tif"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 处理图像
smooth_image, open_images, surface_diffs = process(image)

# 保存原图和半径为10，20，25，30的圆盘开操作结果
os.makedirs("result", exist_ok=True)  # 确保保存目录存在
cv2.imwrite('result\\5-1.jpg', image)
cv2.imwrite('result\\5-2.jpg', smooth_image)
cv2.imwrite('result\\5-3.jpg', open_images[10])  # 半径为10的开操作
cv2.imwrite('result\\5-4.jpg', open_images[20])  # 半径为20的开操作
cv2.imwrite('result\\5-5.jpg', open_images[25])  # 半径为25的开操作
cv2.imwrite('result\\5-6.jpg', open_images[30])  # 半径为30的开操作

# 可视化并保存表面区域差值
radii = list(range(10, 41))
plt.figure()
plt.plot(radii[1:], surface_diffs[1:], marker='o', linestyle='-', color='b')  # 排除第一个差值，以半径10开始
plt.xlabel("Radius")
plt.ylabel("Surface Area Difference (Pixel Sum Difference)")
plt.title("Surface Area Difference vs Radius")
plt.grid()
plt.savefig('result\\5-7.jpg')
plt.close()
