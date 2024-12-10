import cv2
import os
import numpy as np
from sklearn.naive_bayes import GaussianNB

# 使用贝叶斯分类器对多光谱数据进行分类
def process(image_1, image_2, image_3, image_4, mask_1, mask_2, mask_3):
    # 得到四维数据
    data = np.zeros((image_1.shape[0], image_1.shape[1], 4))
    data[:, :, 0] = image_1
    data[:, :, 1] = image_2
    data[:, :, 2] = image_3
    data[:, :, 3] = image_4

    # 将图像数据转换为二维数组形式，每一行是一个像素的四个光谱值
    data_reshaped = data.reshape(-1, 4)

    # 将mask_1, mask_2, mask_3用于提取训练数据（假设它们是标签）
    # 假设每个mask对应一个类别标签
    # 这里mask_1, mask_2, mask_3是二值的，每个像素的位置如果在mask上为1则属于该类别
    labels_1 = mask_1.reshape(-1)
    labels_2 = mask_2.reshape(-1)
    labels_3 = mask_3.reshape(-1)

    # 合并标签为最终的训练标签
    labels = np.zeros(labels_1.shape)
    labels[labels_1 == 255] = 1  # 类别1
    labels[labels_2 == 255] = 2  # 类别2
    labels[labels_3 == 255] = 3  # 类别3

    # 过滤掉类别0的样本（背景类）
    valid_indices = labels > 0
    train_data = data_reshaped[valid_indices]
    train_labels = labels[valid_indices]

    # 训练贝叶斯分类器
    classifier = GaussianNB()
    classifier.fit(train_data, train_labels)

    # 使用训练好的分类器进行预测
    predictions = classifier.predict(data_reshaped)

    # 将预测结果重新调整为图像形状
    classified_image = predictions.reshape(image_1.shape)




    result_img1 = np.zeros_like(classified_image)
    result_img2 = np.zeros_like(classified_image)
    result_img3 = np.zeros_like(classified_image)

    result_img1[classified_image == 1] = 255
    result_img2[classified_image == 2] = 255
    result_img3[classified_image == 3] = 255


    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(labels, classifier.predict(data_reshaped)))

    result_img4 = np.zeros_like(classified_image)

    # 将mask_123中预测错误的像素标记为0
    img1 = mask_1.copy()
    img2 = mask_2.copy()
    img3 = mask_3.copy()

    img1[classified_image != 1] = 0
    img2[classified_image != 2] = 0
    img3[classified_image != 3] = 0

    result_img4 = img1 + img2 + img3

    return result_img1, result_img2, result_img3, result_img4



current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

image_path_1 = r"pic\\Fig1213(a)(WashingtonDC_Band1_512).tif"
image_path_2 = r"pic\\Fig1213(b)(WashingtonDC_Band2_512).tif"
image_path_3 = r"pic\\Fig1213(c)(WashingtonDC_Band3_512).tif"
image_path_4 = r"pic\\Fig1213(d)(WashingtonDC_Band4_512).tif"
image_path_5 = r"pic\\Fig1213(e)(Mask_B1_without_numbers).tif"
image_path_6 = r"pic\\Fig1213(e)(Mask_B2_without_numbers).tif"
image_path_7 = r"pic\\Fig1213(e)(Mask_B3_without_numbers).tif"
image_path_8 = r"pic\\Fig1213(e)(Mask_Composite_without_numbers).tif"

image_1 = cv2.imread(image_path_1, cv2.IMREAD_GRAYSCALE)
image_2 = cv2.imread(image_path_2, cv2.IMREAD_GRAYSCALE)
image_3 = cv2.imread(image_path_3, cv2.IMREAD_GRAYSCALE)
image_4 = cv2.imread(image_path_4, cv2.IMREAD_GRAYSCALE)
image_5 = cv2.imread(image_path_5, cv2.IMREAD_GRAYSCALE)
image_6 = cv2.imread(image_path_6, cv2.IMREAD_GRAYSCALE)
image_7 = cv2.imread(image_path_7, cv2.IMREAD_GRAYSCALE)
image_8 = cv2.imread(image_path_8, cv2.IMREAD_GRAYSCALE)

# 处理图像
result1,result2,result3, result4= process(image_1, image_2, image_3, image_4, image_5, image_6, image_7)

# 保存处理后的图像
cv2.imwrite("result\\1-1.jpg", result1)
cv2.imwrite("result\\1-2.jpg", result2)
cv2.imwrite("result\\1-3.jpg", result3)
cv2.imwrite("result\\1-4.jpg", result4)

