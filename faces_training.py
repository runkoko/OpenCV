# 人脸识别系统之人脸训练模块

import cv2
import os
import numpy as np
from cv2 import face
# 读入多张人脸
faces = []
labels = []

# 将faces目录下的jpg文件作为素材进行训练
for f in os.listdir("faces"):
    # 去除非jpg
    if '.jpg' not in f:
        continue

    # 读取图像
    faces.append(cv2.imread("faces\\" + f, cv2.IMREAD_GRAYSCALE))

    # 添加标签
    labels.append(0)

# 判断一下是否有人脸图像
if len(faces) > 0:
    # 创建识别器
    recognizer = cv2.face.LBPHFaceRecognizer().create()

    # 训练样本模型
    recognizer.train(faces, np.array(labels))

    # 保存训练模型
    recognizer.save("myface_model.yml")

    print("人脸特征训练完成，已保存到yml文件中！")
    

else:
    print("没有人脸图片文件")
