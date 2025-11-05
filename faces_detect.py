# 人脸识别系统之人脸采集模块

import cv2
import os
import numpy as np

# 打开摄像头
camera = cv2.VideoCapture(0)

# 判断摄像头是否开启
if camera.isOpened() == True:
    # 打开成功
    print("摄像头打开成功")

    count = 0
    while count < 6:
        # 抓拍图像
        retval, image = camera.read()

        if retval == True:
            # 从图像中检测人脸
            count = count + 1
            print("已抓取第", count, "张图像！")
            face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
            faces = face_cascade.detectMultiScale(image)

            num = 0
            for (x,y,w,h) in faces:
               # 将人脸图像保存到faces目录下
               img_copied = image[y:y+h, x:x+w]
               img_resized = cv2.resize(img_copied, dsize=(200,200))

               num = num + 1
               cv2.imwrite("faces\\myface_%02d_%02d.jpg" % (count, num), img_resized)

               # 标注矩形
               cv2.rectangle(image,
                             (x,y), # 矩形的左上角坐标
                             (x+w, y+h), # 矩形的右下角坐标
                             color=(255,0,0), # 矩形边线的颜色(BGR-蓝绿红）
                             thickness=2
                             )
            # 显示抓取到的图像
            cv2.imshow("capture image", image)

            # 延时2秒
            cv2.waitKey(2000)
        else:
            # 抓拍失败
            continue # 继续，重新抓

    # 关闭摄像头
    camera.release()

    # 关闭打开的所有窗口
    cv2.destroyAllWindows()


# 打开失败
else:
    print("摄像头打开失败，请检查")
