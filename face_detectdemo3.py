# 人脸识别系统之人脸采集模块

import cv2
import os
import numpy as np

# 清理旧的采集文件
def clean_old_files():
    if not os.path.exists("face"):
        os.makedirs("face")
    else:
        # 清空faces目录中的旧文件
        for filename in os.listdir("face"):
            if filename.startswith("myface_") and filename.endswith(".jpg"):
                os.remove(os.path.join("face", filename))

# 打开摄像头
camera = cv2.VideoCapture(0)

# 判断摄像头是否开启
if camera.isOpened() == True:
    # 打开成功
    print("摄像头打开成功")

    # 清理旧文件
    clean_old_files()

    count = 0
    successful_captures = 0  # 记录成功抓取的次数
    while count < 6:
        # 抓拍图像
        retval, image = camera.read()

        if retval == True:
            # 从图像中检测人脸
            count = count + 1
            print("正在抓取第", count, "张图像...")
            face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
            faces = face_cascade.detectMultiScale(image)

            num = 0
            for (x,y,w,h) in faces:
               # 将人脸图像保存到faces目录下
               img_copied = image[y:y+h, x:x+w]
               img_resized = cv2.resize(img_copied, dsize=(200,200))

               num = num + 1
               filename = "face\\myface_%02d_%02d.jpg" % (count, num)
               cv2.imwrite(filename, img_resized)
               print(f"人脸图像已保存: {filename}")

               # 标注矩形
               cv2.rectangle(image,
                             (x,y), # 矩形的左上角坐标
                             (x+w, y+h), # 矩形的右下角坐标
                             color=(255,0,0), # 矩形边线的颜色(BGR-蓝绿红）
                             thickness=2
                             )

            # 如果检测到人脸
            if len(faces) > 0:
                successful_captures += 1
                print(f"第{count}张图像抓取成功，检测到{len(faces)}个人脸")
            else:
                print(f"第{count}张图像抓取完成，但未检测到人脸")

            # 显示抓取到的图像
            cv2.imshow("capture image", image)

            # 延时2秒
            cv2.waitKey(2000)
        else:
            # 抓拍失败
            print("图像抓取失败，重新尝试...")
            continue # 继续，重新抓

    # 显示最终统计信息
    print(f"\n人脸采集完成！总共尝试{count}次，成功抓取含有人脸的图像{successful_captures}张")

    # 关闭摄像头
    camera.release()

    # 关闭打开的所有窗口
    cv2.destroyAllWindows()

# 打开失败
else:
    print("摄像头打开失败，请检查")
