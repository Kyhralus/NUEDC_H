import cv2
import numpy as np

# 定义回调函数，当滑动条的值改变时调用，这里仅占位不做操作
def nothing(x):
    pass

# 打开摄像头，0 表示默认摄像头
cap = cv2.VideoCapture(2)

if not cap.isOpened():
    print("无法打开摄像头，请检查设备连接。")
else:
    # 创建一个窗口
    cv2.namedWindow('Thresholded Image')

    # 创建三个滑动条：下限阈值、上限阈值、反转开关
    cv2.createTrackbar('Lower Threshold', 'Thresholded Image', 10, 255, nothing)
    cv2.createTrackbar('Upper Threshold', 'Thresholded Image', 90, 255, nothing)
    cv2.createTrackbar('Invert', 'Thresholded Image', 0, 1, nothing)  # 添加反转滑动条

    while True:
        # 读取一帧图像
        ret, frame = cap.read()

        if not ret:
            print("无法获取图像帧，退出。")
            break

        # 将图像转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 获取滑动条的当前值
        lower_thresh = cv2.getTrackbarPos('Lower Threshold', 'Thresholded Image')
        upper_thresh = cv2.getTrackbarPos('Upper Threshold', 'Thresholded Image')
        invert = cv2.getTrackbarPos('Invert', 'Thresholded Image')  # 获取反转开关状态

        # 使用 inRange 函数进行阈值处理
        mask = cv2.inRange(gray, lower_thresh, upper_thresh)

        # 根据反转开关的值决定是否反转图像
        if invert:
            mask = cv2.bitwise_not(mask)  # 反转图像：黑色变白色，白色变黑色

        # 显示二值化后的图像
        cv2.imshow('Thresholded Image', mask)

        # 按 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放摄像头并关闭所有窗口
    cap.release()
    cv2.destroyAllWindows()