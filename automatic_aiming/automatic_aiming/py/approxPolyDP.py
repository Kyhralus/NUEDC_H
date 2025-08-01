"""
    1. 实现对基础图像的多边形拟合，
    2. 同时画出它的外接正方形和圆形
    3. 作为roi区域，并放大图像

"""
import cv2
import numpy as np
import time

# ============= 实时处理 ============
# 闭运算卷积核
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# 打开摄像头，0 表示默认摄像头
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("无法打开摄像头，请检查设备连接。")
    exit()

while True:
    # 读取一帧图像
    start = time.time()
    ret, img_raw = cap.read()
    if not ret:
        print("无法获取图像帧，退出。")
        break

    # 转灰度
    img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
    # 双边滤波
    img = cv2.bilateralFilter(img, 9, 10, 10)
    cv2.imshow('bilateralFilter', img)
    # 闭运算
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('morphologyEx', img)
    # 边缘检测
    edged = cv2.Canny(img, 50, 150)
    cv2.imshow('Canny', edged)

    # 找轮廓
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 找到面积最大的轮廓
    # largest_contour = max(contours, key=cv2.contourArea)
    for contour in contours:
        # 计算参数，多边形逼近
        epsilon = 0.03 * cv2.arcLength(contour, True) # 调大系数，得到的多边形越粗糙
        
        approx = cv2.approxPolyDP(contour, epsilon, True) # 得到的是多边形的角点列表
        # print(approx)
        if len(approx) >= 3:
            # 画轮廓
            cv2.drawContours(img_raw, [approx], 0, (0, 255, 0), 2)
            # 画角点
            for point in approx:
                cv2.circle(img_raw, tuple(point[0]), 5, (0, 0, 255), -1)
            
            # 外接矩形/垂直边界矩形（也称包围矩形或外接矩形）
            x,y,w,h = cv2.boundingRect(contour)
            cv2.rectangle(img_raw,(x,y),(x+w,y+h),(255,255,0),2)
            print(f"外接矩形: {x,y,w,h}")
            # 计算最小外接矩形
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            # 打印最小外接矩形的信息
            print("最小外接矩形信息：")
            print(f"中心点: {rect[0]},宽度: {rect[1][0]},高度: {rect[1][1]}")
            print(f"旋转角度: {rect[2]}")
            # 在原图上绘制最小外接矩形
            cv2.drawContours(img_raw, [box], 0, (0,25,25), 2)  # 使用红色绘制
            # 外接圆
            (x,y),radius = cv2.minEnclosingCircle(contour)
            center = (int(x),int(y))
            radius = int(radius)
            cv2.circle(img_raw,center,radius,(25,0,255),2)

    # 显示处理后的图像
    cv2.imshow('Processed Image', img_raw)
    print(f"耗时: {time.time()-start:.3f}秒")
    # 按 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭所有窗口
cap.release()
cv2.destroyAllWindows()