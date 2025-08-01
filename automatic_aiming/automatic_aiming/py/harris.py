import numpy as np
import cv2

# 初始化摄像头（尝试ID 0或1）
cap = cv2.VideoCapture(2)

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# 创建窗口
cv2.namedWindow('Camera View')
cv2.namedWindow('Harris detect')

while True:
    # 读取一帧
    ret, frame = cap.read()
    
    # 检查是否成功获取帧
    if not ret:
        print("无法获取帧，退出...")
        break
    
    # 显示原始帧
    cv2.imshow('Camera View', frame)
    
    # 图像处理（角点检测）
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray', gray)

    gray = np.float32(gray)    
    dst = cv2.cornerHarris(gray, 2, 3, 0.02)
    '''
        img - 输入图像，应为 float32 类型的灰度图。
        blockSize - 角点检测所考虑的邻域大小。
        ksize - Sobel 导数的内核大小。
        k - Harris 检测器方程中的自由参数。
    '''
    cv2.imshow('dst', dst)
    dst = cv2.dilate(dst, None)  # 膨胀结果，便于标记角点
    
    # 标记角点（绿色）
    frame[dst > 0.01 * dst.max()] = [0, 255, 0]
    
    # 显示处理后的帧
    cv2.imshow('Harris detect', frame)
    
    # 按ESC键退出循环（等待1ms）
    key = cv2.waitKey(1)
    if key & 0xFF == 27:  # ESC键的ASCII码是27
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()