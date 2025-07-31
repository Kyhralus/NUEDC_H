import cv2
import time
# 打开摄像头
cap = cv2.VideoCapture(0)  # 建议加cv2.CAP_DSHOW防止自动曝光锁定

# ================== 设置相机参数 ==================

# 关键设置：指定MJPG编码格式（高分辨率通常需要此格式）
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

# 设置目标分辨率（1920x1080）
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)


# 等待设置生效
time.sleep(1)
# ================== 实时显示 ==================
while True:
    ret, frame = cap.read()
    if not ret:
        print("无法读取摄像头画面")
        break

    cv2.imshow("Camera Live View", frame)

    # 按 'q' 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
