import cv2
import time
import os

def capture_images():
    try:
        # 检查摄像头设备
        if not os.path.exists('/dev/video0'):
            print("错误：未检测到摄像头设备 /dev/video0")
            print("请检查摄像头是否已正确连接")
            return
        
        # 尝试打开摄像头
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("错误：无法打开摄像头")
            print("可能原因：")
            print("1. 摄像头被其他程序占用")
            print("2. 权限不足，请尝试使用sudo运行")
            print("3. 摄像头驱动问题")
            return
        
        # 关键设置：指定MJPG编码格式（高分辨率通常需要此格式）
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        # 设置目标分辨率（1920x1080）
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 120)
        # 等待设置生效
        time.sleep(1)
        
        # 检查实际分辨率
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"摄像头已打开，分辨率: {width}x{height}")
        
        img_count = 10
        
        def crop_center_and_flip(img, crop_w=640, crop_h=480, flip_code=1):
            """
            从图像中心裁剪指定尺寸，然后进行翻转
            
            参数:
                img: 输入图像
                crop_w: 裁剪宽度
                crop_h: 裁剪高度
                flip_code: 翻转模式（1:水平翻转, 0:垂直翻转, -1:180度翻转）
            返回:
                裁剪并翻转后的图像
            """
            # 中心裁剪
            h, w = img.shape[:2]
            start_x = max(0, (w - crop_w) // 2)
            start_y = max(0, (h - crop_h) // 2)
            cropped = img[start_y:start_y+crop_h, start_x:start_x+crop_w]
            
            # 翻转图像（OpenCV底层优化，速度快）
            flipped = cv2.flip(cropped, flip_code)
        
            return flipped
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法获取帧")
                break
            # 裁剪中心区域
            cropped = crop_center_and_flip(frame, 960, 720, 0)
            # 在中心画圆点
            cv2.circle(cropped, (480, 360), 3, (0, 255, 255), -1)
            cv2.imshow('Camera View', cropped)
            # cv2.imshow('Camera View 1290X1080', frame)
            key = cv2.waitKey(1)
            if key == 27:  # ESC键退出
                break
            elif key == ord('s'):  # 按s键开始保存
                print("开始保存10张图片...")
                
                # 确保data目录存在
                if not os.path.exists('data'):
                    os.makedirs('data')
                
                for i in range(2):
                    ret, frame = cap.read()
                    if not ret:
                        print("无法获取帧")
                        break
                    cropped = crop_center(frame, 640, 480)
                    filename = f"/home/orangepi/ros2_workspace/opencv_demo/images/{img_count}.jpg"
                    if cv2.imwrite(filename, cropped):
                        print(f"已保存: {filename}")
                        img_count += 1
                    else:
                        print(f"保存失败: {filename}")
                    cv2.circle(cropped, (320, 240), 5, (0, 255, 255), -1)
                    cv2.imshow('Camera View', cropped)
                    if cv2.waitKey(1) == 27:
                        break
                    time.sleep(0.5)
                
                print("10张图片保存完成")
        
    except Exception as e:
        print(f"程序运行出错: {str(e)}")
    finally:
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
        print("程序结束")

if __name__ == "__main__":
    capture_images()