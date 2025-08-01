import cv2
import time
import os

def capture_images():
    try:

        
        # 尝试打开摄像头
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("错误：无法打开摄像头")
            return
        
        # 关键设置：指定MJPG编码格式（高分辨率通常需要此格式）
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        # 设置目标分辨率（1920x1080）
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # 等待设置生效
        time.sleep(1)
        
        # 检查实际生效的分辨率
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"摄像头已打开，实际分辨率: {width}x{height}")
        
        # 检查当前编码格式
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = chr(fourcc & 0xFF) + chr((fourcc >> 8) & 0xFF) + chr((fourcc >> 16) & 0xFF) + chr((fourcc >> 24) & 0xFF)
        print(f"当前编码格式: {fourcc_str}")
        
        # 确保保存目录存在
        save_dir = "/home/orangepi/ros2_workspace/opencv_demo/images"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        img_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法获取帧")
                break
            
            # 在画面中心绘制标记（适应1920x1080分辨率）
            center_x, center_y = width // 2, height // 2
            frame = cv2.circle(frame, (center_x, center_y), 5, (0, 255, 255), -1)
            cv2.imshow('Camera View', frame)
            
            key = cv2.waitKey(1)
            if key == 27:  # ESC键退出
                break
            elif key == ord('s'):  # 按s键保存2张图片（原代码中是2张，不是10张）
                print("开始保存图片...")
                
                for i in range(2):
                    ret, frame = cap.read()
                    if not ret:
                        print("无法获取帧")
                        break
                        
                    filename = f"{save_dir}/{img_count}.jpg"
                    if cv2.imwrite(filename, frame):
                        print(f"已保存: {filename}")
                        img_count += 1
                    else:
                        print(f"保存失败: {filename}")
                    
                    # 显示保存状态
                    frame = cv2.putText(frame, f"Saved {i+1}/2", (50, 50), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow('Camera View', frame)
                    if cv2.waitKey(500) == 27:  # 等待500ms，方便观察
                        break
                
                print("图片保存完成")
        
    except Exception as e:
        print(f"程序运行出错: {str(e)}")
    finally:
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
        print("程序结束")

if __name__ == "__main__":
    capture_images()
