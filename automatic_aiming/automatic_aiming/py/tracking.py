import cv2
import numpy as np

# 全局变量用于鼠标框选
selecting = False
x1, y1 = -1, -1
current_adjusted_frame = None  # 存储当前调整后帧，用于框选

def create_trackbars():
    """创建所有参数调节滑动条"""
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 640, 500)  # 窗口尺寸减小，移除Dilate/Erode
    
    # HSV阈值滑动条
    cv2.createTrackbar("H_min", "Trackbars", 0, 179, lambda x: x)
    cv2.createTrackbar("S_min", "Trackbars", 40, 255, lambda x: x)
    cv2.createTrackbar("V_min", "Trackbars", 40, 255, lambda x: x)
    cv2.createTrackbar("H_max", "Trackbars", 179, 179, lambda x: x)
    cv2.createTrackbar("S_max", "Trackbars", 255, 255, lambda x: x)
    cv2.createTrackbar("V_max", "Trackbars", 255, 255, lambda x: x)
    
    # 灰度阈值滑动条
    cv2.createTrackbar("Gray_min", "Trackbars", 0, 255, lambda x: x)
    cv2.createTrackbar("Gray_max", "Trackbars", 255, 255, lambda x: x)
    
    # 面积过滤参数
    cv2.createTrackbar("Min Area", "Trackbars", 200, 10000, lambda x: x)
    cv2.createTrackbar("Max Area", "Trackbars", 5000, 50000, lambda x: x)
    
    # 图像增强参数
    cv2.createTrackbar("Brightness", "Trackbars", 50, 100, lambda x: x)
    cv2.createTrackbar("Contrast", "Trackbars", 100, 300, lambda x: x)
    cv2.createTrackbar("Saturation", "Trackbars", 100, 300, lambda x: x)
    
    # 相机硬件参数
    cv2.createTrackbar("Exposure", "Trackbars", 50, 100, lambda x: x)
    
    # 模式选择
    cv2.createTrackbar("Mode", "Trackbars", 0, 1, lambda x: x)  # 0: HSV模式, 1: 灰度模式

def get_trackbar_values():
    """获取所有滑动条当前值"""
    # HSV阈值
    h_min = cv2.getTrackbarPos("H_min", "Trackbars")
    s_min = cv2.getTrackbarPos("S_min", "Trackbars")
    v_min = cv2.getTrackbarPos("V_min", "Trackbars")
    h_max = cv2.getTrackbarPos("H_max", "Trackbars")
    s_max = cv2.getTrackbarPos("S_max", "Trackbars")
    v_max = cv2.getTrackbarPos("V_max", "Trackbars")
    
    # 灰度阈值
    gray_min = cv2.getTrackbarPos("Gray_min", "Trackbars")
    gray_max = cv2.getTrackbarPos("Gray_max", "Trackbars")
    
    # 面积过滤
    min_area = cv2.getTrackbarPos("Min Area", "Trackbars")
    max_area = cv2.getTrackbarPos("Max Area", "Trackbars")
    
    # 图像增强参数
    brightness = cv2.getTrackbarPos("Brightness", "Trackbars") - 50  # 范围：-50~50
    contrast = cv2.getTrackbarPos("Contrast", "Trackbars") / 100.0    # 范围：1.0~3.0
    saturation = cv2.getTrackbarPos("Saturation", "Trackbars") / 100.0 # 范围：1.0~3.0
    
    # 相机硬件参数
    exposure = cv2.getTrackbarPos("Exposure", "Trackbars")
    
    # 模式选择
    mode = cv2.getTrackbarPos("Mode", "Trackbars")
    
    return (h_min, s_min, v_min), (h_max, s_max, v_max), \
           gray_min, gray_max, min_area, max_area, \
           brightness, contrast, saturation, exposure, mode

def set_trackbar_values(h_min, s_min, v_min, h_max, s_max, v_max):
    """设置HSV滑动条值（用于框选后自动更新）"""
    cv2.setTrackbarPos("H_min", "Trackbars", h_min)
    cv2.setTrackbarPos("S_min", "Trackbars", s_min)
    cv2.setTrackbarPos("V_min", "Trackbars", v_min)
    cv2.setTrackbarPos("H_max", "Trackbars", h_max)
    cv2.setTrackbarPos("S_max", "Trackbars", s_max)
    cv2.setTrackbarPos("V_max", "Trackbars", v_max)

def mouse_callback(event, x, y, flags, param):
    """鼠标回调函数，在Adjusted窗口框选目标"""
    global selecting, x1, y1, current_adjusted_frame
    
    if event == cv2.EVENT_LBUTTONDOWN:
        selecting = True
        x1, y1 = x, y
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if selecting and current_adjusted_frame is not None:
            temp = current_adjusted_frame.copy()
            cv2.rectangle(temp, (x1, y1), (x, y), (0, 255, 0), 2)
            cv2.imshow("Adjusted", temp)
    
    elif event == cv2.EVENT_LBUTTONUP:
        selecting = False
        if current_adjusted_frame is None:
            return
            
        x1, x2 = min(x1, x), max(x1, x)
        y1, y2 = min(y1, y), max(y1, y)
        
        if (x2 - x1) < 5 or (y2 - y1) < 5:
            print("选框太小，请重新选择")
            cv2.imshow("Adjusted", current_adjusted_frame)
            return
            
        roi = current_adjusted_frame[y1:y2, x1:x2]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        h_min, s_min, v_min = np.percentile(hsv_roi, 5, axis=(0, 1)).astype(int)
        h_max, s_max, v_max = np.percentile(hsv_roi, 95, axis=(0, 1)).astype(int)
        
        h_min = max(0, h_min - 8)
        s_min = max(0, s_min - 25)
        v_min = max(0, v_min - 25)
        h_max = min(179, h_max + 8)
        s_max = min(255, s_max + 25)
        v_max = min(255, v_max + 25)
        
        set_trackbar_values(h_min, s_min, v_min, h_max, s_max, v_max)
        print(f"自动阈值更新: H[{h_min}-{h_max}], S[{s_min}-{s_max}], V[{v_min}-{v_max}]")
        cv2.imshow("Adjusted", current_adjusted_frame)

def adjust_image(frame, brightness=0, contrast=1.0, saturation=1.0):
    """调整图像亮度、对比度和饱和度"""
    adjusted = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
    
    if saturation != 1.0:
        hsv = cv2.cvtColor(adjusted, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255).astype(np.uint8)
        adjusted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return adjusted

def apply_morphological_filter(mask):
    """应用形态学滤波（双边滤波+闭运算）"""
    # 双边滤波保留边缘同时平滑区域
    filtered = cv2.bilateralFilter(mask, 9, 10, 10)
    
    # 闭运算（先膨胀后腐蚀）填充小孔洞
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    return closed

def track_color(frame, lower_hsv, upper_hsv, min_area, max_area):
    """HSV颜色追踪（使用固定形态学滤波）"""
    result = frame.copy()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    
    # 应用固定的形态学滤波
    processed_mask = apply_morphological_filter(mask)
    
    contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            perimeter = cv2.arcLength(cnt, True)
            if perimeter < 30:
                continue
            M = cv2.moments(cnt)
            if M["m00"] < 1e-6:
                continue
            cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
            blobs.append({"center": (cx, cy), "area": area})
            
            cv2.rectangle(result, (cx-10, cy-10), (cx+10, cy+10), (0, 255, 0), 2)
            cv2.putText(result, f"({cx},{cy})", (cx+15, cy-15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return result, processed_mask, hsv, blobs

def track_gray(frame, gray_min, gray_max, min_area, max_area):
    """灰度追踪（使用固定形态学滤波）"""
    result = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(gray, gray_min, gray_max)
    
    # 应用相同的形态学滤波
    processed_mask = apply_morphological_filter(mask)
    
    contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            perimeter = cv2.arcLength(cnt, True)
            if perimeter < 30:
                continue
            M = cv2.moments(cnt)
            if M["m00"] < 1e-6:
                continue
            cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
            blobs.append({"center": (cx, cy), "area": area})
            
            cv2.rectangle(result, (cx-10, cy-10), (cx+10, cy+10), (0, 0, 255), 2)
            cv2.putText(result, f"({cx},{cy})", (cx+15, cy-15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return result, processed_mask, gray, blobs

def main():
    global current_adjusted_frame
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    try:
        for auto_exp_val in [0.25, 0, -1]:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto_exp_val)
        print("已尝试开启手动曝光调节")
    except:
        print("摄像头不支持手动曝光调节")
    
    cv2.namedWindow("Original")
    cv2.namedWindow("Adjusted")
    cv2.namedWindow("Processed")
    cv2.namedWindow("Mask")
    cv2.namedWindow("Result")
    
    create_trackbars()
    
    ret, frame = cap.read()
    if ret:
        current_adjusted_frame = adjust_image(frame)
        cv2.imshow("Original", frame)
        cv2.imshow("Adjusted", current_adjusted_frame)
        cv2.imshow("Processed", frame)
        cv2.imshow("Mask", np.zeros_like(frame))
        cv2.imshow("Result", frame)
        cv2.waitKey(100)
    
    cv2.setMouseCallback("Adjusted", mouse_callback)
    
    print("\n程序启动成功！功能说明：")
    print("1. 在'Adjusted'窗口用鼠标框选目标，自动生成HSV阈值")
    print("2. 自动应用形态学滤波（双边滤波+闭运算）")
    print("3. 调参建议：")
    print("   - 先调Brightness/Contrast让目标清晰")
    print("   - 框选目标后微调HSV阈值")
    print("   - 调整Min/Max Area过滤噪点和大干扰物")
    print("4. 按'q'键退出程序")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法获取图像")
                break
            
            (lower_hsv, upper_hsv, gray_min, gray_max, 
             min_area, max_area, brightness, contrast, 
             saturation, exposure, mode) = get_trackbar_values()
            
            try:
                exposure_val = (exposure / 100.0) * 2 - 1
                cap.set(cv2.CAP_PROP_EXPOSURE, exposure_val)
            except:
                pass
            
            current_adjusted_frame = adjust_image(frame, brightness, contrast, saturation)
            
            if not selecting:
                cv2.imshow("Adjusted", current_adjusted_frame)
            
            if mode == 0:
                result_frame, mask, processed, blobs = track_color(
                    current_adjusted_frame, lower_hsv, upper_hsv, min_area, max_area
                )
                mode_text = "HSV Mode"
            else:
                result_frame, mask, processed, blobs = track_gray(
                    current_adjusted_frame, gray_min, gray_max, min_area, max_area
                )
                mode_text = "Gray Mode"
            
            cv2.putText(result_frame, f"Mode: {mode_text}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(result_frame, f"Targets: {len(blobs)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Original", frame)
            cv2.imshow("Processed", processed)
            cv2.imshow("Mask", mask)
            cv2.imshow("Result", result_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except Exception as e:
        print(f"发生错误: {e}")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("程序已安全退出")

if __name__ == "__main__":
    main()