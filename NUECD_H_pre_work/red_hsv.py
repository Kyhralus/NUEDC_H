import cv2
import numpy as np

# 全局变量用于鼠标框选
selecting = False
x1, y1 = -1, -1
current_adjusted_frame = None  # 存储当前调整后帧，用于框选

def create_trackbars():
    """创建所有参数调节滑动条，预设红色初始阈值"""
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 320, 240)
    
    # HSV阈值滑动条（针对红色优化初始范围）
    # 红色在HSV中跨0度边界，这里设置两个常用范围的初始值
    cv2.createTrackbar("H_min", "Trackbars", 0, 179, lambda x: x)
    cv2.createTrackbar("S_min", "Trackbars", 43, 255, lambda x: x)  # 红色通常需要较高饱和度
    cv2.createTrackbar("V_min", "Trackbars", 46, 255, lambda x: x)  # 红色亮度阈值
    cv2.createTrackbar("H_max", "Trackbars", 10, 179, lambda x: x)  # 低范围红色
    cv2.createTrackbar("S_max", "Trackbars", 255, 255, lambda x: x)
    cv2.createTrackbar("V_max", "Trackbars", 255, 255, lambda x: x)
    
    # 第二组HSV阈值（用于高范围红色，170-179）
    cv2.createTrackbar("H2_min", "Trackbars", 170, 179, lambda x: x)
    cv2.createTrackbar("H2_max", "Trackbars", 179, 179, lambda x: x)
    
    # 形态学操作参数
    cv2.createTrackbar("Erode", "Trackbars", 1, 5, lambda x: x)
    cv2.createTrackbar("Dilate", "Trackbars", 1, 5, lambda x: x)
    
    # 面积过滤参数
    cv2.createTrackbar("Min Area", "Trackbars", 200, 10000, lambda x: x)
    cv2.createTrackbar("Max Area", "Trackbars", 5000, 50000, lambda x: x)
    
    # 图像增强参数
    cv2.createTrackbar("Brightness", "Trackbars", 50, 100, lambda x: x)  # 0-100 → -50~50
    cv2.createTrackbar("Contrast", "Trackbars", 100, 300, lambda x: x)   # 100-300 → 1.0~3.0
    cv2.createTrackbar("Saturation", "Trackbars", 150, 300, lambda x: x) # 增强饱和度更利于红色识别

def get_trackbar_values():
    """获取所有滑动条当前值，特别处理红色的两个HSV范围"""
    # 第一组HSV阈值（低范围红色）
    h_min = cv2.getTrackbarPos("H_min", "Trackbars")
    s_min = cv2.getTrackbarPos("S_min", "Trackbars")
    v_min = cv2.getTrackbarPos("V_min", "Trackbars")
    h_max = cv2.getTrackbarPos("H_max", "Trackbars")
    s_max = cv2.getTrackbarPos("S_max", "Trackbars")
    v_max = cv2.getTrackbarPos("V_max", "Trackbars")
    
    # 第二组HSV阈值（高范围红色）
    h2_min = cv2.getTrackbarPos("H2_min", "Trackbars")
    h2_max = cv2.getTrackbarPos("H2_max", "Trackbars")
    
    # 形态学参数
    erode = cv2.getTrackbarPos("Erode", "Trackbars")
    dilate = cv2.getTrackbarPos("Dilate", "Trackbars")
    
    # 面积过滤
    # 修正此处的窗口名称
    min_area = cv2.getTrackbarPos("Min Area", "Trackbars")
    max_area = cv2.getTrackbarPos("Max Area", "Trackbars")
    
    # 图像增强参数
    brightness = cv2.getTrackbarPos("Brightness", "Trackbars") - 50  # 范围：-50~50
    contrast = cv2.getTrackbarPos("Contrast", "Trackbars") / 100.0    # 范围：1.0~3.0
    saturation = cv2.getTrackbarPos("Saturation", "Trackbars") / 100.0 # 范围：1.0~3.0
    
    return (h_min, s_min, v_min), (h_max, s_max, v_max), \
           (h2_min, h2_max), erode, dilate, \
           min_area, max_area, brightness, contrast, saturation

def set_trackbar_values(h_min, s_min, v_min, h_max, s_max, v_max, h2_min, h2_max):
    """设置HSV滑动条值（用于框选后自动更新）"""
    cv2.setTrackbarPos("H_min", "Trackbars", h_min)
    cv2.setTrackbarPos("S_min", "Trackbars", s_min)
    cv2.setTrackbarPos("V_min", "Trackbars", v_min)
    cv2.setTrackbarPos("H_max", "Trackbars", h_max)
    cv2.setTrackbarPos("S_max", "Trackbars", s_max)
    cv2.setTrackbarPos("V_max", "Trackbars", v_max)
    cv2.setTrackbarPos("H2_min", "Trackbars", h2_min)
    cv2.setTrackbarPos("H2_max", "Trackbars", h2_max)

def mouse_callback(event, x, y, flags, param):
    """鼠标回调函数，在Adjusted窗口框选红色目标"""
    global selecting, x1, y1, current_adjusted_frame
    
    if event == cv2.EVENT_LBUTTONDOWN:
        selecting = True
        x1, y1 = x, y  # 记录框选起点
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if selecting and current_adjusted_frame is not None:
            # 实时绘制选框（绿色）
            temp = current_adjusted_frame.copy()
            cv2.rectangle(temp, (x1, y1), (x, y), (0, 255, 0), 2)
            cv2.imshow("Adjusted", temp)
    
    elif event == cv2.EVENT_LBUTTONUP:
        selecting = False
        if current_adjusted_frame is None:
            return
            
        x2, y2 = x, y  # 记录框选终点
        # 修正坐标顺序
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        # 过滤过小框选
        if (x2 - x1) < 5 or (y2 - y1) < 5:
            print("选框太小（至少5x5像素），请重新选择")
            cv2.imshow("Adjusted", current_adjusted_frame)
            return
            
        # 从调整后帧提取ROI计算红色阈值
        roi = current_adjusted_frame[y1:y2, x1:x2]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # 提取H通道并处理红色的特殊性（可能跨0度）
        h_values = hsv_roi[:, :, 0].flatten()
        s_values = hsv_roi[:, :, 1].flatten()
        v_values = hsv_roi[:, :, 2].flatten()
        
        # 过滤掉低饱和度的像素（不是纯红色）
        valid_mask = s_values > 50
        h_valid = h_values[valid_mask]
        s_valid = s_values[valid_mask]
        v_valid = v_values[valid_mask]
        
        if len(h_valid) < 10:  # 有效像素太少
            print("有效红色像素不足，请重新框选更纯的红色区域")
            return
        
        # 计算HSV百分位值（过滤异常值）
        h_p5, h_p95 = np.percentile(h_valid, [5, 95]).astype(int)
        s_min, s_max = np.percentile(s_valid, [5, 95]).astype(int)
        v_min, v_max = np.percentile(v_valid, [5, 95]).astype(int)
        
        # 处理红色的H通道跨0度问题
        h2_min, h2_max = 0, 0
        if h_p95 > 170:  # 包含高范围红色
            h_min1 = max(0, h_p5 - 5)
            h_max1 = 10  # 低范围上限
            h_min2 = 170  # 高范围下限
            h_max2 = min(179, h_p95 + 5)
            set_trackbar_values(h_min1, max(0, s_min-20), max(0, v_min-20),
                               h_max1, min(255, s_max+20), min(255, v_max+20),
                               h_min2, h_max2)
            print(f"自动阈值更新: 低范围H[{h_min1}-{h_max1}], 高范围H[{h_min2}-{h_max2}], S[{s_min}-{s_max}], V[{v_min}-{v_max}]")
        else:  # 仅低范围红色
            h_min = max(0, h_p5 - 5)
            h_max = min(179, h_p95 + 5)
            set_trackbar_values(h_min, max(0, s_min-20), max(0, v_min-20),
                               h_max, min(255, s_max+20), min(255, v_max+20),
                               170, 179)  # 保持默认高范围
            print(f"自动阈值更新: H[{h_min}-{h_max}], S[{s_min}-{s_max}], V[{v_min}-{v_max}]")
            
        cv2.imshow("Adjusted", current_adjusted_frame)

def adjust_image(frame, brightness=0, contrast=1.0, saturation=1.0):
    """调整图像亮度、对比度和饱和度（增强红色识别）"""
    # 先调整对比度和亮度
    adjusted = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
    
    # 增强饱和度（利于红色识别）
    if saturation != 1.0:
        hsv = cv2.cvtColor(adjusted, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255).astype(np.uint8)
        adjusted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return adjusted

def track_red(frame, lower1, upper1, lower2, upper2, erode, dilate, min_area, max_area):
    """追踪红色色块（处理两个HSV范围）"""
    result = frame.copy()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 创建两个掩码并合并（覆盖红色的两个范围）
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)
    
    # 形态学操作去除噪声
    kernel = np.ones((3, 3), np.uint8)
    if erode > 0:
        mask = cv2.erode(mask, kernel, iterations=erode)
    if dilate > 0:
        mask = cv2.dilate(mask, kernel, iterations=dilate)
    
    # 轮廓提取与过滤
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            perimeter = cv2.arcLength(cnt, True)
            if perimeter < 30:  # 过滤小噪声
                continue
            M = cv2.moments(cnt)
            if M["m00"] < 1e-6:
                continue
            cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
            blobs.append({"center": (cx, cy), "area": area})
            
            # 绘制红色标记
            cv2.rectangle(result, (cx-10, cy-10), (cx+10, cy+10), (0, 0, 255), 2)
            cv2.putText(result, f"({cx},{cy})", (cx+15, cy-15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return result, mask, hsv, blobs

def main():
    global current_adjusted_frame
    
    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头，请检查设备连接")
        return
    
    # 设置摄像头参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # 创建窗口
    cv2.namedWindow("Original")
    cv2.namedWindow("Adjusted")
    cv2.namedWindow("Processed")
    cv2.namedWindow("Mask")
    cv2.namedWindow("Result")
    cv2.resizeWindow("Original", 320, 240)
    cv2.resizeWindow("Adjusted", 320, 240)
    cv2.resizeWindow("Processed", 320, 240)
    cv2.resizeWindow("Mask", 320, 240)
    cv2.resizeWindow("Result", 320, 240)

    # 创建滑动条（预设红色初始值）
    create_trackbars()
    
    # 初始帧显示
    ret, frame = cap.read()
    if ret:
        current_adjusted_frame = adjust_image(frame)
        cv2.imshow("Original", frame)
        cv2.imshow("Adjusted", current_adjusted_frame)
        cv2.imshow("Processed", frame)
        cv2.imshow("Mask", np.zeros_like(frame))
        cv2.imshow("Result", frame)
        cv2.waitKey(100)
    
    # 设置鼠标回调
    cv2.setMouseCallback("Adjusted", mouse_callback)
    
    print("\n红色色块追踪程序启动成功！")
    print("1. 在'Adjusted'窗口用鼠标框选红色目标，自动生成HSV阈值")
    print("2. 红色在HSV中分为两个范围，滑动条已预设常用值")
    print("3. 调整建议：")
    print("   - 提高Saturation增强红色饱和度区分")
    print("   - 适当提高S_min过滤低饱和度干扰")
    print("   - 若红色偏橙，可增大H_max；若偏紫，可降低H2_min")
    print("4. 按'q'键退出程序")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法获取图像帧，程序退出")
                break
            
            # 获取滑动条参数
            (lower1, upper1, (h2_min, h2_max), erode, dilate,
             min_area, max_area, brightness, contrast, saturation) = get_trackbar_values()
            
            # 构建第二个HSV范围（高范围红色）
            lower2 = (h2_min, lower1[1], lower1[2])
            upper2 = (h2_max, upper1[1], upper1[2])
            
            # 调整图像
            current_adjusted_frame = adjust_image(frame, brightness, contrast, saturation)
            
            # 显示调整后的图像
            if not selecting:
                cv2.imshow("Adjusted", current_adjusted_frame)
            
            # 追踪红色色块
            result_frame, mask, processed, blobs = track_red(
                current_adjusted_frame, lower1, upper1, lower2, upper2,
                erode, dilate, min_area, max_area
            )
            
            # 显示追踪信息
            cv2.putText(result_frame, f"Red Targets: {len(blobs)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(result_frame, f"H: [{lower1[0]}-{upper1[0]}, {h2_min}-{h2_max}]", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 显示所有窗口
            cv2.imshow("Original", frame)
            cv2.imshow("Processed", processed)
            cv2.imshow("Mask", mask)
            cv2.imshow("Result", result_frame)
            
            # 按'q'键退出
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
