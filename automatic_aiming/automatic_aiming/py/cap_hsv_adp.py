import cv2
import numpy as np

# 全局变量用于鼠标框选
selecting = False
x1, y1 = -1, -1
current_adjusted_frame = None  # 存储当前调整后帧，用于框选

def create_trackbars():
    """创建所有参数调节滑动条"""
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 320, 400)  # 增加高度容纳更多参数
    
    # HSV阈值滑动条（扩大初始范围，适应更多场景）
    cv2.createTrackbar("H_min", "Trackbars", 0, 179, lambda x: x)
    cv2.createTrackbar("S_min", "Trackbars", 40, 255, lambda x: x)
    cv2.createTrackbar("V_min", "Trackbars", 40, 255, lambda x: x)
    cv2.createTrackbar("H_max", "Trackbars", 179, 179, lambda x: x)
    cv2.createTrackbar("S_max", "Trackbars", 255, 255, lambda x: x)
    cv2.createTrackbar("V_max", "Trackbars", 255, 255, lambda x: x)
    
    # 灰度阈值滑动条
    cv2.createTrackbar("Gray_min", "Trackbars", 0, 255, lambda x: x)
    cv2.createTrackbar("Gray_max", "Trackbars", 255, 255, lambda x: x)
    
    # 形态学操作参数
    cv2.createTrackbar("Erode", "Trackbars", 1, 5, lambda x: x)
    cv2.createTrackbar("Dilate", "Trackbars", 1, 5, lambda x: x)
    
    # 面积过滤参数（适配高分辨率扩大范围）
    cv2.createTrackbar("Min Area", "Trackbars", 500, 50000, lambda x: x)  # 高分辨率小面积阈值提高
    cv2.createTrackbar("Max Area", "Trackbars", 20000, 200000, lambda x: x)
    
    # 图像增强参数
    cv2.createTrackbar("Brightness", "Trackbars", 50, 100, lambda x: x)
    cv2.createTrackbar("Contrast", "Trackbars", 100, 300, lambda x: x)
    cv2.createTrackbar("Saturation", "Trackbars", 100, 300, lambda x: x)
    
    # 相机硬件参数
    cv2.createTrackbar("Exposure", "Trackbars", 50, 100, lambda x: x)
    
    # 显示缩放参数（高分辨率专用）
    cv2.createTrackbar("Display Scale", "Trackbars", 50, 100, lambda x: x)  # 50%~100%缩放
    
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
    
    # 形态学参数
    erode = cv2.getTrackbarPos("Erode", "Trackbars")
    dilate = cv2.getTrackbarPos("Dilate", "Trackbars")
    
    # 面积过滤
    min_area = cv2.getTrackbarPos("Min Area", "Trackbars")
    max_area = cv2.getTrackbarPos("Max Area", "Trackbars")
    
    # 图像增强参数
    brightness = cv2.getTrackbarPos("Brightness", "Trackbars") - 50
    contrast = cv2.getTrackbarPos("Contrast", "Trackbars") / 100.0
    saturation = cv2.getTrackbarPos("Saturation", "Trackbars") / 100.0
    
    # 相机硬件参数
    exposure = cv2.getTrackbarPos("Exposure", "Trackbars")
    
    # 显示缩放（高分辨率专用）
    display_scale = cv2.getTrackbarPos("Display Scale", "Trackbars") / 100.0  # 50%~100%
    
    # 模式选择
    mode = cv2.getTrackbarPos("Mode", "Trackbars")
    
    return (h_min, s_min, v_min), (h_max, s_max, v_max), \
           gray_min, gray_max, erode, dilate, \
           min_area, max_area, brightness, contrast, saturation, exposure, display_scale, mode

def set_trackbar_values(h_min, s_min, v_min, h_max, s_max, v_max):
    """设置HSV滑动条值（用于框选后自动更新）"""
    cv2.setTrackbarPos("H_min", "Trackbars", h_min)
    cv2.setTrackbarPos("S_min", "Trackbars", s_min)
    cv2.setTrackbarPos("V_min", "Trackbars", v_min)
    cv2.setTrackbarPos("H_max", "Trackbars", h_max)
    cv2.setTrackbarPos("S_max", "Trackbars", s_max)
    cv2.setTrackbarPos("V_max", "Trackbars", v_max)

def mouse_callback(event, x, y, flags, param):
    """鼠标回调函数，在Adjusted窗口框选目标（适配缩放后坐标）"""
    global selecting, x1, y1, current_adjusted_frame, display_scale
    
    # 从参数中获取当前显示缩放比例（实际坐标 = 鼠标坐标 / 缩放比例）
    display_scale = param
    
    if event == cv2.EVENT_LBUTTONDOWN:
        selecting = True
        # 转换为原始图像坐标
        x1 = int(x / display_scale)
        y1 = int(y / display_scale)
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if selecting and current_adjusted_frame is not None:
            # 实时绘制选框（在缩放后的图像上）
            temp = current_adjusted_frame.copy()
            # 转换鼠标坐标到原始图像坐标
            x_raw = int(x / display_scale)
            y_raw = int(y / display_scale)
            cv2.rectangle(temp, (x1, y1), (x_raw, y_raw), (0, 255, 0), 2)
            # 缩放后显示
            scaled_temp = cv2.resize(temp, None, fx=display_scale, fy=display_scale)
            cv2.imshow("Adjusted", scaled_temp)
    
    elif event == cv2.EVENT_LBUTTONUP:
        selecting = False
        if current_adjusted_frame is None:
            return
            
        # 转换为原始图像坐标
        x2 = int(x / display_scale)
        y2 = int(y / display_scale)
        # 修正坐标顺序
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        # 过滤过小框选（高分辨率下最小框选扩大）
        if (x2 - x1) < 10 or (y2 - y1) < 10:
            print("选框太小（至少10x10像素），请重新选择")
            # 恢复显示
            scaled = cv2.resize(current_adjusted_frame, None, fx=display_scale, fy=display_scale)
            cv2.imshow("Adjusted", scaled)
            return
            
        # 提取ROI计算阈值
        roi = current_adjusted_frame[y1:y2, x1:x2]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # 计算ROI的HSV极值（高分辨率下增加容错性）
        h_min, s_min, v_min = np.percentile(hsv_roi, 10, axis=(0, 1)).astype(int)
        h_max, s_max, v_max = np.percentile(hsv_roi, 90, axis=(0, 1)).astype(int)
        
        # 扩展阈值范围
        h_min = max(0, h_min - 10)
        s_min = max(0, s_min - 30)
        v_min = max(0, v_min - 30)
        h_max = min(179, h_max + 10)
        s_max = min(255, s_max + 30)
        v_max = min(255, v_max + 30)
        
        # 更新滑动条并提示
        set_trackbar_values(h_min, s_min, v_min, h_max, s_max, v_max)
        print(f"自动阈值更新: H[{h_min}-{h_max}], S[{s_min}-{s_max}], V[{v_min}-{v_max}]")
        # 恢复显示
        scaled = cv2.resize(current_adjusted_frame, None, fx=display_scale, fy=display_scale)
        cv2.imshow("Adjusted", scaled)

def adjust_image(frame, brightness=0, contrast=1.0, saturation=1.0):
    """调整图像亮度、对比度和饱和度（优化高分辨率处理速度）"""
    # 先调整对比度和亮度
    adjusted = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
    
    # 单独调整饱和度（仅在需要时执行，节省计算）
    if saturation != 1.0:
        hsv = cv2.cvtColor(adjusted, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255).astype(np.uint8)
        adjusted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return adjusted

def track_color(frame, lower_hsv, upper_hsv, erode, dilate, min_area, max_area):
    """HSV颜色追踪（优化高分辨率下的轮廓处理效率）"""
    result = frame.copy()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    
    # 形态学操作（动态核大小，高分辨率下适当增大）
    kernel_size = 5 if max(erode, dilate) > 2 else 3
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    if erode > 0:
        mask = cv2.erode(mask, kernel, iterations=erode)
    if dilate > 0:
        mask = cv2.dilate(mask, kernel, iterations=dilate)
    
    # 轮廓提取与过滤（高分辨率下使用近似算法加速）
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    blobs = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            # 计算中心坐标
            perimeter = cv2.arcLength(cnt, True)
            if perimeter < 50:  # 高分辨率下过滤阈值提高
                continue
            M = cv2.moments(cnt)
            if M["m00"] < 1e-6:
                continue
            cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
            blobs.append({"center": (cx, cy), "area": area})
            
            # 绘制标记（适配高分辨率）
            cv2.rectangle(result, 
                         (cx-20, cy-20), (cx+20, cy+20),  # 扩大标记框
                         (0, 255, 0), 2)
            cv2.putText(result, f"({cx},{cy})", (cx+25, cy-25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return result, mask, hsv, blobs

def track_gray(frame, gray_min, gray_max, erode, dilate, min_area, max_area):
    """灰度追踪（与HSV模式保持一致的交互体验）"""
    result = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(gray, gray_min, gray_max)
    
    # 形态学操作（与HSV模式统一参数）
    kernel_size = 5 if max(erode, dilate) > 2 else 3
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    if erode > 0:
        mask = cv2.erode(mask, kernel, iterations=erode)
    if dilate > 0:
        mask = cv2.dilate(mask, kernel, iterations=dilate)
    
    # 轮廓提取与过滤
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    blobs = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            perimeter = cv2.arcLength(cnt, True)
            if perimeter < 50:
                continue
            M = cv2.moments(cnt)
            if M["m00"] < 1e-6:
                continue
            cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
            blobs.append({"center": (cx, cy), "area": area})
            
            # 绘制标记
            cv2.rectangle(result, 
                         (cx-20, cy-20), (cx+20, cy+20),
                         (0, 0, 255), 2)
            cv2.putText(result, f"({cx},{cy})", (cx+25, cy-25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return result, mask, gray, blobs

def main():
    global current_adjusted_frame
    
    # 初始化摄像头（启用1920x1080分辨率）
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头，请检查设备连接")
        return
    
    # 关键设置：启用MJPG格式以支持高分辨率
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)
    time.sleep(1)  # 等待设置生效
    
    # 验证分辨率
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"摄像头已打开，分辨率: {width}x{height}")
    if (width, height) != (1920, 1080):
        print("警告：摄像头不支持1920x1080，将使用可用最高分辨率")
    
    # 手动曝光设置
    try:
        for auto_exp_val in [0.25, 0, -1]:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto_exp_val)
        print("已尝试开启手动曝光调节")
    except:
        print("摄像头不支持手动曝光调节，将使用自动曝光")
    
    # 创建窗口（高分辨率专用，支持缩放显示）
    cv2.namedWindow("Original")
    cv2.namedWindow("Adjusted")
    cv2.namedWindow("Processed")
    cv2.namedWindow("Mask")
    cv2.namedWindow("Result")
    
    # 创建滑动条
    create_trackbars()
    
    # 初始帧显示
    ret, frame = cap.read()
    if ret:
        current_adjusted_frame = adjust_image(frame)
        # 初始缩放50%显示
        scaled = cv2.resize(current_adjusted_frame, None, fx=0.5, fy=0.5)
        cv2.imshow("Original", scaled)
        cv2.imshow("Adjusted", scaled)
        cv2.imshow("Processed", scaled)
        cv2.imshow("Mask", cv2.resize(np.zeros_like(frame), None, fx=0.5, fy=0.5))
        cv2.imshow("Result", scaled)
        cv2.waitKey(100)
    
    # 设置鼠标回调（绑定缩放参数）
    cv2.setMouseCallback("Adjusted", mouse_callback, param=0.5)  # 初始缩放50%
    
    print("\n程序启动成功！功能说明：")
    print("1. 在'Adjusted'窗口用鼠标框选目标（按住左键拖动），自动生成HSV阈值")
    print("2. 滑动条调参建议：")
    print("   - 先调Brightness/Contrast让目标清晰可见")
    print("   - 再用框选功能自动生成HSV阈值，必要时微调H_min/H_max")
    print("   - Erode/Dilate用于去除噪声，数值不宜过大（1-3为宜）")
    print("   - Min Area过滤小噪点，Max Area排除大干扰物")
    print("   - Display Scale调节显示比例（高分辨率推荐50%）")
    print("3. 按'q'键退出程序")
    
    try:
        while True:
            # 读取当前帧
            ret, frame = cap.read()
            if not ret:
                print("无法获取图像帧，程序退出")
                break
            
            # 获取滑动条参数
            (lower_hsv, upper_hsv, gray_min, gray_max, 
             erode, dilate, min_area, max_area, 
             brightness, contrast, saturation, exposure, display_scale, mode) = get_trackbar_values()
            
            # 设置相机曝光（如果支持）
            try:
                exposure_val = (exposure / 100.0) * 2 - 1  # 映射到[-1,1]
                cap.set(cv2.CAP_PROP_EXPOSURE, exposure_val)
            except:
                pass  # 不支持则忽略
            
            # 调整图像（亮度、对比度、饱和度）
            current_adjusted_frame = adjust_image(frame, brightness, contrast, saturation)
            
            # 根据模式选择追踪方式
            if mode == 0:  # HSV模式
                result_frame, mask, processed, blobs = track_color(
                    current_adjusted_frame, lower_hsv, upper_hsv, erode, dilate, min_area, max_area
                )
                mode_text = "HSV Mode"
            else:  # 灰度模式
                result_frame, mask, processed, blobs = track_gray(
                    current_adjusted_frame, gray_min, gray_max, erode, dilate, min_area, max_area
                )
                mode_text = "Gray Mode"
            
            # 显示追踪信息
            cv2.putText(result_frame, f"Mode: {mode_text}", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            cv2.putText(result_frame, f"Targets: {len(blobs)}", (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            
            # 高分辨率显示优化：按比例缩放所有窗口
            scaled_original = cv2.resize(frame, None, fx=display_scale, fy=display_scale)
            scaled_adjusted = cv2.resize(current_adjusted_frame, None, fx=display_scale, fy=display_scale)
            scaled_processed = cv2.resize(processed, None, fx=display_scale, fy=display_scale)
            scaled_mask = cv2.resize(mask, None, fx=display_scale, fy=display_scale)
            scaled_result = cv2.resize(result_frame, None, fx=display_scale, fy=display_scale)
            
            # 更新显示（非框选状态下）
            if not selecting:
                cv2.imshow("Adjusted", scaled_adjusted)
            cv2.imshow("Original", scaled_original)
            cv2.imshow("Processed", scaled_processed)
            cv2.imshow("Mask", scaled_mask)
            cv2.imshow("Result", scaled_result)
            
            # 更新鼠标回调的缩放参数
            cv2.setMouseCallback("Adjusted", mouse_callback, param=display_scale)
            
            # 按'q'键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except Exception as e:
        print(f"发生错误: {e}")
    
    finally:
        # 释放资源并关闭窗口
        cap.release()
        cv2.destroyAllWindows()
        print("程序已安全退出")

if __name__ == "__main__":
    import time  # 补充time模块引用
    main()