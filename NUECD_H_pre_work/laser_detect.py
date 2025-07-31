import cv2
import numpy as np
import time

# 全局变量
selecting = False
x1, y1 = -1, -1
current_adjusted_frame = None  # 存储当前调整后帧，用于框选

def create_trackbars():
    """创建所有参数调节滑动条，预设蓝紫色激光初始阈值"""
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 320, 240)
    
    # HSV阈值滑动条（蓝紫色：H大约在120-160）
    cv2.createTrackbar("H_min", "Trackbars", 120, 179, lambda x: x)
    cv2.createTrackbar("S_min", "Trackbars", 40, 255, lambda x: x)
    cv2.createTrackbar("V_min", "Trackbars", 80, 255, lambda x: x)
    cv2.createTrackbar("H_max", "Trackbars", 150, 179, lambda x: x)
    cv2.createTrackbar("S_max", "Trackbars", 255, 255, lambda x: x)
    cv2.createTrackbar("V_max", "Trackbars", 255, 255, lambda x: x)
    
    # 第二组HSV阈值（可用于更偏紫色部分）
    cv2.createTrackbar("H2_min", "Trackbars", 150, 179, lambda x: x)
    cv2.createTrackbar("H2_max", "Trackbars", 170, 179, lambda x: x)
    
    # 形态学参数
    cv2.createTrackbar("Erode", "Trackbars", 1, 5, lambda x: x)
    cv2.createTrackbar("Dilate", "Trackbars", 1, 5, lambda x: x)
    
    # 面积过滤
    cv2.createTrackbar("Min Area", "Trackbars", 100, 10000, lambda x: x)
    cv2.createTrackbar("Max Area", "Trackbars", 5000, 50000, lambda x: x)
    
    # 图像增强
    cv2.createTrackbar("Brightness", "Trackbars", 50, 100, lambda x: x)
    cv2.createTrackbar("Contrast", "Trackbars", 120, 300, lambda x: x)
    cv2.createTrackbar("Saturation", "Trackbars", 150, 300, lambda x: x)

def get_trackbar_values():
    """获取所有滑动条参数"""
    h_min = cv2.getTrackbarPos("H_min", "Trackbars")
    s_min = cv2.getTrackbarPos("S_min", "Trackbars")
    v_min = cv2.getTrackbarPos("V_min", "Trackbars")
    h_max = cv2.getTrackbarPos("H_max", "Trackbars")
    s_max = cv2.getTrackbarPos("S_max", "Trackbars")
    v_max = cv2.getTrackbarPos("V_max", "Trackbars")
    
    h2_min = cv2.getTrackbarPos("H2_min", "Trackbars")
    h2_max = cv2.getTrackbarPos("H2_max", "Trackbars")
    
    erode = cv2.getTrackbarPos("Erode", "Trackbars")
    dilate = cv2.getTrackbarPos("Dilate", "Trackbars")
    min_area = cv2.getTrackbarPos("Min Area", "Trackbars")
    max_area = cv2.getTrackbarPos("Max Area", "Trackbars")
    
    brightness = cv2.getTrackbarPos("Brightness", "Trackbars") - 50
    contrast = cv2.getTrackbarPos("Contrast", "Trackbars") / 100.0
    saturation = cv2.getTrackbarPos("Saturation", "Trackbars") / 100.0
    
    return (h_min, s_min, v_min), (h_max, s_max, v_max), \
           (h2_min, h2_max), erode, dilate, \
           min_area, max_area, brightness, contrast, saturation

def set_trackbar_values(h_min, s_min, v_min, h_max, s_max, v_max, h2_min, h2_max):
    """设置HSV滑动条值"""
    cv2.setTrackbarPos("H_min", "Trackbars", h_min)
    cv2.setTrackbarPos("S_min", "Trackbars", s_min)
    cv2.setTrackbarPos("V_min", "Trackbars", v_min)
    cv2.setTrackbarPos("H_max", "Trackbars", h_max)
    cv2.setTrackbarPos("S_max", "Trackbars", s_max)
    cv2.setTrackbarPos("V_max", "Trackbars", v_max)
    cv2.setTrackbarPos("H2_min", "Trackbars", h2_min)
    cv2.setTrackbarPos("H2_max", "Trackbars", h2_max)

def mouse_callback(event, x, y, flags, param):
    """鼠标框选蓝紫色激光区域自动更新HSV阈值"""
    global selecting, x1, y1, current_adjusted_frame
    
    if event == cv2.EVENT_LBUTTONDOWN:
        selecting = True
        x1, y1 = x, y
        
    elif event == cv2.EVENT_MOUSEMOVE and selecting and current_adjusted_frame is not None:
        temp = current_adjusted_frame.copy()
        cv2.rectangle(temp, (x1, y1), (x, y), (0, 255, 0), 2)
        cv2.imshow("Adjusted", temp)
    
    elif event == cv2.EVENT_LBUTTONUP:
        selecting = False
        if current_adjusted_frame is None:
            return
        x2, y2 = x, y
        x1_, x2 = min(x1, x2), max(x1, x2)
        y1_, y2 = min(y1, y2), max(y1, y2)
        if (x2 - x1_) < 5 or (y2 - y1_) < 5:
            print("选框太小，请重新选择")
            return
        roi = current_adjusted_frame[y1_:y2, x1_:x2]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h_values = hsv_roi[:,:,0].flatten()
        s_values = hsv_roi[:,:,1].flatten()
        v_values = hsv_roi[:,:,2].flatten()
        valid_mask = s_values > 30
        h_valid = h_values[valid_mask]
        s_valid = s_values[valid_mask]
        v_valid = v_values[valid_mask]
        if len(h_valid) < 5:
            print("有效像素太少，请重新框选")
            return
        h_p5,h_p95 = np.percentile(h_valid,[5,95]).astype(int)
        s_min,s_max = np.percentile(s_valid,[5,95]).astype(int)
        v_min,v_max = np.percentile(v_valid,[5,95]).astype(int)
        set_trackbar_values(max(0,h_p5-5), max(0,s_min-20), max(0,v_min-20),
                            min(179,h_p95+5), min(255,s_max+20), min(255,v_max+20),
                            150,170)
        print(f"蓝紫色阈值更新: H[{h_p5}-{h_p95}], S[{s_min}-{s_max}], V[{v_min}-{v_max}]")
        cv2.imshow("Adjusted", current_adjusted_frame)

def adjust_image(frame, brightness=0, contrast=1.0, saturation=1.0):
    adjusted = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
    if saturation!=1.0:
        hsv=cv2.cvtColor(adjusted,cv2.COLOR_BGR2HSV)
        hsv[:,:,1]=np.clip(hsv[:,:,1]*saturation,0,255).astype(np.uint8)
        adjusted=cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    return adjusted

def track_purple(frame, lower1, upper1, lower2, upper2, erode, dilate, min_area, max_area):
    result=frame.copy()
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask1=cv2.inRange(hsv,lower1,upper1)
    mask2=cv2.inRange(hsv,lower2,upper2)
    mask=cv2.bitwise_or(mask1,mask2)
    kernel=np.ones((3,3),np.uint8)
    if erode>0:mask=cv2.erode(mask,kernel,iterations=erode)
    if dilate>0:mask=cv2.dilate(mask,kernel,iterations=dilate)
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    blobs=[]
    for cnt in contours:
        area=cv2.contourArea(cnt)
        if min_area<area<max_area:
            M=cv2.moments(cnt)
            if M["m00"]<1e-6:continue
            cx,cy=int(M["m10"]/M["m00"]),int(M["m01"]/M["m00"])
            blobs.append({"center":(cx,cy),"area":area})
            cv2.rectangle(result,(cx-10,cy-10),(cx+10,cy+10),(255,0,255),2)
            cv2.putText(result,f"({cx},{cy})",(cx+15,cy-15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
    return result,mask,hsv,blobs

def main():
    global current_adjusted_frame
    cap=cv2.VideoCapture(0)
    
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    # cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    # cap.set(cv2.CAP_PROP_FOCUS, 10)
    time.sleep(1)
    
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"摄像头已打开，实际分辨率: {width}x{height}")


    cv2.namedWindow("Original")
    cv2.namedWindow("Adjusted")
    cv2.namedWindow("Processed")
    cv2.namedWindow("Mask")
    cv2.namedWindow("Result")
    create_trackbars()
    ret,frame=cap.read()
    if ret:
        current_adjusted_frame=adjust_image(frame)
        cv2.imshow("Original",frame)
    cv2.setMouseCallback("Adjusted",mouse_callback)
    print("\n蓝紫色激光追踪程序启动")
    print("1. 在'Adjusted'窗口框选激光点自动提取HSV阈值")
    print("2. 调整滑条可优化检测效果")
    print("3. 按'q'退出")
    while True:
        ret,frame=cap.read()
        if not ret:break
        (lower1,upper1,(h2_min,h2_max),erode,dilate,min_area,max_area,brightness,contrast,saturation)=get_trackbar_values()
        lower2=(h2_min,lower1[1],lower1[2])
        upper2=(h2_max,upper1[1],upper1[2])
        current_adjusted_frame=adjust_image(frame,brightness,contrast,saturation)
        if not selecting:cv2.imshow("Adjusted",current_adjusted_frame)
        result_frame,mask,processed,blobs=track_purple(current_adjusted_frame,lower1,upper1,lower2,upper2,erode,dilate,min_area,max_area)
        cv2.putText(result_frame,f"Targets:{len(blobs)}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,255),2)
        cv2.imshow("Original",frame)
        cv2.imshow("Processed",processed)
        cv2.imshow("Mask",mask)
        cv2.imshow("Result",result_frame)
        if cv2.waitKey(1)&0xFF==ord('q'):break
    cap.release()
    cv2.destroyAllWindows()
    print("程序已退出")

if __name__=="__main__":
    main()
