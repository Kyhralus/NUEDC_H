import cv2
import numpy as np
import time

def create_trackbars():
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 320, 240)
    cv2.createTrackbar("Threshold", "Trackbars", 200, 255, lambda x: x)
    cv2.createTrackbar("Erode", "Trackbars", 1, 5, lambda x: x)
    cv2.createTrackbar("Dilate", "Trackbars", 1, 5, lambda x: x)
    cv2.createTrackbar("Min Area", "Trackbars", 20, 5000, lambda x: x)
    cv2.createTrackbar("Max Area", "Trackbars", 2000, 10000, lambda x: x)

def get_trackbar_values():
    th = cv2.getTrackbarPos("Threshold", "Trackbars")
    erode = cv2.getTrackbarPos("Erode", "Trackbars")
    dilate = cv2.getTrackbarPos("Dilate", "Trackbars")
    min_area = cv2.getTrackbarPos("Min Area", "Trackbars")
    max_area = cv2.getTrackbarPos("Max Area", "Trackbars")
    return th, erode, dilate, min_area, max_area

def track_bright_spot(gray, th, erode, dilate, min_area, max_area):
    result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    _, mask = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3,3), np.uint8)
    if erode>0: mask = cv2.erode(mask, kernel, iterations=erode)
    if dilate>0: mask = cv2.dilate(mask, kernel, iterations=dilate)
    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs=[]
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area<area<max_area:
            M = cv2.moments(cnt)
            if M["m00"]<1e-6: continue
            cx,cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
            blobs.append({"center":(cx,cy),"area":area})
            cv2.circle(result,(cx,cy),5,(0,255,0),-1)
            cv2.putText(result,f"({cx},{cy})",(cx+10,cy-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
    return result,mask,blobs

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    time.sleep(1)
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    create_trackbars()
    print("\n灰度激光追踪程序启动")
    print("1. 调整阈值使激光点被检测出")
    print("2. 按'q'退出")
    while True:
        ret,frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        th,erode,dilate,min_area,max_area = get_trackbar_values()
        result,mask,blobs = track_bright_spot(gray,th,erode,dilate,min_area,max_area)
        cv2.putText(result,f"Targets:{len(blobs)}",(10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
        cv2.imshow("Gray",gray)
        cv2.imshow("Mask",mask)
        cv2.imshow("Result",result)
        if cv2.waitKey(1)&0xFF==ord('q'):break
    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
