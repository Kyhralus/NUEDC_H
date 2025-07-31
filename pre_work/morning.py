# !/usr/bin/env python3
#  -*- coding: utf-8 -*-

import cv2
import numpy as np
import time
from typing import Optional, Tuple
from collections import deque

class EnhancedCircleDetector:
    def __init__(self):
        # 稳定性参数
        self.history_size = 7
        self.center_history = deque(maxlen=self.history_size)
        self.quality_history = deque(maxlen=self.history_size)
        
        # 检测参数
        self.min_area = 50
        self.max_area = 10000
        self.min_circularity = 0.65
        self.duplicate_dist_thresh = 10
        self.duplicate_radius_thresh = 10
        
        # 质量评估参数
        self.min_quality_score = 0.5
        self.quality_weight_circularity = 0.4
        self.quality_weight_area_ratio = 0.3
        self.quality_weight_stability = 0.3

def detect_nested_rects(img_raw):
    """检测嵌套矩形"""
    start_time = time.time()
    
    if img_raw is None:
        return None, [], None, None, None

    # 步骤1: 灰度转换
    img_gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Step1: gray', img_gray)
    
    # 步骤2: 双边滤波
    img_bilateral = cv2.bilateralFilter(img_gray, 9, 10, 10)
    cv2.imshow('Step2: bi', img_bilateral)
    
    # 步骤3: 形态学闭运算
    kernel = np.ones((3, 3), np.uint8)
    img_open = cv2.morphologyEx(img_bilateral, cv2.MORPH_OPEN, kernel)
    img_close = cv2.morphologyEx(img_open, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('Step3: close', img_close)
    
    # 步骤4: 边缘检测
    edged = cv2.Canny(img_close, 40, 150)
    cv2.imshow('Step4: canny', edged)
    
    # 找出所有轮廓，不包括层级信息
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # 筛选轮廓，保留面积大于min_area且小于max_area的轮廓
    # min_area = 100
    # contours = list(filter(lambda cnt: min_area < cv2.contourArea(cnt), contours))


    # 步骤5: 筛选轮廓，拟合矩形，对符合矩形的轮廓进行筛选。记作candidate_rects:(nX7):[[左上点x,左上点y,宽w,高h,四个角点corners,面积area,中心点center]]
    candidate_rects = []
    img_contours = img_raw.copy()
    img_candidates = img_raw.copy()

    
    for i, contour in enumerate(contours):
        epsilon = 0.04 * cv2.arcLength(contour, True) # 轮廓周长（闭合）
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # debug 打印轮廓信息
        # 计算轮廓基本属性
        area = cv2.contourArea(contour)  # 轮廓面积
        perimeter = cv2.arcLength(contour, True)  
        vertices = len(approx)  # 近似多边形的顶点数
        is_convex = cv2.isContourConvex(approx)  # 是否为凸形
        
        # 打印当前轮廓信息（筛选前）
        print(f"轮廓索引: {i}")
        print(f"  面积: {area:.2f} 像素")
        print(f"  周长: {perimeter:.2f} 像素")
        print(f"  近似后顶点数: {vertices}")
        print(f"  是否为凸形: {'是' if is_convex else '否'}")
        print("  ------------------------")

        # 绘制轮廓
         # 绘制原始轮廓（蓝色）
        cv2.drawContours(img_contours, [contour], -1, (255, 0, 0), 2)
        
        # 绘制近似多边形（红色，可选）
        cv2.drawContours(img_contours, [approx], -1, (0, 0, 255), 2)

        # 绘制近似多边形的角点（绿色圆点 + 序号）
        for idx, corner in enumerate(approx):
            # corner是[[x,y]]格式，需要提取坐标
            x, y = corner[0]
            # 绘制角点（绿色实心圆）
            cv2.circle(img_contours, (x, y), 5, (0, 255, 0), -1)
            # 标注角点序号
            # 标注角点序号
            cv2.putText(img_contours, f"{idx}", (x+5, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        print("找到的轮廓信息")
        if len(approx) == 4 and cv2.contourArea(approx) > 1000:
            x, y, w, h = cv2.boundingRect(approx)
            area = cv2.contourArea(approx)
            center = (x + w // 2, y + h // 2)
            corners = approx.reshape(4, 2)
            candidate_rects.append([x, y, w, h, corners, area, center])
            
            # 在候选矩形图像上绘制矩形（黄色）
            cv2.drawContours(img_candidates, [corners.reshape(-1, 1, 2)], 0, (0, 255, 255), 2)
            cv2.circle(img_candidates, center, 3, (0, 255, 255), -1)
            cv2.putText(img_candidates, f"A:{area}", (center[0]-20, center[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    cv2.imshow('Step5: contors', img_contours)
    cv2.imshow('Step5: cand_rects', img_candidates)
    print(f"[DEBUG] 筛选出 {len(candidate_rects)} 个候选矩形")
    
    # 步骤6: 筛选嵌套矩形，遍历 candidate_rects, 将x,y相近的归为一组矩形，记作 nested_rects = [[外层矩形（x,y都大的矩形）信息],[内层矩形信息]]
    nested_rects = []
    center_threshold = 50  # 中心点距离阈值
    img_nested = img_raw.copy()
    
    for i, outer_rect in enumerate(candidate_rects):
        outer_x, outer_y, outer_w, outer_h, outer_corners, outer_area, outer_center = outer_rect
        for j, inner_rect in enumerate(candidate_rects):
            if i == j:
                continue
            inner_x, inner_y, inner_w, inner_h, inner_corners, inner_area, inner_center = inner_rect
            
            # 检查是否嵌套（外层矩形包含内层矩形）
            is_nested = (outer_x < inner_x and outer_y < inner_y and 
                        outer_x + outer_w > inner_x + inner_w and 
                        outer_y + outer_h > inner_y + inner_h)
            
            # 检查中心点是否相近
            center_distance = np.sqrt((outer_center[0] - inner_center[0])**2 + 
                                    (outer_center[1] - inner_center[1])**2)
            
            if is_nested and center_distance < center_threshold:
                nested_rects.append([outer_rect, inner_rect])
                
                # 在嵌套矩形图像上绘制（外层红色，内层蓝色）
                cv2.drawContours(img_nested, [outer_corners.reshape(-1, 1, 2)], 0, (0, 0, 255), 2)
                cv2.drawContours(img_nested, [inner_corners.reshape(-1, 1, 2)], 0, (255, 0, 0), 2)
                cv2.circle(img_nested, outer_center, 3, (0, 0, 255), -1)
                cv2.circle(img_nested, inner_center, 3, (255, 0, 0), -1)
                cv2.line(img_nested, outer_center, inner_center, (0, 255, 255), 1)
                cv2.putText(img_nested, f"D:{center_distance:.1f}", 
                           ((outer_center[0] + inner_center[0])//2, (outer_center[1] + inner_center[1])//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    cv2.imshow('Step6: nested_rects', img_nested)
    print(f"[DEBUG] 找到 {len(nested_rects)} 对嵌套矩形")
    
    # 步骤7: 计算 nested_rects 的面积比
    img_area_ratio = img_raw.copy()
    for i, nested_pair in enumerate(nested_rects):
        outer_rect, inner_rect = nested_pair
        area_ratio = inner_rect[5] / outer_rect[5]  # inner_area / outer_area
        print(f"[DEBUG] 嵌套矩形面积比: {area_ratio:.3f}")
        
        # 绘制所有嵌套矩形对及其面积比
        outer_corners, inner_corners = outer_rect[4], inner_rect[4]
        outer_center, inner_center = outer_rect[6], inner_rect[6]
        cv2.drawContours(img_area_ratio, [outer_corners.reshape(-1, 1, 2)], 0, (0, 255, 0), 2)
        cv2.drawContours(img_area_ratio, [inner_corners.reshape(-1, 1, 2)], 0, (255, 0, 0), 2)
        cv2.putText(img_area_ratio, f"Ratio:{area_ratio:.2f}", 
                   (inner_center[0]-30, inner_center[1]+20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow('Step7: area_ratio', img_area_ratio)
    
    # 步骤8: 筛选嵌套矩形，将外层矩形和内层矩形同时满足面积比的矩形作为最终嵌套矩形对border_rects = [[外层矩形信息],[内层矩形信息]]
    border_rects = []
    area_ratio_min = 0.3
    area_ratio_max = 0.9
    img_filtered = img_raw.copy()
    
    for nested_pair in nested_rects:
        outer_rect, inner_rect = nested_pair
        area_ratio = inner_rect[5] / outer_rect[5]  # inner_area / outer_area
        if area_ratio_min < area_ratio < area_ratio_max:
            border_rects.append([outer_rect, inner_rect])
            
            # 绘制符合条件的矩形对（外层绿色，内层蓝色）
            outer_corners, inner_corners = outer_rect[4], inner_rect[4]
            outer_center, inner_center = outer_rect[6], inner_rect[6]
            cv2.drawContours(img_filtered, [outer_corners.reshape(-1, 1, 2)], 0, (0, 255, 0), 3)
            cv2.drawContours(img_filtered, [inner_corners.reshape(-1, 1, 2)], 0, (255, 0, 0), 3)
            cv2.circle(img_filtered, outer_center, 5, (0, 255, 0), -1)
            cv2.circle(img_filtered, inner_center, 5, (255, 0, 0), -1)
            cv2.line(img_filtered, outer_center, inner_center, (0, 255, 255), 2)
            cv2.putText(img_filtered, f"PASS:{area_ratio:.2f}", 
                       (inner_center[0]-40, inner_center[1]-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.imshow('Step8: final_rect', img_filtered)
    print(f"[DEBUG] 筛选出 {len(border_rects)} 对符合面积比的嵌套矩形")
    
    # 绘制结果
    img_result = img_raw.copy()
    outer_rect = None
    inner_rect = None
    
    if border_rects:
        # 选择第一对作为最终结果
        final_outer, final_inner = border_rects[0]
        outer_rect = {
            'bbox': (final_outer[0], final_outer[1], final_outer[2], final_outer[3]),
            'corners': final_outer[4],
            'area': final_outer[5],
            'center': final_outer[6]
        }
        inner_rect = {
            'bbox': (final_inner[0], final_inner[1], final_inner[2], final_inner[3]),
            'corners': final_inner[4],
            'area': final_inner[5],
            'center': final_inner[6]
        }
        
        # 绘制外层矩形（绿色）
        cv2.drawContours(img_result, [final_outer[4].reshape(-1, 1, 2)], 0, (0, 255, 0), 2)
        # 绘制内层矩形（蓝色）
        cv2.drawContours(img_result, [final_inner[4].reshape(-1, 1, 2)], 0, (255, 0, 0), 2)
        # 绘制中心点
        cv2.circle(img_result, final_outer[6], 3, (0, 0, 255), -1)
        cv2.circle(img_result, final_inner[6], 3, (0, 0, 255), -1)
        # 连接中心点
        cv2.line(img_result, final_outer[6], final_inner[6], (0, 255, 255), 2)
        # 显示面积比
        area_ratio = final_inner[5] / final_outer[5]
        cv2.putText(img_result, f"{area_ratio:.2f}", final_inner[6], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    
    end_time = time.time()
    print(f"[TIMING] detect_nested_rects 耗时: {(end_time - start_time)*1000:.2f}ms")
    return img_result, outer_rect, inner_rect, img_raw

def detect_deepest_inner_circle(frame: np.ndarray, detector: EnhancedCircleDetector) -> Tuple[np.ndarray, Optional[Tuple[int, int]]]:
    """优化的圆形检测函数"""
    start_time = time.time()
    result_frame = frame.copy()
    
    # 图像预处理
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # ROI处理
    height, width = bilateral.shape[:2]
    margin = 90
    x1, y1 = max(0, margin), max(0, margin)
    x2, y2 = min(width - margin, width), min(height - margin, height)
    
    if x2 <= x1 or y2 <= y1:
        roi = bilateral
        x1, y1, x2, y2 = 0, 0, width, height
    else:
        roi = bilateral[y1:y2, x1:x2]
    
    # 自适应阈值处理
    roi_thresh = cv2.adaptiveThreshold(
        roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    thresh_full = np.zeros_like(bilateral)
    thresh_full[y1:y2, x1:x2] = roi_thresh
    
    # 形态学操作
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_rect = np.ones((2, 2), np.uint8)
    opened = cv2.morphologyEx(thresh_full, cv2.MORPH_OPEN, kernel_rect)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_ellipse)
    
    # 轮廓检测 --- 直接找出所有轮廓，不需要层级关系
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if hierarchy is None or len(hierarchy) == 0 or len(contours) == 0:
        return result_frame, None
    
    hierarchy = hierarchy[0]
    
    # 轮廓筛选 --- 对所有轮廓进行圆度（>0.8）和面积(>200)筛选，筛选得到candidate_circles: (nX4)数组:[[x,y,r,area]]
    candidate_circles = []
    
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < 200:  # 面积筛选
            continue
            
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
            
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.8:  # 圆度筛选
            continue
            
        # 计算最小外接圆
        (x, y), radius = cv2.minEnclosingCircle(contour)
        candidate_circles.append([int(x), int(y), int(radius), int(area)])
    
    if not candidate_circles:
        return result_frame, None
    
    candidate_circles = np.array(candidate_circles)
    print(f"[DEBUG] 筛选出 {len(candidate_circles)} 个候选圆")
    
    # 筛选靶心 --- 遍历所有 candidate_circles 面积最小的圆作为 innerest_circle，用红色画出圆和圆心，并在图上画出坐标
    min_area_idx = np.argmin(candidate_circles[:, 3])  # 找到面积最小的圆
    innerest_circle = candidate_circles[min_area_idx]
    innerest_x, innerest_y, innerest_r, innerest_area = innerest_circle
    
    # 用红色画出靶心圆和圆心
    cv2.circle(result_frame, (innerest_x, innerest_y), innerest_r, (0, 0, 255), 2)
    cv2.circle(result_frame, (innerest_x, innerest_y), 3, (0, 0, 255), -1)
    cv2.putText(result_frame, f"靶心: ({innerest_x}, {innerest_y})", 
                (innerest_x + 10, innerest_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    print(f"[DEBUG] 靶心圆: 中心({innerest_x}, {innerest_y}), 半径{innerest_r}, 面积{innerest_area}")
    
    # 筛选目标圆 --- 筛出半径在 90 - 120 的圆，记作 target_circles：(nX3)，[[x,y,r]]
    target_circles = []
    for circle in candidate_circles:
        x, y, r, area = circle
        if 90 <= r <= 120:
            target_circles.append([x, y, r])
            # 用蓝色画出所有的目标圆和圆心
            cv2.circle(result_frame, (x, y), r, (255, 0, 0), 2)
            cv2.circle(result_frame, (x, y), 3, (255, 0, 0), -1)
            cv2.putText(result_frame, f"目标: ({x}, {y})", 
                        (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            print(f"[DEBUG] 目标圆: 中心({x}, {y}), 半径{r}")
    
    target_circle = None
    if len(target_circles) == 1:
        # 如果只有一个圆，则直接赋值给 target_circle
        target_circle = target_circles[0]
        print(f"[DEBUG] 检测到一个目标圆: {target_circle}")
    elif len(target_circles) > 1:
        # 如果 target_circles 存储超过一个圆，则将 target_circle 的各个数据取 target_circles 所有圆的平均值
        target_circles = np.array(target_circles)
        avg_x = int(np.mean(target_circles[:, 0]))
        avg_y = int(np.mean(target_circles[:, 1]))
        avg_r = int(np.mean(target_circles[:, 2]))
        target_circle = [avg_x, avg_y, avg_r]
        print(f"[DEBUG] 检测到{len(target_circles)}个目标圆，平均值: {target_circle}")
    else:
        print("[DEBUG] 未检测到半径在90-120之间的目标圆")
    
    # 打印出 target_circle 的相关信息，并用绿色画出圆和圆心
    if target_circle is not None:
        tx, ty, tr = target_circle
        cv2.circle(result_frame, (tx, ty), tr, (0, 255, 0), 3)
        cv2.circle(result_frame, (tx, ty), 5, (0, 255, 0), -1)
        cv2.putText(result_frame, f"最终目标: ({tx}, {ty}), R={tr}", 
                    (tx - 80, ty - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        print(f"[DEBUG] 最终目标圆: 中心({tx}, {ty}), 半径{tr}")
        
        # 返回最终目标圆的中心点
        end_time = time.time()
        print(f"[TIMING] detect_deepest_inner_circle 耗时: {(end_time - start_time)*1000:.2f}ms")
        return result_frame, (tx, ty)
    else:
        # 如果没有目标圆，返回靶心
        end_time = time.time()
        print(f"[TIMING] detect_deepest_inner_circle 耗时: {(end_time - start_time)*1000:.2f}ms")
        return result_frame, (innerest_x, innerest_y)


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    time.sleep(1)
    # ==================================/--.//////////////;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;/-}
    if not cap.isOpened():
        print("无法打开摄像头")
        exit(1)
    
    # 初始化检测器
    detector = EnhancedCircleDetector()
    print("使用优化版圆形检测器")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"摄像头已打开，实际分辨率: {width}x{height}")

    prev_time = time.time()
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法获取帧")
            break
        
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / (now - prev_time)) if prev_time else 0
        prev_time = now

        # 1. 检测嵌套矩形
        result_image, outer_rect, inner_rect, raw_img = detect_nested_rects(frame)
        
        if result_image is not None:
            cv2.putText(result_image, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
            cv2.imshow('Nested Rectangles Detection', result_image)

        # 2. 裁剪ROI并检测圆形
        if outer_rect is not None:
            corners = outer_rect['corners']
            xs = corners[:,0]
            ys = corners[:,1]
            x_min = max(xs.min() - 20, 0)
            x_max = min(xs.max() + 20, frame.shape[1])
            y_min = max(ys.min() - 20, 0)
            y_max = min(ys.max() + 20, frame.shape[0])
            print(f"[DEBUG] ROI区域: x={x_min}-{x_max}, y={y_min}-{y_max}")
            
            if y_max > y_min and x_max > x_min:
                roi = raw_img[y_min:y_max, x_min:x_max]
                if roi.size > 0:
                    roi_resized = cv2.resize(roi, (640, 480), interpolation=cv2.INTER_LANCZOS4)
                    
                    # 使用优化的圆形检测
                    circle_result, circle_center = detect_deepest_inner_circle(roi_resized, detector)
                    
                    if circle_result is not None:
                        cv2.imshow('Circle Detection', circle_result)
                    else:
                        cv2.imshow('ROI', roi_resized)
                else:
                    cv2.imshow('ROI', np.zeros((640, 480, 3), dtype=np.uint8))
            else:
                cv2.imshow('ROI', np.zeros((640, 480, 3), dtype=np.uint8))
        else:
            cv2.imshow('ROI', np.zeros((640, 480, 3), dtype=np.uint8))

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC退出
            break
        elif key == ord('r'):  # 按R键重置检测器
            detector.center_history.clear()
            detector.quality_history.clear()
            print("检测器已重置")
            
    cap.release()
    cv2.destroyAllWindo