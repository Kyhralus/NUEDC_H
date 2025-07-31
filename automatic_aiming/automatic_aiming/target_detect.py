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

def detect_nested_rectangles_from_frame(img_raw):
    """检测嵌套矩形"""
    start_time = time.time()
    
    if img_raw is None:
        return None, [], None, None, None

    img_gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
    img_bilateral = cv2.bilateralFilter(img_gray, 9, 10, 10)
    kernel = np.ones((5, 5), np.uint8)
    img_close = cv2.morphologyEx(img_bilateral, cv2.MORPH_CLOSE, kernel)
    edged = cv2.Canny(img_close, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    rectangles = []
    for i, contour in enumerate(contours):
        epsilon = 0.03 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 1000 and cv2.isContourConvex(approx):
            x, y, w, h = cv2.boundingRect(approx)
            area = cv2.contourArea(approx)
            rectangles.append({
                'id': i,
                'contour': contour,
                'approx': approx,
                'bbox': (x, y, w, h),
                'area': area,
                'center': (x + w // 2, y + h // 2),
                'corners': approx.reshape(4, 2)
            })

    rectangles.sort(key=lambda r: r['area'], reverse=True)
    nested_pairs = []
    img_result = img_raw.copy()
    outer_rect = None
    inner_rect = None

    for i, outer in enumerate(rectangles):
        x1, y1, w1, h1 = outer['bbox']
        for j, inner in enumerate(rectangles):
            if i == j:
                continue
            x2, y2, w2, h2 = inner['bbox']
            is_nested = (x1 < x2 and y1 < y2 and x1 + w1 > x2 + w2 and y1 + h1 > y2 + h2)
            area_ratio = inner['area'] / outer['area'] if is_nested else 0
            center_distance = np.sqrt(
                (outer['center'][0] - inner['center'][0])**2 +
                (outer['center'][1] - inner['center'][1])**2
            )
            is_direct_nested = True
            if is_nested:
                for k, middle in enumerate(rectangles):
                    if k == i or k == j:
                        continue
                    x3, y3, w3, h3 = middle['bbox']
                    if (x1 < x3 and y1 < y3 and x1 + w1 > x3 + w3 and y1 + h1 > y3 + h3 and
                        x3 < x2 and y3 < y2 and x3 + w3 > x2 + w2 and y3 + h3 > y2 + h2):
                        is_direct_nested = False
                        break
            if is_nested and is_direct_nested and 0.6 < area_ratio < 0.9:
                pair_exists = False
                for pair in nested_pairs:
                    if pair['outer']['id'] == outer['id'] and pair['inner']['id'] == inner['id']:
                        pair_exists = True
                        break
                if not pair_exists:
                    nested_pairs.append({
                        'outer': outer,
                        'inner': inner,
                        'area_ratio': area_ratio,
                        'center_distance': center_distance
                    })
                    if outer_rect is None:
                        outer_rect = outer
                    if inner_rect is None:
                        inner_rect = inner

    for pair in nested_pairs:
        outer = pair['outer']
        inner = pair['inner']
        cv2.drawContours(img_result, [outer['approx']], 0, (0, 255, 0), 2)
        cv2.drawContours(img_result, [inner['approx']], 0, (255, 0, 0), 2)
        cv2.circle(img_result, outer['center'], 1, (0, 0, 255), -1)
        cv2.circle(img_result, inner['center'], 1, (0, 0, 255), -1)
        cv2.line(img_result, outer['center'], inner['center'], (0, 255, 255), 2)
        cv2.putText(img_result, f"{pair['area_ratio']:.2f}", inner['center'], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    
    end_time = time.time()
    print(f"[TIMING] detect_nested_rectangles_from_frame 耗时: {(end_time - start_time)*1000:.2f}ms")
    return img_result, nested_pairs, outer_rect, inner_rect, img_raw

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
        result_image, nested_pairs, outer_rect, inner_rect, raw_img = detect_nested_rectangles_from_frame(frame)
        
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
                    roi_resized = cv2.resize(roi, (640, 480))
                    
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
    cv2.destroyAllWindows()


# 修改内容：
    # 1. 删除其他没用的函数和类。
    # 2. 删除不必要的展示，仅保留最终结果，将检测到的矩形和最终识别的target圆，圆心和靶心显示在同一张图上，并显示帧率
    # 3. 记录每一个函数运行，以及总体运行的时间。
    # 4.  写成ros2 节点类型，订阅 /imgae_raw 话题获得图像，发布 /target_data 话题发布目标数据。发布的消息类型为"std_msgs/msg/string"。消息格式:靶心“p,x,y”，其中p为目标类型，x,y为目标中心坐标；圆环“c,x,y,r”，其中c为目标类型，x,y为圆心坐标，r为半径。
    # 5. 代码中添加注释，便于理解。