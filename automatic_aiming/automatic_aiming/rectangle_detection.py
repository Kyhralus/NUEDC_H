#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import time
from .geometry_utils import calculate_angle

def detect_nested_rectangles_optimized(edged_image, timing_stats):
    """优化的嵌套矩形检测 - 增加稳定性并显示每步处理结果"""
    rect_start = time.time()
    
    # 1. 形态学操作
    kernel = np.ones((5, 5), np.uint8)
    edged_stable = cv2.morphologyEx(edged_image, cv2.MORPH_CLOSE, kernel)
    edged_stable = cv2.dilate(edged_stable, kernel, iterations=1)
    
    # 2. 轮廓检测
    contours, hierarchy = cv2.findContours(edged_stable, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # 3. 筛选矩形
    rectangles = []
    filter_img = cv2.cvtColor(edged_stable, cv2.COLOR_GRAY2BGR)  # 用于显示筛选过程

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        # 面积筛选
        if area < 6000:
            cv2.drawContours(filter_img, [contour], -1, (128, 128, 128), 1)
            continue
        
        # 多边形逼近
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # 顶点数和凸性筛选
        if not (4 <= len(approx) <= 6 and cv2.isContourConvex(approx)):
            cv2.drawContours(filter_img, [contour], -1, (0, 255, 255), 1)
            continue
        
        # 非四边形时拟合矩形（确保最终是4个顶点）
        if len(approx) != 4:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            approx = np.int0(box).reshape(-1, 1, 2)
        
        # 提取四个角点（按顺序排列，确保相邻性）
        corners = approx.reshape(4, 2)  # 转为4x2的角点列表
        
        # 检查四个角的角度是否在90°±25°范围内
        valid_corners = True
        for k in range(4):
            # 三个连续点：前一个点、当前点（顶点）、后一个点（循环取点）
            pt_prev = corners[(k - 1) % 4]
            pt_current = corners[k]
            pt_next = corners[(k + 1) % 4]
            
            angle = calculate_angle(pt_prev, pt_current, pt_next)
            
            # 角度不在65°~115°范围内，标记为无效
            if not (65 <= angle <= 115):
                valid_corners = False
                break
        
        if not valid_corners:
            # 用紫色绘制角度不符合的轮廓
            cv2.drawContours(filter_img, [approx], -1, (128, 0, 128), 1)
            continue
        
        # 长宽比筛选
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / h if h > 0 else 0
        if not (0.8 < aspect_ratio < 1.5):
            cv2.drawContours(filter_img, [approx], -1, (0, 0, 255), 1)
            continue
        
        # 符合条件的矩形加入列表
        rectangles.append({
            'id': i,
            'contour': contour,
            'approx': approx,
            'bbox': (x, y, w, h),
            'area': area,
            'center': (x + w // 2, y + h // 2),
            'corners': corners
        })
        # 用蓝色绘制通过筛选的矩形
        cv2.drawContours(filter_img, [approx], -1, (255, 0, 0), 2)

    # 4. 按面积排序
    rectangles.sort(key=lambda r: r['area'], reverse=True)

    # 5. 寻找嵌套矩形对
    outer_rect = inner_rect = None
    nested_img = cv2.cvtColor(edged_stable, cv2.COLOR_GRAY2BGR)

    for i, outer in enumerate(rectangles[:10]):
        x1, y1, w1, h1 = outer['bbox']
        cv2.rectangle(nested_img, (x1, y1), (x1 + w1, y1 + h1), (255, 165, 0), 2)
        
        for j, inner in enumerate(rectangles):
            if i == j:
                continue
            x2, y2, w2, h2 = inner['bbox']
            
            # 嵌套条件判断
            margin = 5
            is_nested = (x1 <= x2 + margin and y1 <= y2 + margin and 
                        x1 + w1 >= x2 + w2 - margin and y1 + h1 >= y2 + h2 - margin)
            
            if is_nested:
                area_ratio = inner['area'] / outer['area']
                if 0.7 < area_ratio < 0.9:
                    outer_rect = outer
                    inner_rect = inner
                    cv2.rectangle(nested_img, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)
                    cv2.rectangle(nested_img, (x2, y2), (x2 + w2, y2 + h2), (0, 0, 255), 2)
                    cv2.putText(nested_img, "Outer", (x1, y1 - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(nested_img, "Inner", (x2, y2 - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    break
        if outer_rect is not None:
            break
    
    # 计算耗时
    rect_time = (time.time() - rect_start) * 1000
    timing_stats['rectangle_detection'].append(rect_time)
    
    return outer_rect, inner_rect
