#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import time
from typing import Optional, Tuple

def detect_nested_rectangles_from_frame(img_raw):
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
                    # 只取面积最大的外层矩形
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
    return img_result, nested_pairs, outer_rect, inner_rect, img_raw

def detect_deepest_inner_circle(frame: np.ndarray) -> Tuple[np.ndarray, Optional[Tuple[int, int]]]:
    """检测层级最多的轮廓结构中最里层的圆心"""
    candidate_frame = frame.copy()
    hier_frame = frame.copy()
    result_frame = frame.copy()
    
    # # 图像预处理
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # # debug
    # cv2.imshow("blur Image", blurred)
    # # # 使用大津法进行阈值处理
    # _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # roi区域大津法
    # 图像预处理
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imshow("Blur Image", blurred)

    # 获取图像尺寸
    height, width = blurred.shape[:2]

    # 计算ROI区域坐标（向内缩50像素）
    margin = 90
    x1, y1 = margin, margin
    x2, y2 = width - margin, height - margin

    # 安全边界处理，防止图像过小导致负数
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)

    # 裁剪ROI区域
    roi = blurred[y1:y2, x1:x2]

    # Otsu阈值处理，仅对ROI部分
    _, roi_thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 创建一个与原图同尺寸的全黑图，将处理结果放回对应位置
    thresh_full = np.zeros_like(blurred)
    thresh_full[y1:y2, x1:x2] = roi_thresh

    # 显示结果
    cv2.imshow("Thresholded ROI", thresh_full)


    # 区间阈值处理，保留120~136之间的像素
    # thresh = cv2.inRange(blurred, 90, 140)
    # 形态学操作优化轮廓
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh_full, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # debug
    cv2.imshow("Thresholded Image", thresh)
    # canny

    # 获取轮廓和层级信息 (RETR_TREE 模式)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if hierarchy is None or len(hierarchy) == 0 or len(contours) == 0:
        print("未找到轮廓")
        return result_frame, None
    
    hierarchy = hierarchy[0]  # 获取层级数组
    
    # ===== 步骤1：保留符合条件的父轮廓下的所有子轮廓，并在子轮廓中筛选圆度 =====
    candidate_circles = []
    skipped_circles = []   # 记录被判定为重复的子轮廓索引
    valid_parents = set()

    # 判断是否重复的函数
    def is_duplicate_circle(new_contour, existing_circles, dist_thresh=5, radius_thresh=5):
        (nx, ny), nr = cv2.minEnclosingCircle(new_contour)
        for c in existing_circles:
            (ex, ey), er = cv2.minEnclosingCircle(c['contour'])
            if abs(nx - ex) < dist_thresh and abs(ny - ey) < dist_thresh and abs(nr - er) < radius_thresh:
                return True
        return False

    # 遍历所有轮廓找父轮廓
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area < 200:
            continue
        perimeter = cv2.arcLength(contours[i], True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity >= 0.8:
            valid_parents.add(i)

    # 第二遍递归收集子轮廓
    def add_valid_child_contours(current_idx):
        child_idx = hierarchy[current_idx][2]  # 第一个子轮廓
        while child_idx != -1:
            area = cv2.contourArea(contours[child_idx])
            perimeter = cv2.arcLength(contours[child_idx], True)
            if perimeter == 0:
                child_idx = hierarchy[child_idx][0]
                continue

            circularity = 4 * np.pi * area / (perimeter * perimeter)

            if area >= 30 and circularity >= 0.89:
                # 去重判断
                if not is_duplicate_circle(contours[child_idx], candidate_circles):
                    candidate_circles.append({
                        'index': child_idx,
                        'contour': contours[child_idx],
                        'area': area,
                        'circularity': circularity,
                        'parent_index': current_idx
                    })
                else:
                    skipped_circles.append(child_idx)  # 记录被去掉的重复轮廓

            # 递归处理子轮廓
            add_valid_child_contours(child_idx)
            child_idx = hierarchy[child_idx][0]

    for parent_idx in valid_parents:
        add_valid_child_contours(parent_idx)

    # 调试输出
    if skipped_circles:
        print(f"去掉了 {len(skipped_circles)} 个重复轮廓: {skipped_circles}")


    if not candidate_circles:
        return result_frame, None

    # 不显示轮廓
    # # ===== 步骤2：计算candidate_circles每个轮廓的层级深度 =====
    # circle_levels = {}
    # for circle in candidate_circles:
    #     idx = circle['index']
    #     level = 0
    #     next_idx = idx
    #     while hierarchy[next_idx][3] != -1:  # 向上遍历至最外层轮廓
    #         next_idx = hierarchy[next_idx][3]
    #         level += 1
    #     circle_levels[idx] = level
        # ===== 步骤2：计算candidate_circles每个轮廓的层级深度，并用不同颜色标注 =====
    circle_levels = {}
    max_level = 0
    for circle in candidate_circles:
        idx = circle['index']
        level = 0
        next_idx = idx
        while hierarchy[next_idx][3] != -1:  # 向上遍历至最外层轮廓
            next_idx = hierarchy[next_idx][3]
            level += 1
        circle_levels[idx] = level
        if level > max_level:
            max_level = level

    # 定义一组可循环的颜色
    COLORS = [
        (255, 0, 0),      # 蓝
        (0, 255, 0),      # 绿
        (0, 0, 255),      # 红
        (255, 255, 0),    # 青
        (255, 0, 255),    # 紫
        (0, 255, 255),    # 黄
        (128, 128, 255),  # 淡紫
        (255, 128, 128),  # 粉
        (128, 255, 128),  # 浅绿
        (128, 255, 255),  # 浅黄
    ]

        # 绘制所有候选圆形轮廓，按层级染色，并打印信息
    for circle in candidate_circles:
        idx = circle['index']
        level = circle_levels[idx]
        color = COLORS[level % len(COLORS)]
        cv2.drawContours(result_frame, [circle['contour']], -1, color, 2)
        # 可选：在圆心处标注层级
        M = cv2.moments(circle['contour'])
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(result_frame, f"L{level}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        # 打印层级和圆信息
        print(f"圆索引: {idx}, 层级: {level}, 面积: {circle['area']:.2f}, 圆度: {circle['circularity']:.3f}, 父轮廓: {circle['parent_index']}")
    # ===== 步骤3：计算候选轮廓的平均层级，选择层级大于平均层级的轮廓作为inner_circles =====
    if not circle_levels:
        return result_frame, None

    # ===== 步骤3：计算候选轮廓的平均层级，选择层级大于平均层级的轮廓作为inner_circles =====
    if not circle_levels:
        return result_frame, None

    # 计算平均层级
    average_level = sum(circle_levels.values()) / len(circle_levels)

    # 选择层级大于平均层级的轮廓
    inner_circles = [c for c in candidate_circles if circle_levels[c['index']] >= average_level]

    # 如果没有找到，回退到选择最深层级
    if not inner_circles:
        print(f"没有找到层级大于平均值的轮廓")

    if not inner_circles:
        print("没有找到符合条件的内层轮廓")
        return result_frame, None

    # ===== 步骤4：在inner_circles选取面积最小的轮廓作为靶心 =====
    # 按面积排序，选择最小的
    inner_circles.sort(key=lambda x: x['area'])
    target_circle = inner_circles[0]

    # 计算圆心坐标
    M = cv2.moments(target_circle['contour'])
    if M["m00"] == 0:  # 防止除零错误
        print("Failed to compute center")
        return result_frame, None

    target_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    
    # 可视化靶心
    cv2.circle(result_frame, target_center, 1, (0, 0, 255), -1)  # 红色实心圆标记圆心
    cv2.putText(result_frame, f"Target: ({target_center[0]}, {target_center[1]})",
            (target_center[0] - 100, target_center[1] - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # ===== 步骤4扩展：检测半径在90-120的圆 =====
    circles_90_120 = []
    for circle in candidate_circles:
        (x, y), radius = cv2.minEnclosingCircle(circle['contour'])
        if 90 <= radius <= 120:   
            circles_90_120.append((int(x), int(y), int(radius)))
            print(f"找到90-120范围内的圆: ({int(x)}, {int(y)}, r={int(radius)})")


    target_circle_90_120 = None

    if len(circles_90_120) == 1:
        # 只有一个符合条件的圆
        target_circle_90_120 = circles_90_120[0]
    elif len(circles_90_120) >= 2:
        # 找到两个或以上，取半径最小的为内层圆
        circles_90_120.sort(key=lambda c: c[2])  # 按半径排序
        inner_circle = circles_90_120[0]
        outer_circle = circles_90_120[1]

        avg_half_radius = int((inner_circle[2] + outer_circle[2]) / 2 / 2)
        target_circle_90_120 = (inner_circle[0], inner_circle[1], avg_half_radius)

    # 绘制结果
    if target_circle_90_120 is not None:
        x, y, r = target_circle_90_120
        cv2.circle(result_frame, (x, y), r, (255, 0, 255), 2)        # 紫色画圆
        cv2.circle(result_frame, (x, y), 1, (255, 0, 255), -1)        # 紫色画圆心
        cv2.putText(result_frame, f"Target90_120: ({x}, {y}, r={r})",
                    (x - 100, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    return result_frame, target_center

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 200)
    time.sleep(1)
    
    if not cap.isOpened():
        print("无法打开摄像头")
        exit(1)
    
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
        
        # 显示帧率
        if result_image is not None:
            cv2.putText(result_image, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
            cv2.imshow('Nested Rectangles Detection', result_image)

        # 2. 裁剪ROI并resize显示
        if outer_rect is not None:
            corners = outer_rect['corners']
            # 计算外扩20像素的最小外接矩形区域
            xs = corners[:,0]
            ys = corners[:,1]
            x_min = max(xs.min() - 20, 0)
            x_max = min(xs.max() + 20, frame.shape[1])
            y_min = max(ys.min() - 20, 0)
            y_max = min(ys.max() + 20, frame.shape[0])
            
            if y_max > y_min and x_max > x_min:  # 确保ROI有效
                roi = raw_img[y_min:y_max, x_min:x_max]
                if roi.size > 0:  # 确保ROI不为空
                    roi_resized = cv2.resize(roi, (640, 480))  # 固定大小显示
                    
                    # 3. 在ROI中检测圆环
                    circle_result, circle_center = detect_deepest_inner_circle(roi_resized)
                    
                    # 显示圆环检测结果
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
            
    cap.release()
    cv2.destroyAllWindows()