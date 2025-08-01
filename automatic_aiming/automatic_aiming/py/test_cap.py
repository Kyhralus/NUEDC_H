#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import time

def detect_nested_rectangles_from_frame(img_raw):
    if img_raw is None:
        return None, [], None, None

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

    for pair in nested_pairs:
        outer = pair['outer']
        inner = pair['inner']
        cv2.drawContours(img_result, [outer['approx']], 0, (0, 255, 0), 2)
        cv2.drawContours(img_result, [inner['approx']], 0, (255, 0, 0), 2)
        cv2.circle(img_result, outer['center'], 3, (0, 0, 255), -1)
        cv2.circle(img_result, inner['center'], 3, (0, 0, 255), -1)
        cv2.line(img_result, outer['center'], inner['center'], (0, 255, 255), 2)
        cv2.putText(img_result, f"{pair['area_ratio']:.2f}", inner['center'], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    return img_result, nested_pairs, outer_rect, img_raw

if __name__ == "__main__":
    cap = cv2.VideoCapture(2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
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

        result_image, nested_pairs, outer_rect, raw_img = detect_nested_rectangles_from_frame(frame)
        
        # 显示帧率
        if result_image is not None:
            cv2.putText(result_image, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
            cv2.imshow('Nested Rectangles Detection', result_image)

        # 裁剪ROI并resize显示
        if outer_rect is not None:
            corners = outer_rect['corners']
            # 计算外扩30像素的最小外接矩形区域
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

