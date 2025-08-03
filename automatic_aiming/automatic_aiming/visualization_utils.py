#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np

def render_results(frame, outer_rect, inner_rect, target_center, target_circle, fps, frame_count, target_circle_area):
    """渲染检测结果"""
    result_image = frame.copy()
    
    rect_detected = circle_detected = False
    
    # 计算图像中心
    image_center_x = frame.shape[1] // 2
    image_center_y = frame.shape[0] // 2
    image_center = (image_center_x, image_center_y)
    
    # 绘制图像中心
    cv2.circle(result_image, image_center, 5, (255, 255, 255), -1)
    cv2.line(result_image, (image_center_x - 10, image_center_y), (image_center_x + 10, image_center_y), (255, 255, 255), 2)
    cv2.line(result_image, (image_center_x, image_center_y - 10), (image_center_x, image_center_y + 10), (255, 255, 255), 2)
    cv2.putText(result_image, "Image Center", (image_center_x + 15, image_center_y - 15), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # 绘制矩形
    if outer_rect and inner_rect:
        rect_detected = True
        cv2.drawContours(result_image, [outer_rect['approx']], 0, (0, 255, 0), 2)
        cv2.drawContours(result_image, [inner_rect['approx']], 0, (255, 0, 0), 2)
        
        cv2.putText(result_image, "Outer Rect", 
                   (outer_rect['bbox'][0], outer_rect['bbox'][1]-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(result_image, "Inner Rect", 
                   (inner_rect['bbox'][0], inner_rect['bbox'][1]-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # 绘制圆形目标
    if target_center or target_circle:
        circle_detected = True
        
        if target_center:
            target_center = (target_center[0], target_center[1])
            cv2.circle(result_image, target_center, 5, (0, 0, 255), -1)  # 靶心点
            cv2.putText(result_image, f"Target Center: ({target_center[0]}, {target_center[1]})", 
                       (target_center[0]+15, target_center[1]-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # 计算并显示目标中心与图像中心的偏差
            center_err_x = target_center[0] - image_center_x
            center_err_y = target_center[1] - image_center_y
            cv2.line(result_image, image_center, target_center, (0, 165, 255), 2)
            cv2.putText(result_image, f"Center Error: ({center_err_x}, {center_err_y})", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
        if target_circle:
            tc_x, tc_y, tc_r = target_circle
            cv2.circle(result_image, (tc_x, tc_y), tc_r, (0, 255, 0), 2)  # 目标圆，线宽为2
    
    # 添加状态信息
    cv2.putText(result_image, f"FPS: {fps:.1f}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(result_image, f"Frame: {frame_count}", (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # 显示检测状态
    cv2.putText(result_image, f"Detection Status: {'Detected' if target_center else 'Not Detected'}", 
               (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    # 目标圆面积
    cv2.putText(result_image, f"target circle area: {target_circle_area}", (10, 200), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # 状态显示
    if rect_detected and circle_detected:
        status_text, status_color = "Target Detected", (0, 255, 255)
    elif rect_detected:
        status_text, status_color = "Rectangle Detected", (0, 165, 255)
    else:
        status_text, status_color = "No Target", (0, 0, 255)
    
    text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    text_x = result_image.shape[1] - text_size[0] - 10
    text_y = 30
    
    cv2.rectangle(result_image, (text_x-5, text_y-25), (text_x+text_size[0]+5, text_y+5), (0, 0, 0), -1)
    cv2.putText(result_image, status_text, (text_x, text_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    
    return result_image
