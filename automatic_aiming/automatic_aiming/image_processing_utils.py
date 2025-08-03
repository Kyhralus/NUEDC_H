#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import time

def crop_center_and_flip(image, target_width=960, target_height=720):
    """从图像中心裁剪指定尺寸区域，并进行上下翻转（高效实现）"""
    h, w = image.shape[:2]
    
    # 1. 计算中心裁剪的起始坐标
    start_x = max(0, (w - target_width) // 2)
    start_y = max(0, (h - target_height) // 2)
    
    # 2. 裁剪（直接切片操作，效率极高）
    end_x = start_x + min(target_width, w - start_x)
    end_y = start_y + min(target_height, h - start_y)
    cropped = image[start_y:end_y, start_x:end_x]
    
    # 3. 若尺寸不足则缩放（仅在必要时执行）
    if cropped.shape[0] != target_height or cropped.shape[1] != target_width:
        cropped = cv2.resize(cropped, (target_width, target_height), 
                        interpolation=cv2.INTER_LINEAR)
    
    # 4. 上下翻转（OpenCV底层优化，耗时极短）
    flipped = cv2.flip(cropped, 0)  # flipCode=0 表示沿x轴翻转（上下翻转）
    
    return flipped

def preprocess_image(frame, timing_stats):
    """优化的统一预处理步骤"""
    preprocess_start = time.time()
    
    # 一次性完成灰度化和高斯模糊
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 使用高斯模糊替代双边滤波（更快）
    blurred = cv2.GaussianBlur(gray, (3, 3), 0.8)
    h, w = blurred.shape[:2]

    # 2. 提取中间640x480区域（若原图小于该尺寸则用全图）
    crop_w, crop_h = 640, 480
    # 计算中心区域坐标
    start_x = max(0, (w - crop_w) // 2)
    start_y = max(0, (h - crop_h) // 2)
    end_x = min(w, start_x + crop_w)
    end_y = min(h, start_y + crop_h)
    # 裁剪中心区域
    center_roi = blurred[start_y:end_y, start_x:end_x]
     # 3. 对中心区域计算Otsu阈值
    otsu_thresh, _ = cv2.threshold(center_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 4. 用该阈值对全图进行二值化
    _, binary = cv2.threshold(gray, otsu_thresh, 255, cv2.THRESH_BINARY)
    # 可选：轻微形态学操作去除噪点
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # 5. 边缘检测（基于二值化结果）
    edged = cv2.Canny(binary, 50, 150)  # Canny阈值可根据效果调整
    
    preprocess_time = (time.time() - preprocess_start) * 1000
    timing_stats['preprocess'].append(preprocess_time)
    
    return {
        'gray': gray,
        'blurred': blurred,
        'edged': edged
    }
