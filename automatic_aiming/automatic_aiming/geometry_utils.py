#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np

def calculate_angle(pt1, pt2, pt3):
    """计算由三个点构成的角的角度（pt2为顶点）"""
    # 向量pt2->pt1和pt2->pt3
    vec1 = (pt1[0] - pt2[0], pt1[1] - pt2[1])
    vec2 = (pt3[0] - pt2[0], pt3[1] - pt2[1])
    
    # 点积和模长
    dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
    len1 = (vec1[0]**2 + vec1[1]** 2) **0.5
    len2 = (vec2[0]** 2 + vec2[1] ** 2) **0.5
    
    if len1 == 0 or len2 == 0:
        return 0.0  # 避免除以零
    
    # 计算夹角（弧度转角度）
    cos_theta = max(-1.0, min(1.0, dot_product / (len1 * len2)))  # 防止数值溢出
    angle = np.arccos(cos_theta) * (180 / np.pi)
    return angle

def sort_corners(corners):
    """对角点进行排序：左上、右上、右下、左下"""
    # 计算质心
    centroid = np.mean(corners, axis=0)
    
    # 按角度排序
    def angle_from_centroid(point):
        return np.arctan2(point[1] - centroid[1], point[0] - centroid[0])
    
    sorted_corners = sorted(corners, key=angle_from_centroid)
    
    # 找到最左上角的点作为起始点
    top_left_idx = 0
    min_dist = float('inf')
    for i, corner in enumerate(sorted_corners):
        dist = corner[0] + corner[1]  # 到左上角(0,0)的曼哈顿距离
        if dist < min_dist:
            min_dist = dist
            top_left_idx = i
    
    # 重新排列，从左上角开始顺时针
    reordered = sorted_corners[top_left_idx:] + sorted_corners[:top_left_idx]
    
    return np.array(reordered)
