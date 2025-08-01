#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np

def detect_nested_rectangles(image_path):
    """
    检测图像中的嵌套矩形
    
    Args:
        image_path: 图像路径
        
    Returns:
        img_result: 标记了嵌套矩形的图像
        nested_pairs: 嵌套矩形对列表
    """
    # 1. 读取图片
    img_raw = cv2.imread(image_path)
    if img_raw is None:
        print("图片读取失败！")
        return None, []
    
    # 2. 图像预处理
    img_gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
    img_bilateral = cv2.bilateralFilter(img_gray, 9, 10, 10)
    
    # 3. 形态学操作
    kernel = np.ones((5, 5), np.uint8)
    img_close = cv2.morphologyEx(img_bilateral, cv2.MORPH_CLOSE, kernel)
    
    # 4. 边缘检测
    edged = cv2.Canny(img_close, 50, 150)
    
    # 5. 轮廓检测
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # 6. 筛选矩形轮廓
    rectangles = []
    for i, contour in enumerate(contours):
        # 计算轮廓周长
        epsilon = 0.03 * cv2.arcLength(contour, True)
        # 多边形近似
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # 筛选四边形、面积大于1000且为凸多边形的轮廓
        if len(approx) == 4 and cv2.contourArea(approx) > 1000 and cv2.isContourConvex(approx):
            # 获取矩形信息
            x, y, w, h = cv2.boundingRect(approx)
            area = cv2.contourArea(approx)
            rectangles.append({
                'id': i,
                'contour': contour,
                'approx': approx,
                'bbox': (x, y, w, h),
                'area': area,
                'center': (x + w // 2, y + h // 2)
            })
    
    # 7. 按面积从大到小排序
    rectangles.sort(key=lambda r: r['area'], reverse=True)
    
    # 8. 检测嵌套矩形对
    nested_pairs = []
    img_result = img_raw.copy()
    
    # 使用层次关系检测嵌套矩形
    for i, outer in enumerate(rectangles):
        x1, y1, w1, h1 = outer['bbox']
        for j, inner in enumerate(rectangles):
            if i == j:  # 跳过自身
                continue
                
            x2, y2, w2, h2 = inner['bbox']
            
            # 检查内部矩形是否完全包含在外部矩形内
            is_nested = (x1 < x2 and y1 < y2 and 
                        x1 + w1 > x2 + w2 and 
                        y1 + h1 > y2 + h2)
            
            # 计算面积比例，避免太接近的矩形
            area_ratio = inner['area'] / outer['area'] if is_nested else 0
            
            # 计算中心点距离
            center_distance = np.sqrt(
                (outer['center'][0] - inner['center'][0])**2 + 
                (outer['center'][1] - inner['center'][1])**2
            )
            
            # 检查是否是直接嵌套（没有中间矩形）
            is_direct_nested = True
            if is_nested:
                for k, middle in enumerate(rectangles):
                    if k == i or k == j:
                        continue
                    
                    x3, y3, w3, h3 = middle['bbox']
                    # 检查是否有矩形在外部和内部矩形之间
                    if (x1 < x3 and y1 < y3 and 
                        x1 + w1 > x3 + w3 and 
                        y1 + h1 > y3 + h3 and
                        x3 < x2 and y3 < y2 and 
                        x3 + w3 > x2 + w2 and 
                        y3 + h3 > y2 + h2):
                        is_direct_nested = False
                        break
            
            # 如果是直接嵌套且面积比例合适（内部矩形明显小于外部矩形）
            if is_nested and is_direct_nested and 0.1 < area_ratio < 0.9:
                # 检查是否已经添加过这对矩形
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
    
    # 9. 绘制结果
    for pair in nested_pairs:
        outer = pair['outer']
        inner = pair['inner']
        
        # 绘制外部矩形（绿色）
        cv2.drawContours(img_result, [outer['approx']], 0, (0, 255, 0), 2)
        # 绘制内部矩形（蓝色）
        cv2.drawContours(img_result, [inner['approx']], 0, (255, 0, 0), 2)
        
        # 绘制中心点（红色）
        cv2.circle(img_result, outer['center'], 1, (0, 0, 255), -1)
        cv2.circle(img_result, inner['center'], 1, (0, 0, 255), -1)
        
        # 连接两个中心点
        cv2.line(img_result, outer['center'], inner['center'], (0, 255, 255), 2)
        
        print(f"嵌套矩形对: 外部{outer['bbox']} 内部{inner['bbox']} 面积比: {pair['area_ratio']:.2f}")
    
    return img_result, nested_pairs

if __name__ == "__main__":
    # 测试函数
    image_path = '/home/orangepi/ros2_workspace/opencv_demo/opencv_demo/py/test.jpg'
    result_image, nested_pairs = detect_nested_rectangles(image_path)
    
    if result_image is not None:
        print(f"找到 {len(nested_pairs)} 对嵌套矩形")
        cv2.imshow('result', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()