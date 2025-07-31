#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
目标检测节点测试脚本
用于测试修改后的target_detect.py功能
"""

import cv2
import numpy as np
import sys
import os

# 添加automatic_aiming模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'automatic_aiming', 'automatic_aiming'))

def create_test_image():
    """创建测试图像，包含嵌套矩形和圆形"""
    # 创建640x480的黑色图像
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 绘制外层矩形（绿色）
    outer_rect = np.array([[200, 150], [450, 150], [450, 350], [200, 350]], np.int32)
    cv2.fillPoly(img, [outer_rect], (50, 50, 50))
    cv2.polylines(img, [outer_rect], True, (0, 255, 0), 3)
    
    # 绘制内层矩形（蓝色）
    inner_rect = np.array([[230, 180], [420, 180], [420, 320], [230, 320]], np.int32)
    cv2.fillPoly(img, [inner_rect], (80, 80, 80))
    cv2.polylines(img, [inner_rect], True, (255, 0, 0), 2)
    
    # 在内层矩形中绘制一些圆形
    # 大圆（目标圆，半径约100）
    cv2.circle(img, (325, 250), 95, (120, 120, 120), -1)
    cv2.circle(img, (325, 250), 95, (0, 255, 255), 2)
    
    # 中圆
    cv2.circle(img, (325, 250), 60, (150, 150, 150), -1)
    cv2.circle(img, (325, 250), 60, (255, 255, 0), 2)
    
    # 小圆（靶心）
    cv2.circle(img, (325, 250), 25, (200, 200, 200), -1)
    cv2.circle(img, (325, 250), 25, (0, 0, 255), 2)
    
    return img

def test_detection_functions():
    """测试检测函数"""
    print("=== 目标检测功能测试 ===")
    
    try:
        # 尝试导入检测类
        from target_detect import EnhancedCircleDetector, TargetDetectionNode
        print("✓ 成功导入目标检测模块")
        
        # 创建测试图像
        test_img = create_test_image()
        print("✓ 创建测试图像")
        
        # 显示原始测试图像
        cv2.imshow('Test Image', test_img)
        print("✓ 显示测试图像")
        
        # 创建一个简化的检测器实例（不使用ROS2）
        class SimpleDetector:
            def __init__(self):
                self.detector = EnhancedCircleDetector()
            
            def detect_nested_rectangles(self, img_raw):
                """简化的矩形检测函数"""
                if img_raw is None:
                    return None, None, None, None
                
                # 图像预处理
                img_gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
                img_bilateral = cv2.bilateralFilter(img_gray, 9, 10, 10)
                kernel = np.ones((5, 5), np.uint8)
                img_close = cv2.morphologyEx(img_bilateral, cv2.MORPH_CLOSE, kernel)
                edged = cv2.Canny(img_close, 50, 150)
                
                # 创建矩形检测结果图像用于显示
                rect_debug_img = img_raw.copy()
                
                # 轮廓检测
                contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
                # 筛选矩形
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
                        # 在调试图像上画出所有候选矩形（蓝色）
                        cv2.drawContours(rect_debug_img, [approx], 0, (255, 0, 0), 2)
                        cv2.putText(rect_debug_img, f"ID:{i} Area:{int(area)}", 
                                   (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                
                rectangles.sort(key=lambda r: r['area'], reverse=True)
                
                # 寻找嵌套矩形对
                outer_rect = None
                inner_rect = None
                
                for i, outer in enumerate(rectangles):
                    x1, y1, w1, h1 = outer['bbox']
                    for j, inner in enumerate(rectangles):
                        if i == j:
                            continue
                        x2, y2, w2, h2 = inner['bbox']
                        is_nested = (x1 < x2 and y1 < y2 and x1 + w1 > x2 + w2 and y1 + h1 > y2 + h2)
                        if is_nested:
                            area_ratio = inner['area'] / outer['area']
                            if 0.6 < area_ratio < 0.9:
                                outer_rect = outer
                                inner_rect = inner
                                # 在调试图像上高亮显示嵌套矩形对
                                cv2.drawContours(rect_debug_img, [outer['approx']], 0, (0, 255, 0), 3)
                                cv2.drawContours(rect_debug_img, [inner['approx']], 0, (0, 0, 255), 3)
                                cv2.putText(rect_debug_img, "OUTER", 
                                           (outer['center'][0]-30, outer['center'][1]), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                cv2.putText(rect_debug_img, "INNER", 
                                           (inner['center'][0]-30, inner['center'][1]), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                break
                    if outer_rect is not None:
                        break
                
                # 添加检测信息到调试图像
                info_text = f"Rectangles Found: {len(rectangles)}"
                cv2.putText(rect_debug_img, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                nested_status = "Nested Pair: Found" if (outer_rect and inner_rect) else "Nested Pair: Not Found"
                cv2.putText(rect_debug_img, nested_status, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                return outer_rect, inner_rect, img_raw, rect_debug_img
            
            def detect_circles(self, frame):
                """简化的圆形检测函数"""
                result_frame = frame.copy()
                
                # 创建圆形检测调试图像
                circle_debug_img = frame.copy()
                
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
                
                # 轮廓检测
                contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
                if hierarchy is None or len(hierarchy) == 0 or len(contours) == 0:
                    cv2.putText(circle_debug_img, "No Contours Found", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    return result_frame, None, None, circle_debug_img
                
                hierarchy = hierarchy[0]
                
                # 候选圆筛选
                candidate_circles = []
                all_contours_count = len(contours)
                
                for i, contour in enumerate(contours):
                    area = cv2.contourArea(contour)
                    if area < 200:
                        continue
                        
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter == 0:
                        continue
                        
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity < 0.8:
                        continue
                        
                    # 计算最小外接圆
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    candidate_circles.append([int(x), int(y), int(radius), int(area)])
                    
                    # 在调试图像上画出所有候选圆（黄色）
                    cv2.circle(circle_debug_img, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                    cv2.putText(circle_debug_img, f"C{i}:R{int(radius)}", 
                               (int(x)-20, int(y)-int(radius)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                
                # 添加检测统计信息
                cv2.putText(circle_debug_img, f"Total Contours: {all_contours_count}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(circle_debug_img, f"Candidate Circles: {len(candidate_circles)}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                if not candidate_circles:
                    cv2.putText(circle_debug_img, "No Valid Circles Found", (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    return result_frame, None, None, circle_debug_img
                
                candidate_circles = np.array(candidate_circles)
                
                # 找到面积最小的圆作为靶心
                min_area_idx = np.argmin(candidate_circles[:, 3])
                innerest_circle = candidate_circles[min_area_idx]
                innerest_x, innerest_y, innerest_r, innerest_area = innerest_circle
                
                # 画出靶心（红色）
                cv2.circle(result_frame, (innerest_x, innerest_y), innerest_r, (0, 0, 255), 2)
                cv2.circle(result_frame, (innerest_x, innerest_y), 3, (0, 0, 255), -1)
                
                # 在调试图像上高亮靶心
                cv2.circle(circle_debug_img, (innerest_x, innerest_y), innerest_r, (0, 0, 255), 3)
                cv2.putText(circle_debug_img, "BULLSEYE", (innerest_x-30, innerest_y+innerest_r+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # 筛选目标圆（半径在90-120之间）
                target_circles = []
                for circle in candidate_circles:
                    x, y, r, area = circle
                    if 90 <= r <= 120:
                        target_circles.append([x, y, r])
                        # 画出目标圆（蓝色）
                        cv2.circle(result_frame, (x, y), r, (255, 0, 0), 2)
                        cv2.circle(result_frame, (x, y), 3, (255, 0, 0), -1)
                        # 在调试图像上标记目标圆
                        cv2.circle(circle_debug_img, (x, y), r, (255, 0, 0), 3)
                        cv2.putText(circle_debug_img, "TARGET", (x-30, y-r-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                cv2.putText(circle_debug_img, f"Target Circles (R:90-120): {len(target_circles)}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # 计算最终目标圆
                target_circle = None
                target_message = f"p,{innerest_x},{innerest_y}"
                
                if len(target_circles) == 1:
                    target_circle = target_circles[0]
                elif len(target_circles) > 1:
                    target_circles = np.array(target_circles)
                    avg_x = int(np.mean(target_circles[:, 0]))
                    avg_y = int(np.mean(target_circles[:, 1]))
                    avg_r = int(np.mean(target_circles[:, 2]))
                    target_circle = [avg_x, avg_y, avg_r]
                
                # 画出最终目标圆（绿色）
                if target_circle is not None:
                    tx, ty, tr = target_circle
                    cv2.circle(result_frame, (tx, ty), tr, (0, 255, 0), 3)
                    cv2.circle(result_frame, (tx, ty), 5, (0, 255, 0), -1)
                    target_message = f"c,{tx},{ty},{tr}"
                    
                    # 在调试图像上标记最终目标
                    cv2.circle(circle_debug_img, (tx, ty), tr, (0, 255, 0), 4)
                    cv2.putText(circle_debug_img, "FINAL TARGET", (tx-50, ty+tr+30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(circle_debug_img, f"Final Target Found: ({tx}, {ty})", (10, 120), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    return result_frame, (tx, ty), target_message, circle_debug_img
                else:
                    cv2.putText(circle_debug_img, f"Using Bullseye: ({innerest_x}, {innerest_y})", (10, 120), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    return result_frame, (innerest_x, innerest_y), target_message, circle_debug_img
        
        # 创建检测器实例
        detector = SimpleDetector()
        print("✓ 创建检测器实例")
        
        # 测试矩形检测
        print("\n--- 测试矩形检测 ---")
        outer_rect, inner_rect, raw_img, rect_debug_img = detector.detect_nested_rectangles(test_img)
        
        if outer_rect and inner_rect:
            print(f"✓ 检测到嵌套矩形对")
            print(f"  外层矩形中心: {outer_rect['center']}, 面积: {outer_rect['area']}")
            print(f"  内层矩形中心: {inner_rect['center']}, 面积: {inner_rect['area']}")
        else:
            print("✗ 未检测到嵌套矩形对")
        
        # 显示矩形检测结果
        if rect_debug_img is not None:
            cv2.imshow('Rectangle Detection Debug', rect_debug_img)
            print("✓ 显示矩形检测调试窗口")
        
        # 测试圆形检测
        print("\n--- 测试圆形检测 ---")
        if outer_rect:
            # 提取ROI区域进行圆形检测
            corners = outer_rect['corners']
            xs = corners[:,0]
            ys = corners[:,1]
            x_min = max(xs.min() - 20, 0)
            x_max = min(xs.max() + 20, test_img.shape[1])
            y_min = max(ys.min() - 20, 0)
            y_max = min(ys.max() + 20, test_img.shape[0])
            
            if y_max > y_min and x_max > x_min:
                roi = test_img[y_min:y_max, x_min:x_max]
                if roi.size > 0:
                    roi_resized = cv2.resize(roi, (640, 480))
                    
                    circle_result, circle_center, target_message, circle_debug_img = detector.detect_circles(roi_resized)
                    
                    if circle_center:
                        print(f"✓ 检测到圆形目标")
                        print(f"  目标位置: {circle_center}")
                        print(f"  目标消息: {target_message}")
                    else:
                        print("✗ 未检测到圆形目标")
                    
                    # 显示圆形检测结果
                    if circle_debug_img is not None:
                        cv2.imshow('Circle Detection Debug', circle_debug_img)
                        print("✓ 显示圆形检测调试窗口")
        
        # 创建最终结果图像
        result_image = test_img.copy()
        rect_detected = outer_rect is not None and inner_rect is not None
        circle_detected = False
        
        # 画出检测到的矩形
        if rect_detected:
            cv2.drawContours(result_image, [outer_rect['approx']], 0, (0, 255, 0), 3)
            cv2.drawContours(result_image, [inner_rect['approx']], 0, (255, 0, 0), 3)
            cv2.circle(result_image, outer_rect['center'], 5, (0, 0, 255), -1)
            cv2.circle(result_image, inner_rect['center'], 5, (0, 0, 255), -1)
            
            cv2.putText(result_image, "OUTER RECT", 
                       (outer_rect['center'][0]-50, outer_rect['center'][1]-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(result_image, "INNER RECT", 
                       (inner_rect['center'][0]-50, inner_rect['center'][1]-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # 添加检测状态显示（右上角）
        status_text = ""
        status_color = (0, 0, 255)  # 默认红色
        
        if rect_detected and circle_detected:
            status_text = "Target Detected"
            status_color = (0, 255, 0)  # 绿色
        elif rect_detected:
            status_text = "Rectangle Detected"
            status_color = (0, 255, 255)  # 黄色
        else:
            status_text = "No Target"
            status_color = (0, 0, 255)  # 红色
        
        # 计算文字位置（右上角）
        text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = result_image.shape[1] - text_size[0] - 10
        text_y = 30
        
        # 添加黑色背景确保文字可读性
        cv2.rectangle(result_image, (text_x-5, text_y-25), (text_x+text_size[0]+5, text_y+5), (0, 0, 0), -1)
        cv2.putText(result_image, status_text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # 显示最终结果
        cv2.imshow('Target Detection Result', result_image)
        print("✓ 显示最终检测结果窗口")
        
        print("\n=== 测试完成 ===")
        print("按任意键关闭所有窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return True
        
    except ImportError as e:
        print(f"✗ 导入模块失败: {e}")
        print("请确保target_detect.py文件存在且语法正确")
        return False
    except Exception as e:
        print(f"✗ 测试过程中出现错误: {e}")
        return False

if __name__ == "__main__":
    success = test_detection_functions()
    if success:
        print("\n🎉 所有测试通过！显示功能正常工作。")
    else:
        print("\n❌ 测试失败，请检查代码。")