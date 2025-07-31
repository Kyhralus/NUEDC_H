#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import time

class LaserDetectionTuner:
    """蓝紫色激光检测调参工具"""
    
    def __init__(self, camera_id=0):
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            print("无法打开摄像头")
            return
        
        # 设置摄像头分辨率
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        # 初始化HSV阈值（蓝紫色激光的默认值）
        self.hsv_lower = [117, 45, 159]  # [H_min, S_min, V_min]
        self.hsv_upper = [168, 255, 255]  # [H_max, S_max, V_max]
        
        # 形态学操作参数
        self.erode_iterations = 1
        self.dilate_iterations = 2
        self.kernel_size = 3
        
        # 轮廓筛选参数
        self.min_area = 50
        self.max_area = 1500
        
        # 创建窗口和滑条
        self.setup_trackbars()
        
        print("激光检测调参工具已启动")
        print("按 'q' 退出")
        print("按 's' 保存当前参数")
        print("按 'r' 重置为默认参数")
    
    def setup_trackbars(self):
        """设置滑条界面"""
        # 创建主窗口
        cv2.namedWindow('Laser Detection Tuner', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('HSV Mask', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('Morphology Result', cv2.WINDOW_AUTOSIZE)
        
        # HSV阈值滑条
        cv2.createTrackbar('H_min', 'Laser Detection Tuner', self.hsv_lower[0], 179, self.update_hsv_lower_h)
        cv2.createTrackbar('S_min', 'Laser Detection Tuner', self.hsv_lower[1], 255, self.update_hsv_lower_s)
        cv2.createTrackbar('V_min', 'Laser Detection Tuner', self.hsv_lower[2], 255, self.update_hsv_lower_v)
        cv2.createTrackbar('H_max', 'Laser Detection Tuner', self.hsv_upper[0], 179, self.update_hsv_upper_h)
        cv2.createTrackbar('S_max', 'Laser Detection Tuner', self.hsv_upper[1], 255, self.update_hsv_upper_s)
        cv2.createTrackbar('V_max', 'Laser Detection Tuner', self.hsv_upper[2], 255, self.update_hsv_upper_v)
        
        # 形态学操作滑条
        cv2.createTrackbar('Kernel_Size', 'Laser Detection Tuner', self.kernel_size, 10, self.update_kernel_size)
        cv2.createTrackbar('Erode_Iter', 'Laser Detection Tuner', self.erode_iterations, 5, self.update_erode_iterations)
        cv2.createTrackbar('Dilate_Iter', 'Laser Detection Tuner', self.dilate_iterations, 5, self.update_dilate_iterations)
        
        # 轮廓筛选滑条
        cv2.createTrackbar('Min_Area', 'Laser Detection Tuner', self.min_area, 500, self.update_min_area)
        cv2.createTrackbar('Max_Area', 'Laser Detection Tuner', self.max_area, 3000, self.update_max_area)
    
    # 滑条回调函数
    def update_hsv_lower_h(self, val):
        self.hsv_lower[0] = val
    
    def update_hsv_lower_s(self, val):
        self.hsv_lower[1] = val
    
    def update_hsv_lower_v(self, val):
        self.hsv_lower[2] = val
    
    def update_hsv_upper_h(self, val):
        self.hsv_upper[0] = val
    
    def update_hsv_upper_s(self, val):
        self.hsv_upper[1] = val
    
    def update_hsv_upper_v(self, val):
        self.hsv_upper[2] = val
    
    def update_kernel_size(self, val):
        self.kernel_size = max(1, val)
    
    def update_erode_iterations(self, val):
        self.erode_iterations = val
    
    def update_dilate_iterations(self, val):
        self.dilate_iterations = val
    
    def update_min_area(self, val):
        self.min_area = val
    
    def update_max_area(self, val):
        self.max_area = val
    
    def crop_center_image(self, image, target_width=640, target_height=480):
        """从图像中心裁剪指定尺寸的区域"""
        h, w = image.shape[:2]
        
        # 计算裁剪区域的起始点
        start_x = max(0, (w - target_width) // 2)
        start_y = max(0, (h - target_height) // 2)
        
        # 计算实际裁剪尺寸（防止超出边界）
        actual_width = min(target_width, w - start_x)
        actual_height = min(target_height, h - start_y)
        
        # 裁剪图像
        cropped = image[start_y:start_y + actual_height, start_x:start_x + actual_width]
        
        # 如果裁剪后的尺寸不足目标尺寸，进行填充或缩放
        if cropped.shape[:2] != (target_height, target_width):
            cropped = cv2.resize(cropped, (target_width, target_height))
        
        return cropped
    
    def detect_blue_purple_laser(self, frame, target_point=None):
        """检测蓝紫色激光点"""
        # 转换为HSV色彩空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]
        reference_point = target_point if target_point is not None else (w // 2, h // 2)
        
        # 创建HSV掩码
        lower_hsv = np.array(self.hsv_lower)
        upper_hsv = np.array(self.hsv_upper)
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        
        # 形态学操作
        if self.kernel_size > 0:
            kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
            if self.erode_iterations > 0:
                mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel, iterations=self.erode_iterations)
            if self.dilate_iterations > 0:
                mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=self.dilate_iterations)
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 候选激光点列表
        laser_candidates = []
        result_frame = frame.copy()
        
        # 筛选符合条件的轮廓
        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area:
                continue
            
            # 计算轮廓中心
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            x = int(M["m10"] / M["m00"])
            y = int(M["m01"] / M["m00"])
            
            # 计算距离基准点的距离
            dx = x - reference_point[0]
            dy = y - reference_point[1]
            distance = np.sqrt(dx**2 + dy**2)
            
            # 归一化距离
            max_possible_distance = np.sqrt((w//2)**2 + (h//2)**2)
            normalized_distance = distance / max_possible_distance
            
            # 计算评分
            area_score = area / self.max_area
            distance_score = 1 - normalized_distance
            score = 0.3 * area_score + 0.7 * distance_score
            
            laser_candidates.append({
                'center': (x, y),
                'area': area,
                'distance': distance,
                'score': score,
                'contour': cnt
            })
            
            # 绘制所有候选点（绿色）
            cv2.circle(result_frame, (x, y), 8, (0, 255, 0), 2)
            cv2.circle(result_frame, (x, y), 2, (0, 255, 0), -1)
            cv2.putText(result_frame, f"#{i} A:{area:.0f} S:{score:.2f}", 
                       (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # 找到最佳激光点
        best_laser = None
        if laser_candidates:
            best_candidate = max(laser_candidates, key=lambda c: c['score'])
            best_laser = best_candidate['center']
            
            # 绘制最佳激光点（红色）
            cv2.circle(result_frame, best_laser, 10, (0, 0, 255), 3)
            cv2.circle(result_frame, best_laser, 3, (0, 0, 255), -1)
            cv2.putText(result_frame, f"BEST: ({best_laser[0]}, {best_laser[1]})", 
                       (best_laser[0]+15, best_laser[1]+15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # 如果有参考点，画连接线
            if target_point:
                cv2.line(result_frame, target_point, best_laser, (255, 255, 0), 2)
                err_x = best_laser[0] - target_point[0]
                err_y = best_laser[1] - target_point[1]
                cv2.putText(result_frame, f"Error: ({err_x}, {err_y})", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # 绘制参考点
        cv2.circle(result_frame, reference_point, 5, (255, 255, 255), 2)
        cv2.putText(result_frame, "REF", (reference_point[0]+10, reference_point[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return best_laser, mask, result_frame, len(laser_candidates)
    
    def save_parameters(self):
        """保存当前参数到文件"""
        filename = f"laser_params_{int(time.time())}.txt"
        with open(filename, 'w') as f:
            f.write("# 蓝紫色激光检测参数\n")
            f.write(f"HSV_LOWER = [{self.hsv_lower[0]}, {self.hsv_lower[1]}, {self.hsv_lower[2]}]\n")
            f.write(f"HSV_UPPER = [{self.hsv_upper[0]}, {self.hsv_upper[1]}, {self.hsv_upper[2]}]\n")
            f.write(f"KERNEL_SIZE = {self.kernel_size}\n")
            f.write(f"ERODE_ITERATIONS = {self.erode_iterations}\n")
            f.write(f"DILATE_ITERATIONS = {self.dilate_iterations}\n")
            f.write(f"MIN_AREA = {self.min_area}\n")
            f.write(f"MAX_AREA = {self.max_area}\n")
            f.write("\n# 代码格式:\n")
            f.write(f"lower_hsv = np.array([{self.hsv_lower[0]}, {self.hsv_lower[1]}, {self.hsv_lower[2]}])\n")
            f.write(f"upper_hsv = np.array([{self.hsv_upper[0]}, {self.hsv_upper[1]}, {self.hsv_upper[2]}])\n")
        print(f"参数已保存到: {filename}")
    
    def reset_parameters(self):
        """重置为默认参数"""
        self.hsv_lower = [117, 45, 159]
        self.hsv_upper = [168, 255, 255]
        self.kernel_size = 3
        self.erode_iterations = 1
        self.dilate_iterations = 2
        self.min_area = 50
        self.max_area = 1500
        
        # 更新滑条位置
        cv2.setTrackbarPos('H_min', 'Laser Detection Tuner', self.hsv_lower[0])
        cv2.setTrackbarPos('S_min', 'Laser Detection Tuner', self.hsv_lower[1])
        cv2.setTrackbarPos('V_min', 'Laser Detection Tuner', self.hsv_lower[2])
        cv2.setTrackbarPos('H_max', 'Laser Detection Tuner', self.hsv_upper[0])
        cv2.setTrackbarPos('S_max', 'Laser Detection Tuner', self.hsv_upper[1])
        cv2.setTrackbarPos('V_max', 'Laser Detection Tuner', self.hsv_upper[2])
        cv2.setTrackbarPos('Kernel_Size', 'Laser Detection Tuner', self.kernel_size)
        cv2.setTrackbarPos('Erode_Iter', 'Laser Detection Tuner', self.erode_iterations)
        cv2.setTrackbarPos('Dilate_Iter', 'Laser Detection Tuner', self.dilate_iterations)
        cv2.setTrackbarPos('Min_Area', 'Laser Detection Tuner', self.min_area)
        cv2.setTrackbarPos('Max_Area', 'Laser Detection Tuner', self.max_area)
        
        print("参数已重置为默认值")
    
    def run(self):
        """主运行循环"""
        fps_counter = 0
        fps_start_time = time.time()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("无法读取摄像头画面")
                break
            
            # 裁剪到640x480
            cropped_frame = self.crop_center_image(frame, 640, 480)
            
            # 检测激光
            laser_point, mask, result_frame, candidate_count = self.detect_blue_purple_laser(cropped_frame)
            
            # 计算FPS
            fps_counter += 1
            if fps_counter % 30 == 0:
                fps = 30 / (time.time() - fps_start_time)
                fps_start_time = time.time()
            else:
                fps = 0
            
            # 添加信息显示
            if fps > 0:
                cv2.putText(result_frame, f"FPS: {fps:.1f}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.putText(result_frame, f"Candidates: {candidate_count}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 添加参数显示
            param_text = f"HSV: [{self.hsv_lower[0]}-{self.hsv_upper[0]}, {self.hsv_lower[1]}-{self.hsv_upper[1]}, {self.hsv_lower[2]}-{self.hsv_upper[2]}]"
            cv2.putText(result_frame, param_text, (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            area_text = f"Area: [{self.min_area}-{self.max_area}]"
            cv2.putText(result_frame, area_text, (10, 140), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # 显示结果
            cv2.imshow('Laser Detection Tuner', result_frame)
            cv2.imshow('HSV Mask', mask)
            
            # 形态学处理结果
            if self.kernel_size > 0:
                hsv = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2HSV)
                lower_hsv = np.array(self.hsv_lower)
                upper_hsv = np.array(self.hsv_upper)
                temp_mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
                kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
                morphology_result = cv2.morphologyEx(temp_mask, cv2.MORPH_OPEN, kernel)
                cv2.imshow('Morphology Result', morphology_result)
            
            # 按键处理
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_parameters()
            elif key == ord('r'):
                self.reset_parameters()
            elif key == ord('h'):
                print("\n=== 帮助信息 ===")
                print("q: 退出程序")
                print("s: 保存当前参数")
                print("r: 重置为默认参数")
                print("h: 显示帮助信息")
                print("==================\n")
        
        self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        print("程序已退出")

def main():
    """主函数"""
    print("=== 蓝紫色激光检测调参工具 ===")
    print("使用滑条调整参数，实时查看检测效果")
    print("按 'h' 查看帮助信息")
    print("===============================")
    
    tuner = LaserDetectionTuner()
    tuner.run()

if __name__ == "__main__":
    main()
