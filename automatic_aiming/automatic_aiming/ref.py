"""优化的仿射变换圆形检测"""
        circle_start = time.time()
        
        if inner_rect is None:
            circle_time = (time.time() - circle_start) * 1000
            self.timing_stats['circle_detection'].append(circle_time)
            return None, None, None
        
        # 计算仿射变换矩阵
        affine_matrix, inverse_matrix = self.compute_affine_transform(inner_rect)
        if affine_matrix is None:
            circle_time = (time.time() - circle_start) * 1000
            self.timing_stats['circle_detection'].append(circle_time)
            return None, None, None
        
        # 保存变换矩阵
        self.affine_matrix = affine_matrix
        self.inverse_affine_matrix = inverse_matrix
        
        # 应用仿射变换
        warped_image = cv2.warpPerspective(frame, affine_matrix, (640, 480))
        
        # 优化的图像处理流程
        gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
        
        # 使用高斯模糊减少噪声
        blurred = cv2.GaussianBlur(gray, (7, 7), 2.0)
        
        # 改进的阈值处理 - 使用Otsu自动阈值
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 优化的形态学操作 - 修复断续轮廓
        # 1. 先用较大的椭圆核进行闭运算连接断续的轮廓
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)
        
        # 2. 再用小核去除噪声
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
        
        # 3. 最后用中等核再次闭运算确保轮廓完整
        kernel_final = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        final_processed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_final)
        
        # 轮廓检测
        contours, _ = cv2.findContours(final_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 创建显示图像
        circle_detection_display = cv2.cvtColor(final_processed, cv2.COLOR_GRAY2BGR)
        
        if not contours:
            circle_time = (time.time() - circle_start) * 1000
            self.timing_stats['circle_detection'].append(circle_time)
            return None, None, circle_detection_display
        
        # 优化的候选圆筛选
        candidate_circles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 150 or area > 50000:  # 调整面积阈值
                continue
            
            # 计算轮廓的凸度和圆度
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:
                continue
            
            solidity = area / hull_area
            if solidity < 0.7:  # 凸度阈值
                continue
            
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.5:  # 放宽圆度阈值
                continue
            
            # 使用最小外接圆
            (x, y), radius = cv2.minEnclosingCircle(contour)
            if radius < 20 or radius > 200:  # 调整半径阈值
                continue
            
            # 计算轮廓与最小外接圆的匹配度
            circle_area = np.pi * radius * radius
            area_ratio = area / circle_area
            if area_ratio < 0.6:  # 面积比阈值
                continue
            
            candidate_circles.append([int(x), int(y), int(radius), int(area), circularity])
            
            # 在显示图像上绘制候选圆 - 确保坐标为整数
            center_x, center_y = int(x), int(y)
            cv2.circle(circle_detection_display, (center_x, center_y), int(radius), (0, 255, 255), 2)
            
            # 修复putText坐标类型错误
            text_x = max(0, center_x - 20)
            text_y = max(15, center_y + int(radius) + 15)
            cv2.putText(circle_detection_display, f'C:{circularity:.2f}', 
                       (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        if not candidate_circles:
            circle_time = (time.time() - circle_start) * 1000
            self.timing_stats['circle_detection'].append(circle_time)
            return None, None, circle_detection_display
        
        candidate_circles = np.array(candidate_circles)
        
        # 按圆度和面积综合评分排序
        scores = candidate_circles[:, 4] * 0.7 + (candidate_circles[:, 3] / np.max(candidate_circles[:, 3])) * 0.3
        best_idx = np.argmax(scores)
        
        # 选择最佳圆作为靶心
        best_circle = candidate_circles[best_idx]
        innerest_x, innerest_y, innerest_r = int(best_circle[0]), int(best_circle[1]), int(best_circle[2])
        
        # 在仿射变换后的图像坐标系中的目标中心
        target_center_warped = (innerest_x, innerest_y)
        
        # 在显示图像上绘制靶心 - 确保坐标为整数
        cv2.circle(circle_detection_display, target_center_warped, 8, (0, 0, 255), -1)
        cv2.circle(circle_detection_display, target_center_warped, innerest_r, (0, 0, 255), 3)
        
        # 修复putText坐标类型错误
        text_x = max(0, innerest_x + 15)
        text_y = max(15, innerest_y - 15)
        cv2.putText(circle_detection_display, "Target Center", 
                   (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 筛选目标圆（根据半径范围）
        target_circles = candidate_circles[(candidate_circles[:, 2] >= 60) & (candidate_circles[:, 2] <= 140)]
        
        # 最终目标圆
        target_circle_warped = None
        if len(target_circles) == 1:
            target_circle_warped = (int(target_circles[0][0]), int(target_circles[0][1]), int(target_circles[0][2]))
        elif len(target_circles) > 1:
            # 选择圆度最好的
            best_target_idx = np.argmax(target_circles[:, 4])
            target_circle_warped = (int(target_circles[best_target_idx][0]), 
                                  int(target_circles[best_target_idx][1]), 
                                  int(target_circles[best_target_idx][2]))
        else:
            # 如果没有合适的目标圆，使用最佳圆
            target_circle_warped = (innerest_x, innerest_y, innerest_r)
        
        # 在显示图像上绘制目标圆
        if target_circle_warped:
            tc_x, tc_y, tc_r = target_circle_warped
            cv2.circle(circle_detection_display, (tc_x, tc_y), tc_r, (0, 255, 0), 3)
            
            # 修复putText坐标类型错误
            text_x = max(0, tc_x - 50)
            text_y = max(15, tc_y - tc_r - 15)
            cv2.putText(circle_detection_display, f"Target R={tc_r}", 
                       (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 逆变换到原图坐标系
        target_center = None
        target_circle = None
        
        if target_center_warped and self.inverse_affine_matrix is not None:
            # 逆变换靶心坐标
            warped_point = np.array([[[float(target_center_warped[0]), float(target_center_warped[1])]]], dtype=np.float32)
            original_point = cv2.perspectiveTransform(warped_point, self.inverse_affine_matrix)
            target_center = (int(original_point[0][0][0]), int(original_point[0][0][1]))
        
        if target_circle_warped and self.inverse_affine_matrix is not None:
            # 逆变换目标圆
            tc_x, tc_y, tc_r = target_circle_warped
            warped_circle_point = np.array([[[float(tc_x), float(tc_y)]]], dtype=np.float32)
            original_circle_point = cv2.perspectiveTransform(warped_circle_point, self.inverse_affine_matrix)
            
            # 计算逆变换后的半径（近似）
            warped_radius_point = np.array([[[float(tc_x + tc_r), float(tc_y)]]], dtype=np.float32)
            original_radius_point = cv2.perspectiveTransform(warped_radius_point, self.inverse_affine_matrix)
            
            original_radius = int(np.linalg.norm(
                original_radius_point[0][0] - original_circle_point[0][0]
            ))
            
            target_circle = (
                int(original_circle_point[0][0][0]), 
                int(original_circle_point[0][0][1]), 
                original_radius
            )
        
        circle_time = (time.time() - circle_start) * 1000
        self.timing_stats['circle_detection'].append(circle_time)
        
        return target_center, target_circle, circle_detection_display