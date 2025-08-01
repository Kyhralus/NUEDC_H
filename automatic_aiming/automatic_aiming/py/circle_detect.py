import cv2
import numpy as np

def detect_deepest_inner_circle(frame):
    """检测层级最多的轮廓结构中最里层的圆心"""
    result_frame = frame.copy()
    
    # 图像预处理
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 使用大津法进行阈值处理
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 形态学操作优化轮廓
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # 获取轮廓和层级信息 (RETR_TREE 模式)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if hierarchy is None or len(hierarchy) == 0 or len(contours) == 0:
        return result_frame, None
    
    hierarchy = hierarchy[0]  # 获取层级数组
    
    # ===== 步骤1: 计算每个轮廓的层级深度 =====
    contour_levels = {}  # 存储每个轮廓的层级深度
    max_level = 0        # 最大层级深度
    
    for i in range(len(contours)):
        level = 0
        next_idx = i
        
        # 向上遍历直到最外层轮廓
        while hierarchy[next_idx][3] != -1:
            next_idx = hierarchy[next_idx][3]
            level += 1
            
        contour_levels[i] = level
        if level > max_level:
            max_level = level
    
    # 如果没有嵌套结构，直接返回
    if max_level == 0:
        return result_frame, None
    
    # ===== 步骤2: 找到所有最深层级的轮廓 =====
    deepest_contour_indices = [i for i, level in contour_levels.items() if level == max_level]
    
    # ===== 步骤3: 在最深层级的轮廓中筛选圆形 =====
    valid_circles = []
    
    for i in deepest_contour_indices:
        contour = contours[i]
        
        # 计算轮廓面积
        area = cv2.contourArea(contour)
        
        # 过滤小面积轮廓
        if area < 100:  # 可调整此阈值
            continue
        
        # 计算轮廓的圆形度
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
            
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # 只处理圆形度较高的轮廓
        if circularity < 0.8:  # 可调整此阈值
            continue
        
        # 计算圆心
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            valid_circles.append({
                'index': i,
                'contour': contour,
                'center': (cX, cY),
                'area': area,
                'circularity': circularity
            })
    
    # 如果没有找到符合条件的圆形，返回
    if not valid_circles:
        return result_frame, None
    
    # ===== 步骤4: 确定最终的最里层圆心 =====
    # 按面积从小到大排序（通常最里层的圆形面积最小）
    valid_circles.sort(key=lambda x: x['area'])
    
    # 选择面积最小的圆形作为最里层圆心
    innermost_circle = valid_circles[0]
    
    # ===== 步骤5: 可视化结果 =====
    # 绘制所有轮廓（用不同颜色区分层级）
    color_map = [
        (0, 0, 255),      # 红色 - 层级0 (最外层)
        (0, 255, 0),      # 绿色 - 层级1
        (255, 0, 0),      # 蓝色 - 层级2
        (0, 255, 255),    # 黄色 - 层级3
        (255, 0, 255),    # 紫色 - 层级4
        (255, 255, 0),    # 青色 - 层级5
    ]
    
    for i, contour in enumerate(contours):
        level = contour_levels[i]
        color = color_map[level % len(color_map)]
        cv2.drawContours(result_frame, [contour], -1, color, 1)
    
    # 用特殊颜色标记最里层的轮廓
    cv2.drawContours(result_frame, [innermost_circle['contour']], -1, (255, 255, 255), 3)
    
    # 标记最里层的圆心
    cX, cY = innermost_circle['center']
    cv2.circle(result_frame, (cX, cY), 5, (0, 255, 255), -1)
    cv2.putText(result_frame, f"Deepest Inner: ({cX}, {cY})", 
               (cX - 100, cY - 15), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    return result_frame, innermost_circle['center']

# 主函数示例
def main():
    # 打开摄像头或读取图像
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 处理图像
        result, center = detect_deepest_inner_circle(frame)
        
        # 显示结果
        cv2.imshow('Deepest Inner Circle Detection', result)
        
        # 打印最里层圆心坐标
        if center:
            print(f"最里层圆心坐标: {center}")
        
        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()