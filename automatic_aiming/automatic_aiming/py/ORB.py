"""
    @TODO 1. 模板roi部分用于模板匹配：只将滤波二值化后的模板图像作为匹配模板
          2. 多张模板，进行联合匹配

"""

import cv2
import numpy as np

class RealTimeSIFTMatcher:
    def __init__(self):
        # 初始化摄像头
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("无法打开摄像头")
            
        # 获取摄像头分辨率
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 创建SIFT检测器
        self.sift = cv2.SIFT_create()
        
        # FLANN参数
        # 影响特征提取和匹配速度与质量，通常可保持默认，也可微调以提升速度或识别准确率
        FLANN_INDEX_KDTREE = 1
        self.index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5) 
        self.search_params = dict(checks=50)
        
        # 创建FLANN匹配器
        self.flann = cv2.FlannBasedMatcher(self.index_params, self.search_params)
        
        # 模板和匹配状态
        self.template = None
        self.kp_template = None
        self.des_template = None
        
        # 结果显示参数
        self.match_threshold = 0.7  # 比率测试阈值，设置匹配时的特征距离阈值，越小则匹配越严格
        
        
        print("按 'a' 键捕获模板图像，按 'q' 键退出")
        
    def capture_template(self, frame):
        """捕获模板图像并计算特征"""
        self.template = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.kp_template, self.des_template = self.sift.detectAndCompute(self.template, None)
        print(f"已捕获模板，检测到 {len(self.kp_template)} 个特征点")
        
    def match(self, frame):
        """执行模板匹配并返回结果图像"""
        # 创建固定布局：左侧模板，右侧实时画面
        result = np.zeros((self.height, self.width*2, 3), dtype=np.uint8)
        
        # 左侧显示模板图像（如果已捕获）
        if self.template is not None:
            result[:, :self.width] = cv2.cvtColor(self.template, cv2.COLOR_GRAY2BGR)
        else:
            cv2.putText(result[:, :self.width], "No Template", (self.width//4, self.height//2),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # 右侧显示实时画面
        result[:, self.width:] = frame.copy()
        
        # 如果没有模板，直接返回
        if self.template is None:
            cv2.putText(result[:, self.width:], "Press 'a' to capture template", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return result
            
        # 转换为灰度图进行特征检测
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 检测当前帧的特征点和描述符
        kp_frame, des_frame = self.sift.detectAndCompute(gray, None)
        
        if des_frame is None or len(des_frame) == 0:
            cv2.putText(result[:, self.width:], "Cannot detect features", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return result
            
        # 使用FLANN进行特征匹配
        matches = self.flann.knnMatch(self.des_template, des_frame, k=2)
        
        # 应用Lowe's比率测试
        good_matches = []
        for m, n in matches:
            if m.distance < self.match_threshold * n.distance:
                good_matches.append(m)
        
        # 显示匹配点数量
        match_text = f"Matches: {len(good_matches)}"
        
        # 如果匹配点太少，不进行后续处理
        if len(good_matches) < 4:  # 至少需要4个点计算单应性矩阵
            cv2.putText(result[:, self.width:], match_text, (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return result
            
        # 提取匹配点的坐标
        src_pts = np.float32([self.kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # 使用RANSAC计算单应性矩阵
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if H is not None:
            # 计算内点数量
            inliers = int(mask.sum())
            match_text += f", Inliers: {inliers}"
            
            # 如果内点太少，不绘制边界框
            if inliers < 4:
                cv2.putText(result[:, self.width:], match_text, (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                return result
                
            # 获取模板图像的四个角点
            h, w = self.template.shape
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            
            # 透视变换，将模板的角点映射到当前帧
            dst = cv2.perspectiveTransform(pts, H)
            
            # 在当前帧上绘制检测到的目标边界框
            cv2.polylines(result[:, self.width:], [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
            
            # 绘制匹配线（只绘制内点）
            matches_mask = mask.ravel().tolist()
            draw_params = dict(
                matchColor=(0, 255, 0),  # 绿色匹配线
                singlePointColor=(255, 0, 0),  # 红色单点
                matchesMask=matches_mask,  # 只绘制内点
                flags=0
            )
            
            # 在结果图像上绘制匹配线
            result = cv2.drawMatches(
                result[:, :self.width], self.kp_template, 
                result[:, self.width:], kp_frame, 
                good_matches, None, **draw_params
            )
            
            # 显示匹配信息（绿色表示成功）
            cv2.putText(result[:, self.width:], match_text, (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            # 未能计算单应性矩阵
            cv2.putText(result[:, self.width:], match_text + ", No homography found", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
        return result
            
    def run(self):
        """运行实时匹配程序"""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("无法获取帧")
                break
                
            # 执行匹配并获取结果
            result = self.match(frame)
            
            # 显示帮助信息
            help_text = "Press 'a': Capture Template | 'q': Exit"
            cv2.putText(result, help_text, (10, self.height-30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # 绘制中间分隔线
            cv2.line(result, (self.width, 0), (self.width, self.height), (255, 255, 255), 1)
            
            # 显示结果
            cv2.imshow('Real-time SIFT Template Matching', result)
            
            # 按键处理
            key = cv2.waitKey(1) & 0xFF
            if key == ord('a'):
                self.capture_template(frame)
            elif key == ord('q'):
                break
                
        # 释放资源
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    matcher = RealTimeSIFTMatcher()
    matcher.run()