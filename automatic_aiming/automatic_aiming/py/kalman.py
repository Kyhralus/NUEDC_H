# import cv2
# import numpy as np

# # 初始化视频捕捉
# cap = cv2.VideoCapture(0)

# # 初始化卡尔曼滤波器：跟踪位置(x, y)和速度(dx, dy)
# kalman = cv2.KalmanFilter(4, 2)
# dt = 1  # 时间间隔

# # 状态转移矩阵
# kalman.transitionMatrix = np.array([
#     [1, 0, dt, 0],
#     [0, 1, 0, dt],
#     [0, 0, 1,  0],
#     [0, 0, 0,  1]
# ], dtype=np.float32)

# # 观测矩阵：仅观测中心坐标(x, y)
# kalman.measurementMatrix = np.array([
#     [1, 0, 0, 0],
#     [0, 1, 0, 0]
# ], dtype=np.float32)

# # 初始化协方差矩阵
# kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
# kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
# kalman.errorCovPost = np.eye(4, dtype=np.float32)

# # 初始状态
# kalman.statePost = np.zeros((4, 1), dtype=np.float32)

# # 存储最近一次有效观测的框大小
# last_w, last_h = 0, 0

# # 根据提供的阈值设置HSV范围（绿色系）
# lower_hsv = np.array([66, 43, 46])    # H:66, S:142, V:52
# upper_hsv = np.array([84, 255, 255])   # H:84, S:219, V:111

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # 转为HSV颜色空间
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#     # 根据给定阈值创建单一路径的掩码（绿色）
#     mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

#     # 形态学操作去除噪声
#     kernel = np.ones((5, 5), np.uint8)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # 填充内部空洞
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # 去除小噪声点

#     # 提取轮廓
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     measured_center = None

#     if contours:
#         # 找最大轮廓（过滤小目标）
#         largest = max(contours, key=cv2.contourArea)
#         area = cv2.contourArea(largest)
#         if area > 500:  # 面积阈值，可根据目标大小调整
#             x, y, w, h = cv2.boundingRect(largest)
#             # 更新最近一次有效框的大小
#             last_w, last_h = w, h
#             # 计算中心坐标作为观测值
#             center_x = x + w/2
#             center_y = y + h/2
#             measured_center = np.array([[center_x], [center_y]], dtype=np.float32)

#             # 画观测框（绿色）
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#             cv2.putText(frame, "Observed", (x, y - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     # 若观测值有效，则校正卡尔曼滤波器
#     if measured_center is not None:
#         kalman.correct(measured_center)

#     # 预测位置
#     prediction = kalman.predict()
#     pred_x_center = prediction[0][0]
#     pred_y_center = prediction[1][0]

#     # 绘制预测框（黄色）：使用最近一次有效观测的框大小
#     if last_w > 0 and last_h > 0:
#         # 计算预测框左上角坐标
#         pred_x = int(pred_x_center - last_w / 2)
#         pred_y = int(pred_y_center - last_h / 2)
#         # 确保预测框在画面内
#         pred_x = max(0, min(pred_x, frame.shape[1] - last_w))
#         pred_y = max(0, min(pred_y, frame.shape[0] - last_h))
#         # 绘制预测框
#         cv2.rectangle(frame, (pred_x, pred_y), 
#                      (pred_x + last_w, pred_y + last_h), (0, 255, 255), 2)
#         cv2.putText(frame, "Predicted", (pred_x, pred_y - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
#     else:
#         # 尚未检测到有效目标时显示提示
#         cv2.putText(frame, "No target detected", (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#     # 显示当前使用的HSV阈值范围
#     cv2.putText(frame, f"HSV: H[{66}-{84}], S[{142}-{219}], V[{52}-{111}]", 
#                 (10, frame.shape[0] - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

#     cv2.imshow('Green Object Tracking', frame)
#     # 按'q'键退出
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()



import cv2
import numpy as np
import simdkalman
import time

# 设置最大历史轨迹点数（虽然不绘制轨迹，但仍用于限制存储）
MAX_HISTORY_LENGTH = 100  # 可根据需要调整

# ----------------------
# 1. 初始化 Kalman 滤波器
# ----------------------
STATE_DIM = 4  # 状态维度（x, y, 速度x, 速度y）
OBS_DIM = 2    # 观测维度（x, y 坐标）

# 状态转移矩阵 A：假设匀速运动模型
state_transition = np.array([
    [1, 0, 1, 0],  # x_new = x + vx*dt（dt=1）
    [0, 1, 0, 1],  # y_new = y + vy*dt
    [0, 0, 1, 0],  # vx 保持不变
    [0, 0, 0, 1]   # vy 保持不变
])

# 过程噪声协方差 Q
process_noise = np.diag([0.1, 0.1, 0.5, 0.5])

# 观测模型 H
observation_model = np.array([
    [1, 0, 0, 0],  # 观测 x 位置
    [0, 1, 0, 0]   # 观测 y 位置
])

# 观测噪声协方差 R
observation_noise = np.diag([2.0, 2.0])

# 初始化滤波器
kf = simdkalman.KalmanFilter(
    state_transition=state_transition,
    process_noise=process_noise,
    observation_model=observation_model,
    observation_noise=observation_noise
)

# ----------------------
# 2. 色块检测函数（返回中心坐标和外接矩形）
# ----------------------
def detect_color_block(frame, lower, upper):
    """从图像中检测色块，返回中心坐标 (x, y) 和外接矩形 (x, y, w, h)，未检测到则返回 None"""
    # 转换到 HSV 色彩空间
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # 提取颜色掩码
    mask = cv2.inRange(hsv, lower, upper)
    # 腐蚀和膨胀去除噪声
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    # 查找轮廓
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours :
        return None  # 未检测到色块
    
    # 取最大轮廓（假设目标是最大的色块）
    max_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(max_contour) < 100:
        return None  # 未检测到色块
    # 计算中心矩
    M = cv2.moments(max_contour)
    if M["m00"] == 0:
        return None
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    
    # 计算外接矩形
    x, y, w, h = cv2.boundingRect(max_contour)
    return (cX, cY, x, y, w, h)

# ----------------------
# 3. 辅助函数：根据中心和原始框计算平滑/预测框
# ----------------------
def get_adjusted_rect(center, original_rect):
    """根据新中心和原始框的宽高计算调整后的矩形"""
    orig_x, orig_y, orig_w, orig_h = original_rect
    # 保持原始宽高，仅调整中心位置
    new_x = int(center[0] - orig_w / 2)
    new_y = int(center[1] - orig_h / 2)
    return (new_x, new_y, orig_w, orig_h)

# ----------------------
# 4. 视频处理与追踪主逻辑
# ----------------------
def track_color_block(video_path, lower_color, upper_color):
    # 存储历史位置（仅用于限制存储，不绘制）
    true_positions = []
    smoothed_positions = []
    predicted_positions = []
    # 存储上一帧的边界框信息，用于预测和平滑框的尺寸参考
    last_rect = None

    # 读取视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频文件")
        return

    # 初始化滤波器状态
    data = np.empty((1, 0, OBS_DIM))  # (n_sequences, n_timesteps, n_obs_dims)
    
    # 帧率计算相关变量
    fps = 0
    frame_count = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # 视频结束

        # 帧率计算
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 0:
            fps = frame_count / elapsed_time

        # 检测色块位置和轮廓
        detection = detect_color_block(frame, lower_color, upper_color)
        current_rect = None
        if detection is None:
            # 未检测到目标，用 NaN 表示缺失值
            obs = np.array([[np.nan, np.nan]])
            x, y, w, h = 0, 0, 0, 0  # 无效矩形
        else:
            cX, cY, x, y, w, h = detection
            obs = np.array([[cX, cY]])
            # 限制历史数据长度
            true_positions.append((cX, cY))
            if len(true_positions) > MAX_HISTORY_LENGTH:
                true_positions.pop(0)  # 移除最旧的点
            current_rect = (x, y, w, h)
            last_rect = current_rect  # 更新上一帧矩形信息
            # 绘制原始检测框（红色）
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # 标记原始检测中心
            cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)
            cv2.putText(frame, "Raw", (cX + 10, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 更新数据矩阵
        data = np.concatenate([data, obs[np.newaxis, ...]], axis=1)
        # 限制数据矩阵长度，避免内存占用过大
        if data.shape[1] > MAX_HISTORY_LENGTH:
            data = data[:, -MAX_HISTORY_LENGTH:, :]

        # 平滑当前及历史数据
        smoothed = kf.smooth(
            data,
            initial_value=[0, 0, 0, 0],
            initial_covariance=np.eye(STATE_DIM) * 100
        )

        # 获取当前平滑后的位置并绘制平滑框
        current_smoothed = smoothed.states.mean[0, -1, :2]
        # 限制平滑数据长度
        smoothed_positions.append(current_smoothed)
        if len(smoothed_positions) > MAX_HISTORY_LENGTH:
            smoothed_positions.pop(0)
        # 标记平滑后的位置（绿色）
        cv2.circle(frame, (int(current_smoothed[0]), int(current_smoothed[1])), 5, (0, 255, 0), -1)
        cv2.putText(frame, "Smoothed", (int(current_smoothed[0]) + 10, int(current_smoothed[1])), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # 绘制平滑框（绿色）
        if last_rect is not None:
            smoothed_rect = get_adjusted_rect(current_smoothed, last_rect)
            s_x, s_y, s_w, s_h = smoothed_rect
            cv2.rectangle(frame, (s_x, s_y), (s_x + s_w, s_y + s_h), (0, 255, 0), 2)

        # 预测下一步位置并绘制预测框
        predicted_pos = None
        if data.shape[1] > 1:
            predicted = kf.predict(data, n_test=1)
            predicted_pos = predicted.states.mean[0, 0, :2]
            # 限制预测数据长度
            predicted_positions.append(predicted_pos)
            if len(predicted_positions) > MAX_HISTORY_LENGTH:
                predicted_positions.pop(0)
            # 标记预测位置（蓝色）
            cv2.circle(frame, (int(predicted_pos[0]), int(predicted_pos[1])), 5, (255, 0, 0), -1)
            cv2.putText(frame, "Predicted", (int(predicted_pos[0]) + 10, int(predicted_pos[1])), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            # 绘制预测框（蓝色）
            if last_rect is not None:
                pred_rect = get_adjusted_rect(predicted_pos, last_rect)
                p_x, p_y, p_w, p_h = pred_rect
                cv2.rectangle(frame, (p_x, p_y), (p_x + p_w, p_y + p_h), (255, 0, 0), 2)

        # 显示帧率和状态信息
        cv2.putText(frame, f"FPS: {fps:.1f}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, "Red: Raw | Green: Smoothed | Blue: Predicted", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 显示图像
        cv2.imshow("Color Block Tracking", frame)

        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ----------------------
# 5. 运行追踪（使用绿色系HSV阈值）
# ----------------------
if __name__ == "__main__":
    # 绿色色块的 HSV 范围
    lower_green = np.array([66, 43, 46])
    upper_green = np.array([84, 255, 255])
    # 替换为视频路径或使用 0 调用摄像头
    track_color_block(0, lower_green, upper_green)  # 0 表示默认摄像头