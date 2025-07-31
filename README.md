# 创建功能包
mkdir NUEDC_H
cd NUEDC_H/
ros2 pkg create --build-type ament_python automatic_aiming \--dependencies rclpy geometry_msgs nav_msgs sensor_msgs std_msgs \--description "NUEDC problem H"

# 编译
colcon build --symlink-install # 动态编译，不需要重新编译
# 隔离编译 --- 为了使每个包的编译文件处于同一文件夹
cd 
colcon build --merge-install
# # 终止所有 ros2 进程
# pkill -9 -f "ros2"

# 动态编译，不需要自己重新编译
colcon build --packages-select <package name> --symlink-install   # 不能同时用

# 删除编译文件
rm -rf build install log

# 查看视频设备
ls /dev/video*
# git 取消代理
# 取消 HTTP 代理设置
git config --global --unset http.proxy
# 取消 HTTPS 代理设置
git config --global --unset https.proxy

# 推送出错时改用ssh
# 更换远程地址为 SSH 协议
git remote set-url origin git@github.com:Kyhralus/NUEDC_H.git
# 推送代码
git push origin main 

# 启动节点 - 带日志记录
export RCUTILS_LOGGING_SEVERITY=INFO
export RCUTILS_LOGGING_USE_STDOUT=1
ros2 run automatic_aiming morning.py

# 启动完整系统 - 带详细日志
export RCUTILS_LOGGING_SEVERITY=DEBUG
ros2 launch automatic_aiming full_system.launch.py --log-level debug

# 查看特定节点的详细日志
colcon build --symlink-install
. install/setup.bash
ros2 run automatic_aiming camera_publisher
. install/setup.bash
ros2 run automatic_aiming target_detect 

ros2 run automatic_aiming main_controller 

ros2 run automatic_aiming laser_detect

## 日志配置说明
为了确保节点内部的日志信息正确保存，需要设置以下环境变量：

```bash`
# 设置日志级别为DEBUG以显示更多信息
export RCUTILS_LOGGING_SEVERITY=DEBUG
export RCUTILS_LOGGING_USE_STDOUT=1
export RCUTILS_LOGGING_BUFFERED_STREAM=1

# 启动时使用详细日志
ros2 launch automatic_aiming full_system.launch.py --log-level debug 2>&1 | tee system.log
```

## target_detection函数处理结果
- preprocess_image(): 图像预处理函数 - 显示高斯模糊、HSV转换、颜色阈值结果
- detect_contours(): 轮廓检测函数 - 显示检测到的轮廓数量、有效轮廓、最大面积、中心位置

# # NUEDC_H

帮我把检测圆的层级改掉，改成检测所有轮廓，不要层级，然后进行圆度检测，显示检测到的圆，记作candidate_circles,保留它们的中心和半径，然后选取其中最小面积的圆作为minist_circle,其中半径在90-120的圆作为一个数组，可能有多个圆，然后把这个数组的圆心和半径都取均值作为target_circle，用蓝色画出minist_circle以及它的圆心，用红色画出target_circle以及它的圆心。

    # 修改逻辑 【存在问题，检测出来不稳定】
    '''
    1. detect_nested_rects 换成 detect_border_rects
        处理逻辑：
        # 步骤1: 灰度转换
        # 步骤2: 双边滤波
        # 步骤3: 形态学闭运算
        # 步骤4: 边缘检测
        # ---- 前四步同之前
        # 找出所有外轮廓，并创建窗口显示
        # 筛选轮廓，取中心点靠近画面中心的最大轮廓，记为 candidate_contor 并画出，并打印出该轮廓的信息
        # 对 candidate_contor 进行多边形拟合，存储为 border_rect ，并画出轮廓角点和中心，并打印出该矩形的信息
        # 打印出函数运行时间
        # 返回 border_rect


    2. detect_deepest_inner_circle 换成 detect_ring
        处理逻辑：
        # 1. 图像预处理
        # 2. ROI处理，将图片 向内偏移 20 个像素
        # 3. 自适应阈值处理
        # 4. 形态学操作
        # 5.1 轮廓检测 --- 直接找出外轮廓，不需要层级关系，用黄色画出所有轮廓，
        # 5.2 检测出最大的轮廓打印出信息并记为 candidate_contor ，用绿色画出
        # 6. 对 candidate_contor 进行圆拟合，记为 target_cricle 并用粉色画出圆和圆心，打印出结果
        # 打印出函数运行时间
        # 返回最终目标圆
    3. 主函数：
        # 1. 读取图片
        # 2. 对图片进行 detect_border_rects 处理得到外边框矩形信息，记为 broder_rect
        # 2. 对 broder_rect 进行投影变换形成一个新的画面，并存储投影变换的转换矩阵，并resize为640x480，显示出来，记为 roi_region
        # 3. 对 roi_region 进行 detect_ring, 得到最终的圆环，记为 target_circle
        # 4. 对 target_circle 的圆形用之前得到的投影矩阵进行逆变换得到实际点 visual_target_point，打印出影矩阵和实际点 visual_target_point的信息
        # 4. 将 target_circle 画到 roi_region 上，并画出圆形，显示帧率，并用另一种颜色 画出visual_target_point
    '''

这个帧率的计算逻辑有问题，同时给我加个标志来管理激光的寻找，默认关闭，现在也关闭。在最终图像中显示出图像的中心，并显示target_center和图像中心的偏差值。另外裁剪图像改成1290X720，找到矩形后则变成矩形向外扩30像素，没找到就还是1280X720。



----------
## 节点介绍
Note: 
- [ ] is holding
- [/] is doing
- [X] is done

---

[X] ROS2环境配置与工作空间创建

[X] 摄像头驱动与图像采集节点开发

[/] 图像处理节点 多线程
    线程一：靶心，靶环识别
    - 1. 获得内外边框，获得中心点（粗略中心）
    - 2. 提取目标区域（外边框外扩20piexs），此后处理该区域，其他区域不做处理 --- resize 为 640X480 [@TODO 投影校正]
    - 3. 识别圆形靶环，获得最里面的圆心 --- 记作 target_center
    - 4. 提取出从内到外的第三个圆环 --- 记作 target_circle
    - 5. 选取 target_circle 构造 target_circle_table，得到圆的每个像素点坐标，第一个元素为最上方的像素点，顺时针存储其他点

    线程二：蓝紫色激光识别
    - 1. 识别蓝紫色激光，获得激光点的坐标
    - 2. 计算蓝色激光点和 target_center 的像素偏差，delta_x, delta_y, 发送"@0,delta_x,delta_y"到"uart1_sender_topic"
    - 3. 计算蓝色激光点和 target_circle_table 中依次每个像素点的偏差，delta_x, delta_y, 逐个发送"@1,delta_x,delta_y"到"uart1_sender_topic"

[X] main_controller节点
    读取和小车的通信指令，同时发送指令，对不同指令执行不同任务

[X] GUI界面节点
    用 pytk 写一个简易调参界面，完成对阈值的调整，任务的选择，任务结果的显示

@TODO
ekf_
依旧不会显示画面，继续更改，并每3s绘制并保存一次轨迹，以及得到的点和过滤以及预测点的图。

warp
用缓存机制 (cached_circle_mapping)，并且由于仿射变换后的圆基本固定，所以所有仿射变换后的圆采样的点的坐标基本固定，也即仿射变换后采样点的坐标固定，随着时间的变化只有仿射变换矩阵在变，因此我对采样点矩阵我可以在前几次就保存好，之后根据不同的仿射变换矩阵，直接根据开始的时间对应上采样点的坐标，再将该坐标进行逆仿射变换即可得到此刻原始图像上的点。