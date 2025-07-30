# 创建功能包
mkdir NUEDC_H
cd NUEDC_H/
ros2 pkg create --build-type ament_python automatic_aiming \--dependencies rclpy geometry_msgs nav_msgs sensor_msgs std_msgs \--description "NUEDC problem H"

# 编译
colcon build --symlink-install # 动态编译，不需要重新编译

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
    - 2. 提取目标区域（外边框外扩20piexs），此后处理该区域，其他区域不做处理 --- resize 为 60X480 [@TODO 投影校正]
    - 3. 识别圆形靶环，获得最里面的圆心 --- 记作 target_center
    - 4. 提取出从内到外的第三个圆环 --- 记作 target_circle
    - 5. 选取 target_circle 构造 target_circle_table，得到圆的每个像素点坐标，第一个元素为最上方的像素点，顺时针存储其他点

    线程二：蓝紫色激光识别
    - 1. 识别蓝紫色激光，获得激光点的坐标
    - 2. 计算蓝色激光点和 target_center 的像素偏差，delta_x, delta_y, 发送"@0,delta_x,delta_y"到"uart1_sender_topic"
    - 3. 计算蓝色激光点和 target_circle_table 中依次每个像素点的偏差，delta_x, delta_y, 逐个发送"@1,delta_x,delta_y"到"uart1_sender_topic"

[] main_controller节点
    读取和小车的通信指令，同时发送指令，对不同指令执行不同任务

[] GUI界面节点
    用 pytk 写一个简易调参界面，完成对阈值的调整，任务的选择，任务结果的显示

[]

# # NUEDC_H
