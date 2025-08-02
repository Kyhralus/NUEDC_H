#!/bin/bash

# 设置脚本在出错时退出
set -e

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="${SCRIPT_DIR}"
LOGS_DIR="${WORKSPACE_DIR}/logs"

# 创建日志目录
mkdir -p "${LOGS_DIR}"

# 清理旧日志文件
echo "清理旧日志文件..."
rm -f "${LOGS_DIR}"/*.log

# 获取当前时间戳
TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')
LAUNCH_LOG="${LOGS_DIR}/launch_${TIMESTAMP}.log"

echo "=== 使用ROS2 Launch启动自动瞄准系统 ==="
echo "工作空间: ${WORKSPACE_DIR}"
echo "日志目录: ${LOGS_DIR}"

# 切换到工作空间目录
cd "${WORKSPACE_DIR}"

# Source ROS2环境
echo "加载ROS2环境..."
source /opt/ros/humble/setup.bash
source "${WORKSPACE_DIR}/install/setup.bash"

# 检查功能包是否已构建
if [ ! -d "${WORKSPACE_DIR}/install/automatic_aiming" ]; then
    echo "功能包未构建，正在构建..."
    colcon build --packages-select automatic_aiming
    source "${WORKSPACE_DIR}/install/setup.bash"
fi

# 日志函数
log_info() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] [LAUNCH] $1"
    echo "$msg" | tee -a "${LAUNCH_LOG}"
}

log_error() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $1"
    echo "$msg" | tee -a "${LAUNCH_LOG}"
}

# 清理函数
cleanup() {
    echo
    log_info "收到退出信号，正在关闭launch进程..."
    # 杀死所有ROS2相关进程
    pkill -f "ros2 launch automatic_aiming" 2>/dev/null || true
    pkill -f "ros2 run automatic_aiming" 2>/dev/null || true
    log_info "Launch进程已关闭"
    exit 0
}

# 设置信号处理
trap cleanup SIGINT SIGTERM EXIT

log_info "启动所有节点..."
log_info "日志将保存到: ${LAUNCH_LOG}"
log_info "按 Ctrl+C 停止所有节点"

# 使用ROS2 launch启动所有节点，增强日志输出
ros2 launch automatic_aiming all_nodes.launch.py 2>&1 | while IFS= read -r line; do
    # 添加时间戳
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    formatted_line="[$timestamp] $line"
    
    # 根据日志级别设置颜色
    if [[ "$line" == *"[INFO]"* ]]; then
        echo -e "\033[32m${formatted_line}\033[0m"  # 绿色
    elif [[ "$line" == *"[WARN]"* ]]; then
        echo -e "\033[33m${formatted_line}\033[0m"  # 黄色
    elif [[ "$line" == *"[ERROR]"* ]]; then
        echo -e "\033[31m${formatted_line}\033[0m"  # 红色
    elif [[ "$line" == *"[DEBUG]"* ]]; then
        echo -e "\033[36m${formatted_line}\033[0m"  # 青色
    else
        echo "$formatted_line"  # 默认颜色
    fi
    
    # 写入日志文件
    echo "$formatted_line" >> "${LAUNCH_LOG}"
done
