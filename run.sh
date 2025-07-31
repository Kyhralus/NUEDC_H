#!/bin/bash
# chmod +x /home/orangepi/ros2_workspace/NUEDC_H/run.sh

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
MAIN_LOG="${LOGS_DIR}/main_${TIMESTAMP}.log"

# 全局进程数组
declare -a ALL_PIDS=()

# 日志函数
log_info() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $1"
    echo "$msg" | tee -a "${MAIN_LOG}"
}

log_error() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $1"
    echo "$msg" | tee -a "${MAIN_LOG}"
}

# 清理函数
cleanup() {
    echo
    log_info "收到退出信号，正在关闭所有节点..."
    
    # 杀死所有子进程
    for pid in "${ALL_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            log_info "正在关闭 PID: $pid"
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    
    # 等待进程结束
    sleep 3
    
    # 强制杀死剩余进程
    for pid in "${ALL_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            log_info "强制关闭 PID: $pid"
            kill -KILL "$pid" 2>/dev/null || true
        fi
    done
    
    # 额外清理：杀死所有ROS2相关进程
    pkill -f "ros2 run automatic_aiming" 2>/dev/null || true
    
    log_info "所有节点已关闭"
    exit 0
}

# 设置信号处理
trap cleanup SIGINT SIGTERM EXIT

log_info "=== 自动瞄准系统启动脚本 ==="
log_info "工作空间: ${WORKSPACE_DIR}"
log_info "日志目录: ${LOGS_DIR}"

# 切换到工作空间目录
cd "${WORKSPACE_DIR}"

# 检查ROS2环境
log_info "检查ROS2环境..."
if ! command -v ros2 &> /dev/null; then
    log_error "ROS2未安装或未正确配置环境变量"
    exit 1
fi

# Source ROS2环境
log_info "加载ROS2环境..."
source /opt/ros/humble/setup.bash 2>/dev/null || true
source "${WORKSPACE_DIR}/install/setup.bash" 2>/dev/null || true

# 检查功能包是否已构建
if [ ! -d "${WORKSPACE_DIR}/install/automatic_aiming" ]; then
    log_info "功能包未构建，正在构建..."
    colcon build --packages-select automatic_aiming
    source "${WORKSPACE_DIR}/install/setup.bash"
fi

# 定义所有节点（注释掉uart0相关节点）
NODES=(
    "main_controller" 
    "target_detect"
     # "laser_tuner"
    "gimbal_controller"
    "uart1_receiver"
    "uart1_sender"
    "uart3_receiver"
    "uart3_sender"
    "speakr_node"
    # "uart0_receiver"  # 注释掉
    # "uart0_sender"    # 注释掉
)

# 启动所有节点
log_info "开始启动所有节点..."

for node in "${NODES[@]}"; do
    log_info "启动节点: ${node}"
    NODE_LOG="${LOGS_DIR}/${node}_${TIMESTAMP}.log"
    
    # 创建带颜色标识的日志输出函数
    start_node_with_logging() {
        local node_name="$1"
        local log_file="$2"
        
        # 为不同节点设置不同颜色
        case "$node_name" in
            
            "main_controller") color="\033[33m" ;;       # 黄色
            "target_detect") color="\033[31m" ;;         # 红色
            # "laser_tuner") color="\033[32m" ;;      # 绿色
            "gimbal_controller") color="\033[30m" ;;     # 橙色
            "uart1_receiver") color="\033[34m" ;;        # 蓝色
            "uart1_sender") color="\033[35m" ;;          # 紫色
            "uart3_receiver") color="\033[36m" ;;        # 青色
            "uart3_sender") color="\033[37m" ;;          # 白色
            "speakr_node") color="\033[90m" ;;           # 灰色
            *) color="\033[0m" ;;                        # 默认色
        esac
        
        # 启动节点并处理输出
        ros2 run automatic_aiming "$node_name" 2>&1 | while IFS= read -r line; do
            # 添加时间戳和节点名称
            timestamp=$(date '+%Y-%m-%d %H:%M:%S')
            formatted_line="[$timestamp] [$node_name] $line"
            
            # 同时输出到终端（带颜色）和日志文件（无颜色）
            echo -e "${color}${formatted_line}\033[0m"
            echo "$formatted_line" >> "$log_file"
            
            # 同时写入主日志
            echo "$formatted_line" >> "${MAIN_LOG}"
        done
    }
    
    # 后台启动节点
    start_node_with_logging "$node" "$NODE_LOG" &
    NODE_PID=$!
    ALL_PIDS+=($NODE_PID)
    
    log_info "节点 ${node} 已启动，PID: ${NODE_PID}，日志文件: ${NODE_LOG}"
    sleep 1.5
done

log_info "所有节点启动完成"
echo "=========================================="
echo "系统运行中... 按 Ctrl+C 停止所有节点"
echo "=========================================="

# 监控节点状态
while true; do
    active_count=0
    for pid in "${ALL_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            ((active_count++))
        fi
    done
    
    echo -ne "\r[$(date '+%H:%M:%S')] 活动节点数: $active_count/${#NODES[@]}"
    
    if [ $active_count -eq 0 ]; then
        echo
        log_error "所有节点已意外退出"
        break
    fi
    
    sleep 2
done

cleanup