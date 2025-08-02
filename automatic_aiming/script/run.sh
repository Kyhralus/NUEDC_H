#!/bin/bash
# 确保脚本具有可执行权限：chmod +x /home/orangepi/ros2_workspace/NUEDC_H/automatic_aiming/script/run.sh

# 获取脚本所在目录（script目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 工作空间目录：脚本目录的上一级（即automatic_aiming功能包目录）
WORKSPACE_DIR="$(dirname "${SCRIPT_DIR}")"
LOGS_DIR="${WORKSPACE_DIR}/logs"  # 日志目录放在工作空间下

# 创建日志目录
mkdir -p "${LOGS_DIR}"

# 清理旧日志文件
echo "清理旧日志文件..."
rm -f "${LOGS_DIR}"/*.log

# 获取当前时间戳
TIMESTAMP=$(date '+%H-%M-%S')
MAIN_LOG="${LOGS_DIR}/main_${TIMESTAMP}.log"

# 全局进程数组
declare -a ALL_PIDS=()

# 日志函数
log_info() {
    local msg="[$(date '+%H:%M:%S')] [INFO] $1"
    echo "$msg" | tee -a "${MAIN_LOG}"
}

log_error() {
    local msg="[$(date '+%H:%M:%S')] [ERROR] $1"
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
    sleep 2
    
    # 强制杀死剩余进程
    for pid in "${ALL_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            log_info "强制关闭 PID: $pid"
            kill -KILL "$pid" 2>/dev/null || true
        fi
    done
    
    log_info "所有节点已关闭"
}

# 设置信号处理
trap cleanup SIGINT SIGTERM EXIT

log_info "=== 自动瞄准系统启动脚本 ==="
log_info "工作空间: ${WORKSPACE_DIR}"  # 此处将显示上一级目录路径
log_info "日志目录: ${LOGS_DIR}"

# 切换到工作空间目录（功能包目录）
cd "${WORKSPACE_DIR}" || {
    log_error "无法切换到工作空间目录: ${WORKSPACE_DIR}"
    exit 1
}

# 检查ROS2环境（强化环境加载，解决服务启动问题）
log_info "检查ROS2环境..."
# 手动加载环境变量（关键：服务模式下需显式加载）
source /etc/profile 2>/dev/null || true
source ~/.bashrc 2>/dev/null || true
source /opt/ros/humble/setup.bash 2>/dev/null || true

if ! command -v ros2 &> /dev/null; then
    log_error "ROS2未安装或未正确配置环境变量"
    log_error "当前环境变量PATH: $PATH"  # 调试用：输出PATH查看是否包含ROS2路径
    exit 1
fi

# 加载工作空间环境（功能包的install目录）
log_info "加载工作空间环境..."
source "${WORKSPACE_DIR}/install/setup.bash" 2>/dev/null || {
    log_error "工作空间install目录不存在，尝试构建功能包..."
    # 构建当前功能包（automatic_aiming）
    colcon build --packages-select automatic_aiming
    source "${WORKSPACE_DIR}/install/setup.bash" 2>/dev/null || {
        log_error "构建后仍无法加载工作空间，请检查CMakeLists.txt"
        exit 1
    }
}

# 定义所有节点
NODES=(
    "main_controller" 
    "target_detect"
    "gimbal_controller"
    "uart1_receiver"
    "uart1_sender"
    "uart3_sender"
    "circle_calc_perspective"
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
        
        # 修正颜色代码（原橙色30m为黑色，改为正确的橙色代码）
        case "$node_name" in
            "main_controller") color="\033[33m" ;;       # 黄色
            "target_detect") color="\033[31m" ;;         # 红色
            "gimbal_controller") color="\033[38;5;208m" ;;  # 橙色
            "circle_calc_perspective") color="\033[32m" ;;  # 绿色
            "uart1_receiver") color="\033[34m" ;;        # 蓝色
            "uart1_sender") color="\033[34m" ;;          # 蓝色
           
            "uart3_sender") color="\033[36m" ;;          # 青色

            *) color="\033[0m" ;;                        # 默认色
        esac
        
        # 启动节点并处理输出（显式指定ROS环境）
        ROS_DOMAIN_ID=0 ros2 run automatic_aiming "$node_name" 2>&1 | while IFS= read -r line; do
            timestamp=$(date '+%H:%M:%S')
            formatted_line="[$timestamp] [$node_name] $line"
            
            # 同时输出到终端（带颜色）和日志文件（无颜色）
            echo -e "${color}${formatted_line}\033[0m"
            echo "$formatted_line" >> "$log_file"
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