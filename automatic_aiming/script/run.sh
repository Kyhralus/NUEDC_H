#!/bin/bash

# 自动瞄准系统启动脚本 - 简化版
# 功能：启动ROS2节点，提供彩色输出和日志记录

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# 获取脚本目录和工作空间
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(dirname "${SCRIPT_DIR}")"
LOGS_DIR="${WORKSPACE_ROOT}/logs"

# 清理旧日志并创建日志目录
log_cleanup() {
    if [ -d "${LOGS_DIR}" ]; then
        echo -e "${YELLOW}[CLEANUP]${NC} 清理旧日志文件..."
        rm -rf "${LOGS_DIR}"/*
        echo -e "${GREEN}[SUCCESS]${NC} 旧日志已清理"
    fi
    mkdir -p "${LOGS_DIR}"
}

# 执行日志清理
log_cleanup

# 日志文件设置
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
MAIN_LOG="${LOGS_DIR}/run_${TIMESTAMP}.log"

# 进程数组
declare -a ALL_PIDS=()

# 日志函数
log_info() {
    local message="$1"
    local timestamp=$(date '+%H:%M:%S')
    echo -e "${GREEN}[INFO]${NC} ${WHITE}[$timestamp]${NC} $message" | tee -a "$MAIN_LOG"
}

log_warn() {
    local message="$1"
    local timestamp=$(date '+%H:%M:%S')
    echo -e "${YELLOW}[WARN]${NC} ${WHITE}[$timestamp]${NC} $message" | tee -a "$MAIN_LOG"
}

log_error() {
    local message="$1"
    local timestamp=$(date '+%H:%M:%S')
    echo -e "${RED}[ERROR]${NC} ${WHITE}[$timestamp]${NC} $message" | tee -a "$MAIN_LOG"
}

log_success() {
    local message="$1"
    local timestamp=$(date '+%H:%M:%S')
    echo -e "${GREEN}[SUCCESS]${NC} ${WHITE}[$timestamp]${NC} $message" | tee -a "$MAIN_LOG"
}

# 清理函数
cleanup() {
    echo
    log_info "正在关闭所有节点..."
    
    for pid in "${ALL_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            log_info "关闭进程 PID: $pid"
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    
    sleep 2
    
    # 强制关闭剩余进程
    for pid in "${ALL_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -KILL "$pid" 2>/dev/null || true
        fi
    done
    
    log_success "所有节点已关闭"
}

# 设置信号处理
trap cleanup SIGINT SIGTERM EXIT

# 打印标题
echo -e "${PURPLE}================================${NC}"
echo -e "${PURPLE}    自动瞄准系统启动脚本${NC}"
echo -e "${PURPLE}================================${NC}"
echo ""

# 切换到工作空间
cd "${WORKSPACE_ROOT}" || {
    log_error "无法切换到工作空间: ${WORKSPACE_ROOT}"
    exit 1
}

# 检查ROS2环境
log_info "检查ROS2环境..."
source /opt/ros/humble/setup.bash 2>/dev/null || true

if ! command -v ros2 &> /dev/null; then
    log_error "ROS2未安装或环境未配置"
    exit 1
fi
log_success "ROS2环境检查通过"

# 构建项目
# log_info "构建项目..."
# if colcon build --packages-select automatic_aiming --symlink-install 2>&1 | tee -a "$MAIN_LOG"; then
#     log_success "项目构建成功"
# else
#     log_error "项目构建失败"
#     exit 1
# fi

# 加载工作空间环境
source install/setup.bash
log_success "工作空间环境加载完成"

# GPIO初始化配置
log_info "配置GPIO引脚..."
if command -v gpio &> /dev/null; then
    # 配置 GPIO54 为输入模式（对应 wiringPi 编号 2）
    if gpio mode 2 in 2>/dev/null; then
        log_success "GPIO54 (wiringPi 2) 配置为输入模式"
    else
        log_warn "GPIO54 (wiringPi 2) 配置失败，可能需要sudo权限"
    fi
    
    # 配置 GPIO59 为输出模式（对应 wiringPi 编号 9）
    if gpio mode 9 out 2>/dev/null; then
        log_success "GPIO59 (wiringPi 9) 配置为输出模式"
    else
        log_warn "GPIO59 (wiringPi 9) 配置失败，可能需要sudo权限"
    fi
else
    log_warn "gpio命令未找到，跳过GPIO配置"
    log_info "请确保安装了wiringPi库或手动配置GPIO"
fi

# 定义要启动的节点
NODES=(
    "main_controller"
    "target_detect"
    # "gimbal_controller"  # 已删除
    "uart1_receiver"
    "uart1_sender"
    "uart3_sender"
    "circle_calc_perspective"
)

# 节点颜色映射
get_node_color() {
    case "$1" in
        "main_controller") echo "${YELLOW}" ;;
        "target_detect") echo "${RED}" ;;
        # "gimbal_controller") echo "${PURPLE}" ;;   # 已删除
        "circle_calc_perspective") echo "${GREEN}" ;;
        "uart1_receiver") echo "${BLUE}" ;;
        "uart1_sender") echo "${BLUE}" ;;
        "uart3_sender") echo "${CYAN}" ;;
        *) echo "${NC}" ;;
    esac
}

# 启动节点函数 - 高速精准模式
start_node() {
    local node_name="$1"
    local node_color=$(get_node_color "$node_name")
    local node_log="${LOGS_DIR}/${node_name}_${TIMESTAMP}.log"
    
    # 高速启动，减少日志输出
    {
        ros2 run automatic_aiming "$node_name" --ros-args --log-level WARN
    } > "$node_log" 2>&1 &
    
    local pid=$!
    ALL_PIDS+=($pid)
    echo -e "${node_color}[FAST]${NC} ${node_name} 已启动 (PID: $pid)"
    
    # 移除延迟，提高启动速度
}

# 高速启动所有节点
log_info "高速启动模式 - 启动所有节点..."
for node in "${NODES[@]}"; do
    start_node "$node"
done

# 快速验证节点启动状态
sleep 0.5
log_success "所有节点启动完成，共 ${#ALL_PIDS[@]} 个进程"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  高速精准自动瞄准系统运行中...${NC}"
echo -e "${GREEN}    按 Ctrl+C 停止所有节点${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
log_info "日志位置: ${LOGS_DIR} (旧日志已清理)"
echo ""

# 高效监控节点状态
while true; do
    active_count=0
    for pid in "${ALL_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            ((active_count++))
        fi
    done
    
    # 减少监控频率，降低系统开销
    echo -ne "\r${CYAN}[$(date '+%H:%M:%S')] 运行中: $active_count/${#ALL_PIDS[@]} 节点${NC}  "
    
    if [ $active_count -eq 0 ]; then
        echo
        log_error "所有节点已退出"
        break
    fi
    
    sleep 5  # 从2秒增加到5秒，减少系统开销
done

cleanup