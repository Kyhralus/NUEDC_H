#!/bin/bash

# 自动瞄准系统启动脚本 - 稳定版
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
declare -a NODE_NAMES=()

# 全局标志变量
SHUTDOWN_REQUESTED=false

# 日志函数 - 修复ANSI代码写入文件问题
log_info() {
    local message="$1"
    local timestamp=$(date '+%H:%M:%S')
    echo -e "${GREEN}[INFO]${NC} ${WHITE}[$timestamp]${NC} $message"
    echo "[INFO] [$timestamp] $message" >> "$MAIN_LOG"
}

log_warn() {
    local message="$1"
    local timestamp=$(date '+%H:%M:%S')
    echo -e "${YELLOW}[WARN]${NC} ${WHITE}[$timestamp]${NC} $message"
    echo "[WARN] [$timestamp] $message" >> "$MAIN_LOG"
}

log_error() {
    local message="$1"
    local timestamp=$(date '+%H:%M:%S')
    echo -e "${RED}[ERROR]${NC} ${WHITE}[$timestamp]${NC} $message"
    echo "[ERROR] [$timestamp] $message" >> "$MAIN_LOG"
}

log_success() {
    local message="$1"
    local timestamp=$(date '+%H:%M:%S')
    echo -e "${GREEN}[SUCCESS]${NC} ${WHITE}[$timestamp]${NC} $message"
    echo "[SUCCESS] [$timestamp] $message" >> "$MAIN_LOG"
}

# 清理函数
cleanup() {
    if [ "$SHUTDOWN_REQUESTED" = true ]; then
        return
    fi
    
    SHUTDOWN_REQUESTED=true
    echo
    echo -e "${YELLOW}[SHUTDOWN]${NC} 收到退出信号 (Ctrl+C)，正在优雅关闭所有节点..."
    
    # 首先发送TERM信号进行优雅关闭
    for i in "${!ALL_PIDS[@]}"; do
        pid=${ALL_PIDS[$i]}
        node_name=${NODE_NAMES[$i]}
        if kill -0 "$pid" 2>/dev/null; then
            echo -e "${CYAN}[SHUTDOWN]${NC} 正在关闭节点: $node_name (PID: $pid)"
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    
    # 等待节点优雅关闭
    echo -e "${YELLOW}[SHUTDOWN]${NC} 等待节点优雅关闭 (3秒)..."
    for i in {1..3}; do
        echo -n "."
        sleep 1
    done
    echo
    
    # 检查并强制关闭仍在运行的进程
    force_killed=0
    for i in "${!ALL_PIDS[@]}"; do
        pid=${ALL_PIDS[$i]}
        node_name=${NODE_NAMES[$i]}
        if kill -0 "$pid" 2>/dev/null; then
            echo -e "${RED}[FORCE KILL]${NC} 强制关闭节点: $node_name (PID: $pid)"
            kill -KILL "$pid" 2>/dev/null || true
            force_killed=1
        fi
    done
    
    if [ $force_killed -eq 1 ]; then
        echo -e "${YELLOW}[WARNING]${NC} 部分节点被强制关闭"
    fi
    
    echo -e "${GREEN}[SUCCESS]${NC} 所有节点已安全关闭"
    echo -e "${PURPLE}感谢使用自动瞄准系统！${NC}"
    exit 0
}

# 设置信号处理 - 捕获Ctrl+C (SIGINT)
trap 'cleanup' SIGINT SIGTERM

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

# 检查节点是否成功启动
check_node_ready() {
    local node_name="$1"
    local max_attempts=10
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if ros2 node list 2>/dev/null | grep -q "$node_name"; then
            return 0
        fi
        sleep 0.5
        ((attempt++))
    done
    return 1
}

# 启动节点函数 - 完整日志记录模式
start_node() {
    local node_name="$1"
    local node_color=$(get_node_color "$node_name")
    local node_log="${LOGS_DIR}/${node_name}_${TIMESTAMP}.log"
    
    log_info "正在启动节点: $node_name"
    
    # 启动节点，完整输出到终端和日志文件
    {
        ros2 run automatic_aiming "$node_name" --ros-args --log-level INFO
    } 2>&1 | tee "$node_log" | while IFS= read -r line; do
        # 输出到终端，带节点名称前缀和颜色
        echo -e "${node_color}[$node_name]${NC} $line"
    done &
    
    local pid=$!
    ALL_PIDS+=($pid)
    NODE_NAMES+=("$node_name")
    
    # 简短等待，让节点开始启动
    sleep 2
    
    # 检查进程是否还在运行
    if kill -0 "$pid" 2>/dev/null; then
        echo -e "${node_color}[STARTED]${NC} ${node_name} 启动成功 (PID: $pid)"
    else
        log_error "${node_name} 节点启动失败"
        return 1
    fi
    
    # 节点间等待时间，确保稳定启动
    sleep 2
}

# 稳定启动所有节点
log_info "稳定启动模式 - 逐个启动节点..."
failed_nodes=()

for node in "${NODES[@]}"; do
    if ! start_node "$node"; then
        failed_nodes+=("$node")
    fi
done

# 启动完成提示
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  稳定自动瞄准系统运行中...${NC}"
echo -e "${GREEN}    按 Ctrl+C 安全停止所有节点${NC}"
echo -e "${GREEN}  所有节点输出已同步记录到日志${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
log_info "日志位置: ${LOGS_DIR} (旧日志已清理)"
log_info "系统已就绪，按 Ctrl+C 可安全退出"
echo -e "${CYAN}[提示] 节点输出同时显示在终端并记录到对应日志文件${NC}"

# 显示启动结果摘要
if [ ${#failed_nodes[@]} -gt 0 ]; then
    echo -e "${RED}[警告] 启动失败的节点: ${failed_nodes[*]}${NC}"
fi
echo -e "${CYAN}[状态] 成功启动 $((${#ALL_PIDS[@]} - ${#failed_nodes[@]}))/${#NODES[@]} 个节点${NC}"
echo ""

# 监控节点状态，不覆盖节点输出
while true; do
    if [ "$SHUTDOWN_REQUESTED" = true ]; then
        break
    fi
    
    active_count=0
    dead_nodes=()
    
    for i in "${!ALL_PIDS[@]}"; do
        pid=${ALL_PIDS[$i]}
        node_name=${NODE_NAMES[$i]}
        if kill -0 "$pid" 2>/dev/null; then
            ((active_count++))
        else
            dead_nodes+=("$node_name")
        fi
    done
    
    # 如果有节点意外退出，报告并继续
    if [ ${#dead_nodes[@]} -gt 0 ]; then
        echo -e "\n${RED}[ERROR]${NC} 检测到节点意外退出: ${dead_nodes[*]}"
    fi
    
    if [ $active_count -eq 0 ]; then
        echo -e "\n${RED}[ERROR]${NC} 所有节点已退出，程序结束"
        break
    fi
    
    sleep 5
done

# 如果到达这里说明所有节点都已退出，清理资源
if [ "$SHUTDOWN_REQUESTED" = false ]; then
    log_info "程序正常结束"
fi