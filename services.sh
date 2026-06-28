#!/usr/bin/env bash
#
# Enterprise RAG 服务管理脚本
#
# 用法:
#   ./services.sh start          # 启动全部服务（基础设施 + API）
#   ./services.sh stop           # 停止全部服务
#   ./services.sh restart        # 重启全部服务
#   ./services.sh infra up       # 仅启动基础设施容器
#   ./services.sh infra down     # 停止并删除基础设施容器
#   ./services.sh app start      # 仅启动 FastAPI 应用
#   ./services.sh app stop       # 仅停止 FastAPI 应用
#   ./services.sh status         # 查看所有服务状态
#   ./services.sh logs [service] # 查看日志（app / mysql / redis / milvus）

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"
DOCKER_COMPOSE_FILE="$PROJECT_DIR/docker-compose.yml"
PID_FILE="$PROJECT_DIR/.rag-app.pid"
LOG_FILE="$PROJECT_DIR/logs/app.log"

# Docker compose 关键服务（排除 rag-app 构建）
INFRA_SERVICES="milvus-etcd milvus-minio milvus-standalone mysql redis attu"

# ──────────────────────────────────────────────
# 颜色输出
# ──────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }

# ──────────────────────────────────────────────
# 基础设施管理
# ──────────────────────────────────────────────
infra_up() {
    info "启动基础设施容器..."
    cd "$PROJECT_DIR"
    docker compose up -d $INFRA_SERVICES
    info "基础设施已启动"
    infra_status
}

infra_down() {
    local delete_volumes="${1:-}"
    info "停止基础设施容器..."
    cd "$PROJECT_DIR"
    if [ "$delete_volumes" = "-v" ]; then
        warn "将删除所有数据卷（向量库/数据库/缓存数据将丢失）"
        docker compose down -v
    else
        docker compose down
    fi
    info "基础设施已停止"
}

infra_stop() {
    info "暂停基础设施容器（保留容器和网络）..."
    cd "$PROJECT_DIR"
    docker compose stop
    info "基础设施已暂停"
}

infra_status() {
    echo ""
    echo "══════════════════════════════════════════════════"
    echo "  Docker 容器状态"
    echo "══════════════════════════════════════════════════"
    cd "$PROJECT_DIR"
    docker compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null || echo "  无运行中的容器"
    echo ""
}

# ──────────────────────────────────────────────
# 应用管理
# ──────────────────────────────────────────────
app_start() {
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        warn "FastAPI 应用已在运行 (PID: $(cat "$PID_FILE"))"
        return 0
    fi

    # 检查端口是否被占用
    local port="${RAG_SERVER_PORT:-8000}"
    if fuser "${port}/tcp" &>/dev/null 2>&1; then
        warn "端口 $port 已被占用，尝试释放..."
        fuser -k "${port}/tcp" 2>/dev/null || true
        sleep 1
    fi

    info "启动 FastAPI 应用..."
    mkdir -p "$(dirname "$LOG_FILE")"
    cd "$PROJECT_DIR"
    nohup "$VENV_PYTHON" -m src.main >> "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"

    # 等待服务就绪
    local max_wait=30
    local waited=0
    while [ $waited -lt $max_wait ]; do
        if curl -sf http://localhost:$port/health > /dev/null 2>&1; then
            echo ""
            info "FastAPI 应用启动成功 → http://localhost:$port"
            info "API 文档 → http://localhost:$port/docs"
            return 0
        fi
        echo -n "."
        sleep 2
        waited=$((waited + 2))
    done
    echo ""
    error "应用启动超时，请检查日志: tail -f $LOG_FILE"
    return 1
}

app_stop() {
    if fuser "${RAG_SERVER_PORT:-8000}/tcp" &>/dev/null 2>&1; then
        info "停止 FastAPI 应用..."
        fuser -k "${RAG_SERVER_PORT:-8000}/tcp" 2>/dev/null || true
    else
        warn "FastAPI 应用未在运行"
    fi
    rm -f "$PID_FILE"
}

app_status() {
    echo ""
    echo "══════════════════════════════════════════════════"
    echo "  FastAPI 应用状态"
    echo "══════════════════════════════════════════════════"
    local port="${RAG_SERVER_PORT:-8000}"
    if fuser "${port}/tcp" &>/dev/null 2>&1; then
        local pid
        pid=$(fuser "${port}/tcp" 2>/dev/null | cut -d' ' -f1)
        echo "  状态: 运行中 (PID: $pid)"
        echo "  端口: $port"
        if curl -sf http://localhost:$port/health > /dev/null 2>&1; then
            local health
            health=$(curl -s http://localhost:$port/health)
            echo "  健康: $health"
        fi
        echo "  API 文档: http://localhost:$port/docs"
    else
        echo "  状态: 未运行"
    fi
    echo ""
}

# ──────────────────────────────────────────────
# 查看日志
# ──────────────────────────────────────────────
show_logs() {
    local target="${1:-app}"
    case "$target" in
        app)
            if [ -f "$LOG_FILE" ]; then
                tail -f "$LOG_FILE"
            else
                warn "应用日志文件不存在: $LOG_FILE"
                warn "请先启动应用: ./services.sh app start"
            fi
            ;;
        mysql|redis|milvus|mivlus-standalone|attu|milvus-etcd|milvus-minio)
            cd "$PROJECT_DIR"
            docker compose logs -f --tail=100 "$target"
            ;;
        all)
            cd "$PROJECT_DIR"
            docker compose logs -f --tail=100
            ;;
        *)
            error "未知服务: $target (可选: app, mysql, redis, milvus-standalone, attu, all)"
            ;;
    esac
}

# ──────────────────────────────────────────────
# 环境检查
# ──────────────────────────────────────────────
check_env() {
    echo ""
    echo "══════════════════════════════════════════════════"
    echo "  环境检查"
    echo "══════════════════════════════════════════════════"
    # Python
    if [ -f "$VENV_PYTHON" ]; then
        echo "  Python: $("$VENV_PYTHON" --version 2>&1)"
    else
        echo "  Python: ⚠ 虚拟环境不存在 ($VENV_PYTHON)"
    fi
    # Docker
    if command -v docker &>/dev/null; then
        echo "  Docker: $(docker --version 2>&1)"
    else
        echo "  Docker: ✗ 未安装"
    fi
    # .env
    if [ -f "$PROJECT_DIR/.env" ]; then
        echo "  .env:   ✅ 已配置"
    else
        echo "  .env:   ⚠ 未配置 (cp .env.example .env)"
    fi
    echo ""
}

# ──────────────────────────────────────────────
# 帮助
# ──────────────────────────────────────────────
show_help() {
    echo "Enterprise RAG 服务管理脚本"
    echo ""
    echo "用法:  ./services.sh <command> [subcommand]"
    echo ""
    echo "命令:"
    echo "  start           启动全部服务（基础设施 + API）"
    echo "  stop            停止全部服务"
    echo "  restart         重启全部服务"
    echo "  status          查看所有服务状态"
    echo "  check           环境检查"
    echo ""
    echo "  infra up        仅启动基础设施容器"
    echo "  infra down      停止并删除基础设施容器"
    echo "  infra down -v   停止并删除容器 + 所有数据卷"
    echo "  infra restart   重启基础设施"
    echo ""
    echo "  app start       仅启动 FastAPI 应用"
    echo "  app stop        仅停止 FastAPI 应用"
    echo "  app restart     重启 FastAPI 应用"
    echo ""
    echo "  logs [service]  查看日志 (app / mysql / redis / milvus-standalone / all)"
    echo "  help            显示此帮助"
    echo ""
    echo "示例:"
    echo "  ./services.sh start              # 一键启动全部"
    echo "  ./services.sh stop               # 一键停止全部"
    echo "  ./services.sh logs app           # 查看应用日志"
    echo "  ./services.sh infra down -v      #  清空所有数据"
}

# ──────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────
main() {
    if [ $# -eq 0 ]; then
        show_help
        exit 0
    fi

    case "${1:-}" in
        start)
            infra_up
            app_start
            ;;
        stop)
            app_stop
            infra_stop
            ;;
        restart)
            app_stop
            infra_stop
            infra_up
            app_start
            ;;
        status)
            check_env
            app_status
            infra_status
            ;;
        check)
            check_env
            ;;
        infra)
            case "${2:-}" in
                up)      infra_up ;;
                down)    infra_down "${3:-}" ;;
                restart) infra_stop; infra_up ;;
                status)  infra_status ;;
                *)       error "用法: $0 infra {up|down|restart|status}"; exit 1 ;;
            esac
            ;;
        app)
            case "${2:-}" in
                start)   app_start ;;
                stop)    app_stop ;;
                restart) app_stop; app_start ;;
                status)  app_status ;;
                *)       error "用法: $0 app {start|stop|restart|status}"; exit 1 ;;
            esac
            ;;
        logs)
            show_logs "${2:-app}"
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            error "未知命令: $1"
            show_help
            exit 1
            ;;
    esac
}

main "$@"
