#!/usr/bin/env bash
# 启动/停止/重启 ComfyUI。
#
# 为什么需要它:
#   1. ComfyUI 必须用 conda `sensenova` 环境启动(不是 test,也不是仓库里的 venv)——
#      SenseNova-U1 节点的依赖只装在那个环境里。
#   2. custom_nodes/ComfyUI-SenseNova-U1 把模型缓存在插件私有的 _LOCAL_MODEL_CACHE
#      全局 dict 里,ComfyUI 的 /free 接口**管不到**(实测调用返回 200 但显存不降)。
#      跑过 SenseNova 想跑 LTX/Wan 这种吃显存的,只能重启进程来释放 —— 就是 `restart`。
#
# 用法:
#   bash start_comfy.sh            # 启动(已在跑则什么都不做)
#   bash start_comfy.sh restart    # 重启(释放显存;队列非空会先警告)
#   bash start_comfy.sh stop       # 停止
#   bash start_comfy.sh status     # 看进程/端口/显存/队列
set -euo pipefail

COMFY=/home/ubuntu/projects/mead/ComfyUI
PY=/home/ubuntu/miniconda3/envs/sensenova/bin/python
PORT=8188
LOG=$COMFY/user/default/comfy.log

cd "$COMFY"

pids() { pgrep -f "main.py --listen .* --port $PORT" || true; }

vram() { nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader; }

queue_busy() {
    local r p
    r=$(curl -s -m 3 "http://127.0.0.1:$PORT/queue" 2>/dev/null) || return 1
    p=$(printf '%s' "$r" | "$PY" -c "
import json,sys
try:
    q=json.load(sys.stdin)
    print(len(q.get('queue_running',[]))+len(q.get('queue_pending',[])))
except Exception: print(0)
" 2>/dev/null) || return 1
    [ "${p:-0}" != "0" ]
}

do_status() {
    local ps; ps=$(pids)
    if [ -n "$ps" ]; then
        echo "进程:  运行中 (pid $(echo $ps | tr '\n' ' '))"
        ps -o etime=,cmd= -p $ps | sed 's/^/       /'
    else
        echo "进程:  未运行"
    fi
    echo "端口:  $(ss -ltnp 2>/dev/null | grep ":$PORT " | head -1 || echo "  $PORT 未监听")"
    echo "显存:  $(vram)"
    if curl -s -m 3 "http://127.0.0.1:$PORT/queue" >/dev/null 2>&1; then
        if queue_busy; then echo "队列:  ⚠ 有任务在跑/排队"; else echo "队列:  空"; fi
    else
        echo "队列:  接口不通"
    fi
}

do_stop() {
    local ps; ps=$(pids)
    if [ -z "$ps" ]; then echo "[停止] 本来就没在跑"; return 0; fi
    echo "[停止] pid $(echo $ps | tr '\n' ' ')"
    kill -TERM $ps 2>/dev/null || true
    for i in $(seq 1 30); do
        sleep 1
        [ -z "$(pids)" ] && { echo "  ✓ 已退出 (${i}s)"; return 0; }
    done
    echo "  ⚠ 30s 未退出,强杀"
    kill -KILL $(pids) 2>/dev/null || true
    sleep 2
}

do_start() {
    if [ -n "$(pids)" ]; then
        echo "[启动] 已在运行,跳过。要释放显存请用 restart"
        do_status
        return 0
    fi
    # 端口残留检查:避免起来后静默抢不到端口
    if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
        echo "✗ 端口 $PORT 已被别的进程占用:"
        ss -ltnp 2>/dev/null | grep ":$PORT "
        return 1
    fi
    echo "[启动] $PY main.py --listen 0.0.0.0 --port $PORT"
    nohup "$PY" main.py --listen 0.0.0.0 --port "$PORT" >>"$LOG" 2>&1 &
    for i in $(seq 1 90); do
        sleep 1
        if curl -s -m 2 "http://127.0.0.1:$PORT/system_stats" >/dev/null 2>&1; then
            echo "  ✓ 就绪 (${i}s)  http://127.0.0.1:$PORT"
            echo "  显存: $(vram)"
            return 0
        fi
    done
    echo "  ✗ 90s 内没起来,看日志尾部:"
    tail -25 "$LOG"
    return 1
}

case "${1:-start}" in
    start)  do_start;;
    stop)   do_stop;   echo "显存: $(vram)";;
    status) do_status;;
    restart)
        if queue_busy; then
            echo "⚠ 队列里还有任务,重启会打断它们。"
            read -r -p "  继续? [y/N] " a
            case "$a" in y|Y) ;; *) echo "已取消"; exit 1;; esac
        fi
        do_stop; do_start;;
    *) echo "用法: bash start_comfy.sh [start|stop|restart|status]"; exit 1;;
esac
