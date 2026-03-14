#!/bin/bash
# Helper for managing training runs on RunPod via SSH+tmux.
#
# Usage:
#   ./scripts/pod.sh launch "python scripts/train.py --max-steps 3 -v 3"
#   ./scripts/pod.sh logs           # tail the live log
#   ./scripts/pod.sh logs-full      # cat the full log
#   ./scripts/pod.sh trace          # tail the structured trace.jsonl
#   ./scripts/pod.sh status         # check if training is running
#   ./scripts/pod.sh kill           # kill the training process
#   ./scripts/pod.sh pull           # git pull on the pod
#   ./scripts/pod.sh ssh            # open interactive SSH session
#   ./scripts/pod.sh cmd "any cmd"  # run an arbitrary command on the pod

set -euo pipefail

# Pod SSH config — update these if the pod changes
POD_KEY="$HOME/.runpod/ssh/RunPod-Key-Go"
POD_HOST="root@67.223.143.80"
POD_PORT="19325"
POD_DIR="/root/OM-rl"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10"

ssh_cmd() {
    ssh $SSH_OPTS -i "$POD_KEY" "$POD_HOST" -p "$POD_PORT" "$@"
}

case "${1:-help}" in
    launch)
        shift
        CMD="${*:-python scripts/train.py --max-steps 3 -v 2}"
        echo "Launching on pod: $CMD"
        ssh_cmd "cd $POD_DIR && tmux kill-session -t train 2>/dev/null; tmux new-session -d -s train '$CMD 2>&1 | tee /root/train.log; echo DONE; sleep 999999'"
        echo "Started in tmux session 'train'. Use './scripts/pod.sh logs' to watch."
        ;;
    logs)
        echo "=== Live log (Ctrl-C to stop watching) ==="
        ssh_cmd "tail -f /root/train.log" || true
        ;;
    logs-full)
        ssh_cmd "cat /root/train.log"
        ;;
    logs-last)
        N="${2:-30}"
        ssh_cmd "tail -$N /root/train.log"
        ;;
    trace)
        ssh_cmd "tail -f $POD_DIR/outputs/logs/trace.jsonl" || true
        ;;
    trace-last)
        N="${2:-10}"
        ssh_cmd "tail -$N $POD_DIR/outputs/logs/trace.jsonl" | python3 -m json.tool --no-ensure-ascii 2>/dev/null || ssh_cmd "tail -$N $POD_DIR/outputs/logs/trace.jsonl"
        ;;
    status)
        ssh_cmd "tmux has-session -t train 2>/dev/null && echo 'RUNNING (tmux session active)' || echo 'NOT RUNNING'"
        ssh_cmd "ps aux | grep '[t]rain.py' | head -3 || true"
        ;;
    kill)
        echo "Killing training..."
        ssh_cmd "tmux kill-session -t train 2>/dev/null; killall python 2>/dev/null; echo done"
        ;;
    pull)
        echo "Pulling latest code on pod..."
        ssh_cmd "cd $POD_DIR && git pull"
        ;;
    ssh)
        echo "Opening interactive SSH..."
        ssh $SSH_OPTS -t -i "$POD_KEY" "$POD_HOST" -p "$POD_PORT" "cd $POD_DIR && bash"
        ;;
    cmd)
        shift
        ssh_cmd "cd $POD_DIR && $*"
        ;;
    help|*)
        echo "Usage: ./scripts/pod.sh <command>"
        echo ""
        echo "Commands:"
        echo "  launch \"cmd\"  — Start a training run in tmux (kills any existing)"
        echo "  logs          — Tail the live training log"
        echo "  logs-full     — Cat the full log"
        echo "  logs-last [N] — Show last N lines (default 30)"
        echo "  trace         — Tail the structured trace.jsonl"
        echo "  trace-last [N]— Show last N trace entries as pretty JSON"
        echo "  status        — Check if training is running"
        echo "  kill          — Kill the training process"
        echo "  pull          — Git pull on the pod"
        echo "  ssh           — Open interactive SSH to the pod"
        echo "  cmd \"cmd\"     — Run an arbitrary command on the pod"
        ;;
esac
