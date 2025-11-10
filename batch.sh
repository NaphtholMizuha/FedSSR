#!/bin/bash

# Default value for method
method_arg=""

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --method)
        method_arg="$2"
        shift # past argument
        shift # past value
        ;; 
        *)
        shift # past argument
        ;; 
    esac
done

# 确保日志目录存在
mkdir -p log

# 启用 nullglob，避免 conf/*.toml 无匹配时字面返回 "conf/*.toml"
shopt -s nullglob

for conf in conf/*.toml; do
    # 跳过空匹配（虽然 nullglob 已处理，双重保险）
    [[ -f "$conf" ]] || continue

    # 如果传递了 method 参数，则修改配置文件
    

    # 提取文件名（不含路径和 .toml 后缀）
    name=$(basename "$conf" .toml)

    # 清理对应的旧日志文件
    log_file="log/${name}.log"
    echo "Clearing old log file: $log_file"
    rm -f "$log_file"

    # 如果 tmux 会话已存在，先 kill 掉
    if tmux has-session -t "$name" 2>/dev/null; then
        tmux kill-session -t "$name"
        echo "Existing tmux session '$name' killed."
    fi

    # 启动新的后台 tmux 会话
    # 注意：tmux 会将命令传给 sh，因此需确保变量正确引用
    

    if [[ -n "$method_arg" ]]; then
        tmux new-session -d -s "$name" "uv run -m src.main -c '$conf' --method '$method_arg'"
    else
        tmux new-session -d -s "$name" "uv run -m src.main -c '$conf'"
    fi
    



    echo "Tmux session '$name' created for config '$conf'."
done