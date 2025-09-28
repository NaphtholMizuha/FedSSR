#!/bin/fish

for conf in conf/*.toml
    set name (basename "$conf" .toml)
    set log_file "log/$name.log"

    # 如果会话已存在，先 kill 掉
    if tmux has-session -t "$name" 2>/dev/null
        tmux kill-session -t "$name"
        echo "Existing tmux session '$name' killed."
    end

    # 创建新会话
    tmux new-session -d -s "$name" "uv run -m src.main -c '$conf' > '$log_file' 2>&1"

    echo "Tmux session '$name' created for config '$conf'."
end