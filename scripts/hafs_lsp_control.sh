#!/bin/bash
# hafs-lsp control script - enable/disable, swap models, check status

CONFIG_FILE="$HOME/Code/hafs/config/lsp.toml"

show_status() {
    echo "========================================================================"
    echo "hafs-lsp Status"
    echo "========================================================================"
    echo ""

    # Check if enabled
    enabled=$(grep "^enabled" "$CONFIG_FILE" | awk '{print $3}')
    auto_start=$(grep "^auto_start" "$CONFIG_FILE" | awk '{print $3}')
    strategy=$(grep "^strategy" "$CONFIG_FILE" | awk '{print $3}' | tr -d '"')

    if [ "$enabled" = "true" ]; then
        echo "Status: ✓ ENABLED"
    else
        echo "Status: ✗ DISABLED"
    fi

    echo "Auto-start: $auto_start"
    echo "Strategy: $strategy"
    echo ""

    # Show active models
    echo "Models:"
    fast=$(grep "^fast_model" "$CONFIG_FILE" | awk '{print $3}' | tr -d '"')
    quality=$(grep "^quality_model" "$CONFIG_FILE" | awk '{print $3}' | tr -d '"')

    echo "  Fast:    $fast"
    echo "  Quality: $quality"
    echo ""

    # Check if models are downloaded
    echo "Downloaded Models:"
    ollama list | grep qwen2.5-coder || echo "  (no models found)"
    echo ""

    # Show resource usage if running
    if pgrep -f "hafs_lsp.py" > /dev/null; then
        echo "LSP Server: RUNNING"
        pid=$(pgrep -f "hafs_lsp.py")
        echo "  PID: $pid"
        ps -p "$pid" -o %cpu,%mem,rss | tail -1 | awk '{printf "  CPU: %s%%  Memory: %s%% (%s KB)\n", $1, $2, $3}'
    else
        echo "LSP Server: NOT RUNNING"
    fi

    echo ""
    echo "========================================================================"
}

enable_lsp() {
    sed -i.bak 's/^enabled = false/enabled = true/' "$CONFIG_FILE"
    echo "✓ hafs-lsp ENABLED"
    echo "  Restart your editor to activate"
}

disable_lsp() {
    sed -i.bak 's/^enabled = true/enabled = false/' "$CONFIG_FILE"
    echo "✓ hafs-lsp DISABLED"
    echo "  Restart your editor to deactivate"
}

set_strategy() {
    strategy=$1

    if [ -z "$strategy" ]; then
        echo "Available strategies:"
        echo "  fast_only       - Always use fast model (1.5B)"
        echo "  quality_only    - Always use quality model (7B)"
        echo "  adaptive        - Switch based on context size"
        echo "  manual_trigger  - Only on Ctrl+Space (no auto-complete)"
        echo ""
        echo "Usage: $0 strategy <name>"
        return 1
    fi

    sed -i.bak "s/^strategy = .*/strategy = \"$strategy\"/" "$CONFIG_FILE"
    echo "✓ Strategy set to: $strategy"
}

set_model() {
    model_type=$1
    model_name=$2

    if [ -z "$model_type" ] || [ -z "$model_name" ]; then
        echo "Usage: $0 model <fast|quality> <model-name>"
        echo ""
        echo "Example:"
        echo "  $0 model fast qwen2.5-coder:1.5b"
        echo "  $0 model quality hafs-asm-7b-20251221-gold"
        return 1
    fi

    if [ "$model_type" = "fast" ]; then
        sed -i.bak "s|^fast_model = .*|fast_model = \"$model_name\"|" "$CONFIG_FILE"
        echo "✓ Fast model set to: $model_name"
    elif [ "$model_type" = "quality" ]; then
        sed -i.bak "s|^quality_model = .*|quality_model = \"$model_name\"|" "$CONFIG_FILE"
        echo "✓ Quality model set to: $model_name"
    else
        echo "✗ Invalid model type. Use 'fast' or 'quality'"
        return 1
    fi
}

set_custom_model() {
    model_type=$1
    model_path=$2

    if [ -z "$model_type" ] || [ -z "$model_path" ]; then
        echo "Usage: $0 custom <fast|quality> <path-to-model>"
        echo ""
        echo "Example:"
        echo "  $0 custom fast ~/Code/hafs/models/hafs-asm-1.5b-20251221-gold"
        return 1
    fi

    if [ "$model_type" = "fast" ]; then
        sed -i.bak "s|^custom_fast = .*|custom_fast = \"$model_path\"|" "$CONFIG_FILE"
        echo "✓ Custom fast model set to: $model_path"
    elif [ "$model_type" = "quality" ]; then
        sed -i.bak "s|^custom_quality = .*|custom_quality = \"$model_path\"|" "$CONFIG_FILE"
        echo "✓ Custom quality model set to: $model_path"
    else
        echo "✗ Invalid model type. Use 'fast' or 'quality'"
        return 1
    fi
}

list_models() {
    echo "Available Models (from ollama):"
    echo ""
    ollama list
    echo ""
    echo "To download a new model:"
    echo "  ollama pull <model-name>"
}

case "$1" in
    status)
        show_status
        ;;
    enable)
        enable_lsp
        ;;
    disable)
        disable_lsp
        ;;
    strategy)
        set_strategy "$2"
        ;;
    model)
        set_model "$2" "$3"
        ;;
    custom)
        set_custom_model "$2" "$3"
        ;;
    list)
        list_models
        ;;
    *)
        echo "hafs-lsp control script"
        echo ""
        echo "Usage: $0 <command> [args]"
        echo ""
        echo "Commands:"
        echo "  status              - Show current status"
        echo "  enable              - Enable hafs-lsp"
        echo "  disable             - Disable hafs-lsp"
        echo "  strategy <name>     - Set model selection strategy"
        echo "  model <type> <name> - Set fast or quality model"
        echo "  custom <type> <path> - Set custom fine-tuned model"
        echo "  list                - List available ollama models"
        echo ""
        echo "Examples:"
        echo "  $0 status"
        echo "  $0 enable"
        echo "  $0 strategy manual_trigger"
        echo "  $0 model fast qwen2.5-coder:1.5b"
        echo "  $0 custom fast ~/Code/hafs/models/hafs-asm-1.5b-gold"
        echo ""
        exit 1
        ;;
esac
