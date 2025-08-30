#!/bin/bash
# 2D 動作空間 IC CAD 強化學習系統 - OpenROAD 啟動腳本

# 設置工作目錄
cd /root/ruan_workspace/ic_cad/rl

echo "=== 2D 動作空間 IC CAD 強化學習系統 ==="
echo "使用 OpenROAD Python 環境運行"
echo ""

# 檢查參數
if [ $# -eq 0 ]; then
    echo "使用方式:"
    echo "  $0 [train|optimize|full|test] [其他參數...]"
    echo ""
    echo "範例:"
    echo "  $0 test                                          # 運行系統測試"
    echo "  $0 train c17 c432                               # 訓練模式"
    echo "  $0 optimize c17                                 # 優化模式 (使用預設模型)"
    echo "  $0 optimize c17 /path/to/model.pth             # 優化模式 (指定模型)"
    echo "  $0 full c17 c432                               # 完整流程"
    echo ""
    exit 1
fi

MODE=$1
shift

case $MODE in
    "test")
        echo "運行 2D 動作空間系統測試..."
        openroad -python -exit test_2d_system.py
        ;;
    "train")
        echo "啟動訓練模式..."
        echo "案例: $@"
        openroad -python -exit -c "
import sys
sys.argv = ['main.py', '--mode', 'train', '--cases'] + \"$*\".split()
exec(open('main.py').read())
"
        ;;
    "optimize")
        echo "啟動優化模式..."
        
        # 檢查是否提供了模型路徑
        MODEL_ARGS=""
        CASES=""
        
        # 解析參數：第一個是案例，第二個（如果存在）是模型路徑
        for arg in "$@"; do
            if [[ -f "$arg" && "$arg" == *.pth ]]; then
                # 這是一個存在的 .pth 檔案，當作模型路徑
                MODEL_ARGS="--model-path $arg"
                echo "使用指定模型: $arg"
            else
                # 這是案例名稱
                CASES="$CASES $arg"
            fi
        done
        
        if [ -z "$CASES" ]; then
            echo "錯誤：沒有指定電路案例"
            exit 1
        fi
        
        echo "案例: $CASES"
        openroad -python -exit -c "
import sys
sys.argv = ['main.py', '--mode', 'optimize', '--cases'] + '$CASES'.split()
if '$MODEL_ARGS':
    sys.argv.extend('$MODEL_ARGS'.split())
exec(open('main.py').read())
"
        ;;
    "full")
        echo "啟動完整流程..."
        echo "案例: $@"
        openroad -python -exit -c "
import sys
sys.argv = ['main.py', '--mode', 'full', '--cases'] + \"$*\".split()
exec(open('main.py').read())
"
        ;;
    *)
        echo "未知模式: $MODE"
        echo "支援的模式: test, train, optimize, full"
        exit 1
        ;;
esac
