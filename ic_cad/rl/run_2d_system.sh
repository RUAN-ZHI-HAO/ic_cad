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
    echo "  $0 train c17 c432                               # 訓練模式（從頭開始）"
    echo "  $0 train c17 --episodes 100                     # 訓練模式（指定回合數）"
    echo "  $0 train c17 --model-path models/best.pth       # 繼續訓練（載入模型）"
    echo "  $0 train c17 --tns-weight 2.0 --power-weight 1.0  # 訓練模式（指定權重）"
    echo "  $0 optimize c17                                 # 優化模式（使用預設模型）"
    echo "  $0 optimize c17 --model-path /path/to/model.pth # 優化模式（指定模型）"
    echo "  $0 optimize c17 --tns-weight 1.5 --power-weight 0.5  # 優化模式（指定權重）"
    echo "  $0 full c17 c432                               # 完整流程"
    echo "  $0 full c17 --tns-weight 3.0 --power-weight 1.0     # 完整流程（指定權重）"
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
        echo "參數: $@"
        openroad -python -exit -c "
import sys
sys.argv = ['main.py', '--mode', 'optimize'] + \"$*\".split()
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
