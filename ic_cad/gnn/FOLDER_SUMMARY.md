# GNN 資料夾整理完成！

## 📁 gnn/ 資料夾結構

```
gnn/
├── 📄 README.md                    # 完整使用說明 (新)
├── 📄 README_USAGE.md             # 原有使用說明
│
├── 🔧 核心模組
│   ├── graph_builder.py           # 電路圖構建 (32維 embedding)
│   ├── gnn_api.py                 # GNN 模型接口
│   └── inference.py               # 推論模組
│
├── 🎯 訓練工具  
│   ├── config_train_dgi.py        # DGI 訓練腳本
│   ├── retrain_gnn.sh            # 快速重新訓練 (新)
│   └── test_embedding.py         # 架構測試工具 (新)
│
├── 📊 數據檔案
│   ├── cell_groups.json          # ASAP7 cell 分組 (新位置)
│   └── cell_id_mapping.json      # Cell ID 映射表 (新)
│
└── 🏆 訓練輸出
    ├── encoder.pt                 # 訓練好的模型
    ├── encoder_meta.json          # 模型元數據
    ├── best_encoder_all_by_probe.pt  # 最佳模型
    └── loss_*.png                 # 訓練曲線
```

## ✅ 整理完成的改進

### 🔄 檔案移動
- ✅ `cell_groups.json` → `gnn/`
- ✅ `test_embedding.py` → `gnn/`  
- ✅ `retrain_gnn.sh` → `gnn/`

### 📝 文檔更新
- ✅ 創建詳細的 `README.md`
- ✅ 說明新的 32 維 embedding 架構
- ✅ 提供完整的使用指南

### 🔗 路徑修正
- ✅ 更新檔案內部的路徑引用
- ✅ 確保 gnn 資料夾內檔案可獨立運行
- ✅ 保持與 RL 模組的正確接口

## 🚀 使用方式

### 在 gnn 資料夾內工作
```bash
cd /root/ruan_workspace/ic_cad/gnn

# 測試架構
python test_embedding.py

# 重新訓練
./retrain_gnn.sh

# 手動訓練
python config_train_dgi.py --test-only
```

### 從外部調用
```python
# 在 RL 或其他模組中
from gnn.graph_builder import build_graph_from_case
from gnn.gnn_api import load_encoder
```

## 🎯 架構優勢

- **模組化**: 所有 GNN 相關檔案集中管理
- **獨立性**: gnn 資料夾可獨立使用
- **高效性**: 32 維 embedding 取代 852 維 one-hot
- **可維護**: 清晰的檔案結構和完整文檔

現在 gnn 資料夾已經完全整理好，可以開始重新訓練模型了！
