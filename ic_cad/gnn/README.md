# GNN 模組說明

本資料夾包含 IC CAD 優化系統的圖神經網路（GNN）相關檔案，實現了基於 32 維 cell embedding 的高效電路表示學習。

## 📁 檔案結構

### 🔧 核心模組
- **`graph_builder.py`** - 電路圖構建器
  - 將電路數據轉換為 PyTorch Geometric 圖格式
  - 支援 32 維 cell embedding（替代原 one-hot 編碼）
  - 生成 23 維特徵向量（22 基礎特徵 + 1 cell_id）

- **`gnn_api.py`** - GNN 模型接口
  - `ConfigurableGATEncoder` - 支援 embedding 的 GAT 編碼器
  - 模型載入/保存功能
  - 圖嵌入生成接口

- **`inference.py`** - 推論模組
  - 單電路/批量推論
  - 模型兼容性處理
  - 嵌入向量導出

### 🎯 訓練相關
- **`config_train_dgi.py`** - DGI 訓練腳本
  - 支援新的 embedding 架構
  - 可配置的 GAT 層數、隱藏維度等
  - 驗證探頭功能

- **`retrain_gnn.sh`** - 快速重新訓練腳本
  - 一鍵重新訓練 embedding 模型
  - 自動清理舊模型檔案

### 📊 數據檔案
- **`cell_groups.json`** - ASAP7 cell 分組數據
  - 220 個功能組
  - 852 個獨特 cells
  - 支援分組或個別 embedding

- **`cell_id_mapping.json`** - Cell ID 映射表
  - cell_name → id 的映射
  - 用於 embedding 層索引
  - 自動從 cell_groups.json 生成

### 🏃‍♂️ 測試工具
- **`test_embedding.py`** - Embedding 架構測試
  - 驗證 cell mapping 載入
  - 測試圖構建功能
  - 檢查模型前向傳播

### 📈 訓練輸出
- **`encoder.pt`** - 訓練好的模型權重
- **`encoder_meta.json`** - 模型元數據
- **`best_encoder_all_by_probe.pt`** - 最佳驗證模型
- **`loss_*.png`** - 訓練曲線圖

## 🚀 快速開始

### 1. 測試新架構
```bash
cd /root/ruan_workspace/ic_cad/gnn
python test_embedding.py
```

### 2. 重新訓練模型
```bash
# 方法1：使用腳本
./retrain_gnn.sh

# 方法2：手動訓練
python config_train_dgi.py --test-only --epochs 50 --save-meta
```

### 3. 使用訓練好的模型
```python
from gnn.gnn_api import load_encoder

# 載入模型
encoder, meta = load_encoder('encoder.pt', 'encoder_meta.json')

# 生成嵌入
embeddings = get_embeddings('c17')
```

## 🔄 架構升級

### 舊架構 → 新架構
- **輸入維度**: 84 維 → 23 維
- **編碼方式**: One-hot (852 維) → Embedding (32 維)
- **記憶體效率**: 13 倍改善
- **語義表達**: 學習式 embedding 捕捉 cell 關係

### 特徵分解
```
輸入: 23 維
├── 基礎特徵: 22 維
│   ├── 物理特徵: 9 維 (area, leakage, 電容統計等)
│   ├── 位置特徵: 6 維 (座標, 相對位置等)
│   ├── 拓撲特徵: 3 維 (度數, fanin, fanout)
│   └── 功能特徵: 3 維 (邏輯門類型, 記憶體, 複雜度)
└── Cell ID: 1 維 → Embedding → 32 維

實際處理: 22 + 32 = 54 維
```

## ⚙️ 配置說明

### 模型參數
- **隱藏維度**: 128
- **GAT 層數**: 3
- **Dropout**: 0.1
- **Learning Rate**: 0.0005
- **Cell Embedding**: 32 維

### 訓練參數
- **Epochs**: 200（完整訓練）/ 50（測試）
- **Batch Size**: 4-8
- **優化器**: AdamW
- **調度器**: ReduceLROnPlateau

## 🔧 維護指南

### 更新 Cell 映射
```bash
# 當添加新 cells 時，重新生成映射
python -c "
from graph_builder import get_or_create_cell_id_mapping
import os
os.remove('cell_id_mapping.json') if os.path.exists('cell_id_mapping.json') else None
get_or_create_cell_id_mapping()
"
```

### 檢查模型兼容性
```bash
python -c "
import torch
model = torch.load('encoder.pt', map_location='cpu')
print('Model keys:', list(model.keys())[:5])
"
```

## 📝 更新記錄

### v2.0 (2025-08-22)
- ✅ 實現 32 維 cell embedding 架構
- ✅ 替換 one-hot 編碼為學習式 embedding
- ✅ 優化記憶體使用（13x 改善）
- ✅ 支援 854 個 ASAP7 cells
- ✅ 向後兼容的模型載入

### v1.0 (之前)
- 原始 one-hot 編碼架構
- 84 維輸入特徵
- 固定 cell type 映射

## 🎯 下一步

1. **RL 整合**: 將新模型整合到強化學習訓練
2. **效能評估**: 比較新舊架構的效果
3. **擴展支援**: 支援更多 technology libraries
4. **優化調參**: 基於實際效果調整 embedding 維度
