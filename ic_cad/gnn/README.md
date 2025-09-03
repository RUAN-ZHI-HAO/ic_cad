# GNN 模組說明

本資料夾包含 IC CAD 優化系統的圖神經網路（GNN）相關檔案，實現了基於 32 維 cell embedding 的高效電路表示學習。支援多種對比學習方法（DGI、GRACE）進行無監督節點表示學習。

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
  - 支援批量推論和特徵維度自動調整

- **`inference.py`** - 推論模組
  - 單電路/批量推論
  - 模型兼容性處理
  - 嵌入向量導出
  - 支援 dummy 模式用於快速測試

### 🎯 訓練相關
- **`config_train_dgi.py`** - 對比學習訓練腳本（支援 DGI 和 GRACE）
  - 支援新的 embedding 架構
  - 可配置的 GAT 層數、隱藏維度等
  - GRACE 節點級對比學習
  - 驗證探頭功能和自動調參
  - 支援投影層和多種損失函數

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

- **`check_model.py`** - 模型檢查工具
  - 驗證模型完整性
  - 檢查權重和架構

### 📈 訓練輸出
- **`encoder_grace.pt`** - GRACE 訓練的模型權重
- **`encoder_grace_proj.pt`** - 包含投影層的 GRACE 模型
- **`encoder_meta.json`** - 模型元數據
- **`best_encoder_all_by_probe.pt`** - 最佳驗證模型
- **`best_proj_all_by_probe.pt`** - 最佳投影層權重
- **`loss_*.png`** - 訓練曲線圖
- **`train.log`** - 訓練日誌檔案

## 🚀 快速開始

### 1. 測試新架構
```bash
cd /root/ruan_workspace/ic_cad/gnn
python test_embedding.py
```

### 2. 重新訓練模型
```bash
# 方法1：手動訓練 GRACE 模型
python config_train_dgi.py --test-only --epochs 50 --save-meta

# 方法2：完整訓練
python config_train_dgi.py --epochs 200 --save-meta --use-proj
```

### 3. 使用訓練好的模型
```python
from gnn.gnn_api import load_encoder, get_embeddings

# 載入模型
encoder, meta = load_encoder('encoder_grace.pt', 'encoder_meta.json')

# 生成嵌入
embeddings = get_embeddings('c17', encoder)
```

### 4. 批量推論
```python
from gnn.gnn_api import get_batch_embeddings

# 批量生成嵌入
circuits = ['c17', 'c432', 'c499']
batch_results = get_batch_embeddings(circuits, encoder)
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
- **注意力頭數**: 動態調度（8→4→1）
- **Dropout**: 0.1
- **Learning Rate**: 0.0005
- **Cell Embedding**: 32 維

### 訓練參數
- **訓練方法**: GRACE（節點級對比學習）
- **Epochs**: 200（完整訓練）/ 50（測試）
- **溫度參數 τ**: 0.2
- **特徵擾動**: 0.1
- **邊丟棄**: 0.1
- **優化器**: AdamW
- **調度器**: ReduceLROnPlateau

### 驗證設置
- **探頭電路**: s1488（獨立於訓練）
- **驗證指標**: AUC
- **最佳模型保存**: 基於驗證 AUC

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
# 檢查模型結構
python -c "
import torch
model = torch.load('encoder_grace.pt', map_location='cpu')
print('Model keys:', list(model.keys())[:5])
"

# 使用檢查工具
python check_model.py
```

### 使用 Dummy 模式
```bash
# 快速測試 RL pipeline 時跳過圖構建
export USE_DUMMY_GNN=1
export DUMMY_GNN_NUM_NODES=50
```

## 📝 更新記錄

### v2.1 (2025-09-02)
- ✅ 實現 GRACE 節點級對比學習訓練
- ✅ 支援投影層和多種損失函數
- ✅ 動態注意力頭數調度
- ✅ 新增驗證探頭機制
- ✅ 增強模型兼容性處理
- ✅ 支援 Dummy 模式快速測試
- ✅ 完善批量推論功能

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

1. **RL 整合**: 將新 GRACE 模型整合到強化學習訓練
2. **效能評估**: 比較 DGI vs GRACE 架構的效果
3. **超參調優**: 基於驗證結果調整溫度參數和擾動強度
4. **擴展支援**: 支援更多 technology libraries
5. **模型壓縮**: 研究知識蒸餾等壓縮技術
