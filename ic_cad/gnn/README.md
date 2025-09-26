# GNN 模組說明

本資料夾包含 IC CAD 優化系統的圖神經網路（GNN）相關檔案，實現了基於 32 維 cell embedding 的高效電路表示學習。支援多種對比學習方法（DGI、GRACE）進行無監督節點表示學習。

## 📁 檔案結構

### 🔧 核心模組
- **`graph_builder.py`** - 電路圖構建器
  - 將電路數據轉換為 PyTorch Geometric 圖格式
  - 支援 16 維 cell embedding（替代原 one-hot 編碼）
  - 生成 20 維特徵向量（19 基礎特徵 + 1 cell_id）

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


### 1. 重新訓練模型
```bash
# 方法1：手動訓練 DGI 模型
python config_train_dgi.py --test-only --epochs 50 --save-meta

# 方法2：完整訓練
python config_train_dgi.py --epochs 200 --save-meta --use-proj
```

### 2. 使用訓練好的模型
```python
from gnn.gnn_api import load_encoder, get_embeddings

# 載入模型
encoder, meta = load_encoder('encoder_grace.pt', 'encoder_meta.json')

# 生成嵌入
embeddings = get_embeddings('c17', encoder)
```

### 3. 批量推論
```python
from gnn.gnn_api import get_batch_embeddings

# 批量生成嵌入
circuits = ['c17', 'c432', 'c499']
batch_results = get_batch_embeddings(circuits, encoder)
```

## 🔄 架構升級

### 舊架構 → 新架構
- **輸入維度**: 84 維 → 20 維  
- **編碼方式**: One-hot (852 維) → Embedding (16 維)
- **記憶體效率**: 顯著改善
- **語義表達**: 學習式 embedding 捕捉 cell 關係
- **拓撲特徵**: 移除度數，fanin/fanout 包含 I/O terminals

### 特徵分解
```
輸入: 20 維
├── 基礎特徵: 19 維
│   ├── 物理特徵: 8 維 (area, leakage, 輸入電容統計(3), output_load_cap, num_pins, drive_strength)
│   ├── 位置特徵: 6 維 (座標, 相對位置等)
│   ├── 拓撲特徵: 2 維 (fanin, fanout - 包含terminals)
│   └── 功能特徵: 3 維 (邏輯門類型, 記憶體, 複雜度)
└── Cell ID: 1 維 → Embedding → 8 維

實際處理: 19 + 8 = 27 維
邊特徵: 3 維 (網路大小, 物理距離, 網路重要性)
```

## ⚙️ 配置說明

### 模型參數
- **隱藏維度**: 64（低顯存優化）
- **GAT 層數**: 2
- **注意力頭數**: [2,1] 調度
- **Dropout**: 0.3
- **Learning Rate**: 1e-3
- **Cell Embedding**: 8 維（低顯存優化）

### 訓練參數
- **訓練方法**: DGI（Deep Graph Infomax）對比學習
- **池化方式**: 平均池化（Mean Pooling）
- **Epochs**: 250（完整訓練）/ 10（測試）
- **損失函數**: Binary Cross Entropy
- **優化器**: AdamW (lr=1e-3, weight_decay=5e-4)
- **調度器**: Cosine Annealing + Linear Warmup

### 驗證設置
- **探頭方法**: Link Prediction AUC
- **驗證頻率**: 每 10 epochs
- **早停機制**: 基於 loss 改善 (patience=50)
- **最佳模型保存**: 支援 loss 和 probe 雙重標準

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

### v2.2 (2025-09-26) 🚀 **最新更新**
- ✅ **優化拓撲特徵計算**
  - 移除 degree 特徵，簡化拓撲表示
  - Fanin/Fanout 現在包含 I/O terminals，更準確反映信號流向
  - 特徵維度從 23 維降至 20 維
- ✅ **精簡物理特徵表示**
  - 輸入電容統計從 5 維降至 3 維 (移除 sum 和 min，保留 max, avg, std)
  - 更聚焦於關鍵的電容分布特性
- ✅ **邊特徵優化**
  - 移除冗餘的連接強度特徵，邊特徵從 4 維降至 3 維
  - 保留核心的網路規模、物理距離、網路重要性特徵
- ✅ **低顯存優化**
  - Cell embedding 維度從 32 降至 8 維
  - 隱藏維度從 128 降至 64 維
  - 適配 10GB GPU 環境
- ✅ **DGI 訓練框架更新**
  - 使用平均池化 (Mean Pooling) 進行全域表示
  - 支援梯度累積和記憶體優化
  - 改進的洩漏功率計算（考慮所有輸入條件）

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

### 🔍 **拓撲特徵改進詳解**

#### **舊版拓撲特徵 (3 維)**
- `degree`: 節點連接的網路總數
- `fanin`: 只計算邏輯門之間的輸入連接
- `fanout`: 只計算邏輯門之間的輸出連接

#### **舊版物理特徵 (9 維)**
- 輸入電容統計: sum, max, min, avg, std (5 維)
- 其他物理特徵: area, leakage, output_load_cap, num_pins (4 維)

#### **舊版邊特徵 (4 維)**
- 網路大小, 物理距離, 網路重要性, 連接強度

#### **新版拓撲特徵 (2 維)**
- `fanin`: **包含 I/O terminals** 的所有輸入連接
- `fanout`: **包含 I/O terminals** 的所有輸出連接

#### **新版物理特徵 (8 維)**  
- 輸入電容統計: max, avg, std (3 維) - **移除冗餘的 sum, min**
- 其他物理特徵: area, leakage, output_load_cap, num_pins, drive_strength (5 維)

#### **新版邊特徵 (3 維)**
- 網路大小, 物理距離, 網路重要性 - **移除冗餘的連接強度**

#### **改進效果**
```python
# 範例電路: INPUT_A → NAND1 → OUTPUT_Z
# 舊版計算結果:
INPUT_A:  fanin=0, fanout=0    # terminals 被忽略
NAND1:    fanin=1, fanout=0    # 只計算邏輯門間連接
OUTPUT_Z: fanin=1, fanout=0    # terminals 被忽略

# 新版計算結果:
INPUT_A:  fanin=0, fanout=1    # ✅ 正確反映驅動1個邏輯門
NAND1:    fanin=1, fanout=1    # ✅ 正確反映驅動1個輸出端口  
OUTPUT_Z: fanin=1, fanout=0    # ✅ 正確反映被1個邏輯門驅動
```

**優勢**:
- 🎯 **更準確**: 完整反映 IC 設計中的信號流向
- 💾 **更簡潔**: 移除冗餘特徵，降維度 (node: 23→20維, edge: 4→3維)
- ⚡ **更高效**: 減少記憶體使用和計算複雜度
- 🔧 **更聚焦**: 保留核心特徵，移除統計冗餘 (如 sum 可由 avg×count 推導)

1. **RL 整合**: 將新 DGI 模型整合到強化學習訓練
2. **效能評估**: 比較新舊拓撲特徵對模型效果的影響
3. **超參調優**: 基於低顯存環境調整最佳參數組合
4. **擴展支援**: 支援更多 technology libraries  
5. **模型壓縮**: 研究進一步的記憶體優化技術
