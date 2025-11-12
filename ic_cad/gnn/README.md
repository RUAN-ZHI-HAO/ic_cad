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
- **`config_train_dgi.py`** - DGI 訓練腳本（針對低顯存優化）
  - 支援新的 embedding 架構
  - 可配置的 GAT 層數、隱藏維度等
  - 驗證探頭功能和自動調參
  - 支援投影層和多種損失函數
  - 適配 10GB GPU 環境：梯度累積、batch_size=1、hidden_dim=64、cell_embed_dim=16

### 📊 數據檔案
- **`cell_groups.json`** - ASAP7 cell 分組數據
  - 220 個功能組
  - 852 個獨特 cells
  - 支援分組或個別 embedding

- **`cell_id_mapping.json`** - Cell ID 映射表
  - cell_name → id 的映射
  - 用於 embedding 層索引
  - 自動從 cell_groups.json 生成

### 📈 訓練輸出
- **`encoder_dgi.pt`** - DGI 訓練的模型權重（低顯存優化版本）
- **`encoder_dgi_proj.pt`** - 包含投影層的 DGI 模型
- **`encoder_meta.json`** - 模型元數據
- **`best_encoder_all_by_probe.pt`** - 最佳驗證模型
- **`best_proj_all_by_probe.pt`** - 最佳投影層權重
- **`loss_*.png`** - 訓練曲線圖
- **`train.log`** - 訓練日誌檔案（如有生成）

## 🚀 快速開始

### 1. 訓練 DGI 模型（低顯存優化版本）
```bash
# 快速測試（10 epochs）
python config_train_dgi.py --test-only --epochs 10 --save-meta

# 完整訓練（250 epochs，針對 10GB GPU 優化）
python config_train_dgi.py --epochs 250 --save-meta --use-proj \
    --hidden-dim 64 --cell-embed-dim 16 --batch-size 1 \
    --gradient-accumulation 8 --lr 3e-4
```

### 2. 使用訓練好的模型
```python
from gnn.gnn_api import load_encoder, get_embeddings

# 載入模型
encoder, meta = load_encoder('encoder_dgi.pt', 'encoder_meta.json')

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

### 模型參數（針對低顯存優化）
- **隱藏維度**: 64（低顯存優化，從 128 降低）
- **GAT 層數**: 2
- **注意力頭數**: [1,1] 調度（低顯存優化）
- **Dropout**: 0.1（降低避免過度正則化）
- **Learning Rate**: 3e-4（降低避免過度擬合）
- **Cell Embedding**: 16 維（低顯存優化，從 32 降低）

### 訓練參數（適配 10GB GPU）
- **訓練方法**: DGI（Deep Graph Infomax）對比學習
- **池化方式**: 平均池化（Mean Pooling）
- **Epochs**: 250（完整訓練）/ 10（測試）
- **Batch Size**: 1（強制設為 1 以節省顯存）
- **梯度累積**: 8 步（模擬 batch_size=8）
- **損失函數**: Binary Cross Entropy
- **優化器**: AdamW (lr=3e-4, weight_decay=1e-4)
- **調度器**: Cosine Annealing + Linear Warmup（默認）

### 驗證設置
- **探頭方法**: Link Prediction AUC
- **驗證頻率**: 每 10 epochs
- **早停機制**: 基於 loss 改善 (patience=20)
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

### v2.3 (2025-11-12) 🚀 **最新更新**
- ✅ **低顯存深度優化**
  - Cell embedding 維度從 32 降至 16 維
  - 隱藏維度維持 64 維（已優化）
  - Batch size 強制設為 1，梯度累積增至 8 步
  - 注意力頭數調整為 [1,1]
- ✅ **訓練策略優化**
  - 學習率降至 3e-4，避免過度擬合
  - Dropout 降至 0.1，減少過度正則化
  - Weight decay 降至 1e-4，避免過度約束
  - 早停 patience 調整為 20，適度的提前停止
- ✅ **檔案結構更新**
  - 模型檔案更新為 `encoder_dgi.pt` 和 `encoder_dgi_proj.pt`
  - 移除 GRACE 訓練腳本，專注於 DGI 方法
  - 完善文檔說明，反映當前實際配置

### v2.2 (2025-09-26)
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
- ✅ **改進的洩漏功率計算**
  - 考慮所有輸入條件的洩漏功率

### v2.1 (2025-09-02)
- ✅ 實現 DGI 節點級對比學習訓練
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

1. **RL 整合**: 將新 DGI 模型整合到強化學習訓練
2. **效能評估**: 評估低顯存優化對模型效果的影響
3. **超參調優**: 基於 10GB GPU 環境進一步調整最佳參數組合
4. **擴展支援**: 支援更多 technology libraries  
5. **模型壓縮**: 研究進一步的記憶體優化技術（如量化、剪枝）

## 💡 使用建議

### 針對低顯存環境（10GB GPU）
當前配置已針對 10GB GPU 環境優化：
- ✅ Batch size = 1（強制）
- ✅ 梯度累積 = 8 步（模擬 batch_size=8）
- ✅ Hidden dim = 64
- ✅ Cell embedding = 16 維
- ✅ 注意力頭數 = [1,1]

### 如果有更多顯存（>16GB）
可以嘗試提升性能：
```bash
python config_train_dgi.py --epochs 250 --save-meta --use-proj \
    --hidden-dim 128 --cell-embed-dim 32 --batch-size 2 \
    --gradient-accumulation 4 --heads-schedule "2,1"
```

### 快速驗證
使用測試模式快速驗證訓練流程：
```bash
python config_train_dgi.py --test-only --epochs 10 --save-meta
```
