"""GNN API for RL Integration
================================
提供簡潔可重複使用的函式，讓強化學習 (RL) 環境直接取得：
1. 已建構好的圖資料 (features, edge_index, edge_attr, node_names)
2. 對應的節點嵌入 (embeddings)
3. 模型與 meta 參數 (hidden_dim, benchmarks, feature_dim ...)

使用範例：
    from gnn_api import load_encoder, get_graph_data, get_embeddings
    encoder, meta = load_encoder('encoder.pt', 'encoder_meta.json')
    graph = get_graph_data('c17')           # graph 為 PyG Data 物件
    emb = get_embeddings('c17', encoder)    # torch.Tensor [num_nodes, hidden_dim]

批次：
    batch = get_batch_embeddings(['c17','c432'], encoder)
    # batch['c17']['embeddings'] -> tensor

注意：encoder 為 torch.nn.Module，RL 可在回合開始時載入一次後重複使用。
"""
from __future__ import annotations
import os
import json
import torch
from typing import Dict, List, Optional, Tuple
from graph_builder import build_graph_from_case, collect_all_cell_types

# === 與 inference.py 一致的 Encoder 定義 ===
class ConfigurableGATEncoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_layers: int = 3, heads_schedule=None, dropout: float = 0.1, 
                 cell_embedding_dim: int = 32, num_cells: int = 852):
        super().__init__()
        from torch_geometric.nn import GATConv
        
        # Cell embedding layer - num_cells 已經包含了 terminal_NI 和 unknown
        self.cell_embedding = torch.nn.Embedding(num_cells, cell_embedding_dim)
        
        # 計算實際輸入維度：原始特徵(22) + cell embedding(32) = 54
        actual_in_channels = in_channels - 1 + cell_embedding_dim  # -1 移除 cell_id，+32 加入 embedding
        
        if heads_schedule is None:
            heads_schedule = [4, 2, 1]
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        in_dim = actual_in_channels
        for i in range(num_layers):
            heads = heads_schedule[i] if i < len(heads_schedule) else 1
            if i == num_layers - 1:
                out_dim = out_channels
                concat = True
            else:
                out_dim = 2 * out_channels
                concat = True
            self.convs.append(GATConv(in_dim, out_dim, heads=heads, concat=concat, dropout=dropout))
            if i < num_layers - 1:
                self.bns.append(torch.nn.BatchNorm1d(out_dim * heads))
            in_dim = out_dim * heads
        self.dropout = dropout

    def forward(self, x, edge_index):
        import torch.nn.functional as F
        
        # 分離 cell_id 和其他特徵
        base_features = x[:, :-1]  # 前22個特徵
        cell_ids = x[:, -1].long()  # 最後一個是 cell_id
        
        # Cell embedding
        cell_embeds = self.cell_embedding(cell_ids)
        
        # 拼接基礎特徵和 cell embedding
        x = torch.cat([base_features, cell_embeds], dim=1)
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = self.bns[i](x)
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

# === 內部快取 ===
_GLOBAL = {
    'cell_types': None,
    'graphs': {},  # circuit_name -> Data
}

# 允許透過環境變數覆寫設計根目錄，預設沿用既有路徑。
ROOT_DIR = os.environ.get('GNN_DESIGN_ROOT', '/root/ruan_workspace/gtlvl_design')

# === 載入模型與 meta ===

def load_encoder(model_path: str = 'encoder.pt', meta_path: Optional[str] = 'encoder_meta.json') -> Tuple[torch.nn.Module, Dict]:
    meta = {}
    if meta_path and os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
    else:
        # 嘗試推測 hidden_dim 只能靠外部提供，因此建議一定要有 meta
        print('⚠️ 未找到 meta 檔，使用預設 hidden_dim=128,num_layers=3 (請確認)')
        meta = {'hidden_dim': 128, 'num_layers': 3, 'dropout': 0.1, 'feature_dim': 23}
    
    hidden_dim = meta.get('hidden_dim', 128)
    num_layers = meta.get('num_layers', 3)
    dropout = meta.get('dropout', 0.1)
    heads_schedule = meta.get('heads_schedule')  # 允許為 None 時採預設
    feature_dim = meta.get('feature_dim', 23)  # 預設使用訓練時的特徵維度（23 = 22 基礎 + 1 cell_id）
    
    # 獲取 cell 數量用於 embedding
    try:
        from graph_builder import get_or_create_cell_id_mapping
        cell_to_id = get_or_create_cell_id_mapping()
        num_cells = len(cell_to_id)
    except:
        num_cells = 854  # 預設值 (852 + terminal_NI + unknown)

    print(f'📚 載入模型：feature_dim={feature_dim}, hidden_dim={hidden_dim}, num_layers={num_layers}, num_cells={num_cells}')
    
    # 使用 meta 中的 feature_dim 構建編碼器
    encoder = ConfigurableGATEncoder(feature_dim, hidden_dim, num_layers=num_layers, heads_schedule=heads_schedule, 
                                   dropout=dropout, num_cells=num_cells)
    
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        try:
            encoder.load_state_dict(state_dict)
            print('✓ 模型載入成功 (strict)')
        except Exception as e:
            print(f'⚠️ 嚴格載入失敗，改用 strict=False: {e}')
            encoder.load_state_dict(state_dict, strict=False)
            print('✓ 模型載入成功 (non-strict)')
    except Exception as e:
        raise RuntimeError(f'載入模型失敗: {e}')
    
    encoder.eval()
    return encoder, meta

# === 取得或構建圖 ===

def _ensure_global_cell_types(benchmarks: List[str]):
    if _GLOBAL['cell_types'] is None:
        # 🔧 修復：使用靜態 cell_id_mapping 確保與 GNN 模型一致
        from graph_builder import get_or_create_cell_id_mapping
        cell_mapping = get_or_create_cell_id_mapping()
        _GLOBAL['cell_types'] = sorted(cell_mapping.keys())
        print(f"🔧 使用靜態 cell mapping: {len(_GLOBAL['cell_types'])} 個 cell 類型")
        
        # 原始動態收集邏輯（已註解）
        # full_benchmarks = ["c17", "c432", "c499", "c880", "c1355", "c1908", "c2670", "c3540", "c5315", "c6288", "c7552"]
        # _GLOBAL['cell_types'] = collect_all_cell_types(full_benchmarks, ROOT_DIR, ROOT_DIR)
    
    return _GLOBAL['cell_types']


def get_graph_data(circuit_name: str) -> torch.nn.Module:
    if circuit_name in _GLOBAL['graphs']:
        return _GLOBAL['graphs'][circuit_name]
    _ensure_global_cell_types([circuit_name])
    case_dir = os.path.join(ROOT_DIR, circuit_name, 'bookshelf_run', 'output', circuit_name)
    print('123', case_dir)
    nodes_file = os.path.join(case_dir, f"{circuit_name}.nodes")
    if not os.path.exists(nodes_file):
        raise FileNotFoundError(f'找不到 nodes 檔案: {nodes_file}')
    data = build_graph_from_case(case_dir, verilog_root=ROOT_DIR, global_cell_types=_GLOBAL['cell_types'])
    print('1234', case_dir)
    _GLOBAL['graphs'][circuit_name] = data
    return data

# === 取得 embeddings ===

def get_embeddings(circuit_name: str, encoder: torch.nn.Module) -> torch.Tensor:
    # Dummy 模式：快速繞過圖構建（例如僅測 RL pipeline）
    if os.environ.get('USE_DUMMY_GNN') == '1':
        hidden = getattr(encoder, 'convs', [None])[0].out_channels if hasattr(encoder, 'convs') and len(encoder.convs) > 0 else 128
        num_nodes = int(os.environ.get('DUMMY_GNN_NUM_NODES', '50'))
        return torch.zeros((num_nodes, hidden))
    print(circuit_name)
    try:
        data = get_graph_data(circuit_name)
        print(circuit_name)
        
        # 檢查特徵維度是否匹配模型期望
        model_input_dim = 23  # 預設使用訓練時的維度（23 = 22 基礎 + 1 cell_id）
        try:
            # 嘗試從模型獲取輸入維度
            first_conv = encoder.convs[0]
            if hasattr(first_conv, 'lin_src'):
                model_input_dim = first_conv.lin_src.in_features
            elif hasattr(first_conv, 'lin_l'):  # GAT 的另一種實現
                model_input_dim = first_conv.lin_l.in_features
            elif hasattr(first_conv, 'lin'):
                model_input_dim = first_conv.lin.in_features
            elif hasattr(first_conv, 'in_channels'):
                model_input_dim = first_conv.in_channels
        except:
            # 如果無法獲取，使用元數據中的值
            pass
            
        current_feature_dim = data.x.shape[1]
        
        if current_feature_dim != model_input_dim:
            print(f'⚠️ 特徵維度不匹配：當前={current_feature_dim}, 模型期望={model_input_dim}')
            # 調整特徵維度以匹配模型
            if current_feature_dim < model_input_dim:
                # 填充零到目標維度
                padding = torch.zeros((data.x.shape[0], model_input_dim - current_feature_dim))
                data.x = torch.cat([data.x, padding], dim=1)
                print(f'✓ 特徵維度已填充至 {model_input_dim}')
            else:
                # 截斷到目標維度
                data.x = data.x[:, :model_input_dim]
                print(f'✓ 特徵維度已截斷至 {model_input_dim}')
        
        with torch.no_grad():
            emb = encoder(data.x, data.edge_index)
        return emb
    except Exception as e:
        # 最後退路：回傳隨機向量避免整體流程中斷（並提示）
        print(f'⚠️ 取得 {circuit_name} 圖或嵌入失敗，使用隨機嵌入: {e}')
        # 計算正確的輸出維度
        hidden = 128  # 預設值
        try:
            if hasattr(encoder, 'convs') and len(encoder.convs) > 0:
                last_conv = encoder.convs[-1]
                if hasattr(last_conv, 'out_channels'):
                    # 對於最後一層 GAT，如果 concat=True，需要乘以 heads
                    if hasattr(last_conv, 'heads'):
                        hidden = last_conv.out_channels * last_conv.heads
                    else:
                        hidden = last_conv.out_channels
        except:
            pass
        
        num_nodes = 32
        return torch.randn((num_nodes, hidden))

# 批次

def get_batch_embeddings(circuits: List[str], encoder: torch.nn.Module) -> Dict[str, Dict]:
    out = {}
    for c in circuits:
        try:
            emb = get_embeddings(c, encoder)
            out[c] = {
                'embeddings': emb,
                'num_nodes': emb.shape[0],
                'dim': emb.shape[1],
                'node_names': _GLOBAL['graphs'][c].node_names
            }
        except Exception as e:
            out[c] = {'error': str(e)}
    return out

# === 導出到檔案 (npy / pt) ===

def export_embeddings(circuit_name: str, encoder: torch.nn.Module, out_dir: str, formats=('pt', 'npy')) -> str:
    os.makedirs(out_dir, exist_ok=True)
    emb = get_embeddings(circuit_name, encoder)
    node_names = _GLOBAL['graphs'][circuit_name].node_names
    base = os.path.join(out_dir, circuit_name)
    if 'pt' in formats:
        torch.save({'embeddings': emb, 'node_names': node_names}, base + '_emb.pt')
    if 'npy' in formats:
        try:
            import numpy as np
            np.save(base + '_emb.npy', emb.numpy())
            with open(base + '_nodes.txt', 'w') as f:
                f.write('\n'.join(node_names))
        except ImportError:
            print('⚠️ numpy 未安裝，跳過 .npy 匯出')
    return base

__all__ = [
    'load_encoder', 'get_graph_data', 'get_embeddings', 'get_batch_embeddings', 'export_embeddings', 'ConfigurableGATEncoder'
]
