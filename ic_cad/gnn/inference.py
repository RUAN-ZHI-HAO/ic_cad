#!/usr/bin/env python3
"""
GNN 推論專用腳本
=================

使用方法:
    # 基本推論
    python inference.py --circuit c17
    
    # 帶分析的推論
    python inference.py --circuit c17 --analysis --visualize
    
    # 批次推論
    python inference.py --batch c17,c432,c499 --output results/
    
    # 自定義模型
    python inference.py --circuit s27 --model my_encoder.pt

作者: IC CAD Team
版本: v1.0
"""

import argparse
import torch
import os
import sys
import json
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# 添加必要的路徑
sys.path.append("../parser")

try:
    from graph_builder import build_graph_from_case, collect_all_cell_types
except ImportError as e:
    print(f"❌ 導入錯誤: {e}")
    print("請確保您在正確的目錄中，並且相關模組存在")
    sys.exit(1)

# === 與 config_train_dgi.py 一致的 Encoder 定義 ===
class ConfigurableGATEncoder(torch.nn.Module):
    """可配置 GAT 編碼器（附 cell embedding，concat=False 省顯存）"""
    def __init__(self, in_channels, out_channels, num_layers=2, heads_schedule=None, dropout=0.1,
                 cell_embedding_dim=16, num_cells=854):
        super().__init__()
        from torch_geometric.nn import GATConv
        self.cell_embedding = torch.nn.Embedding(num_cells, cell_embedding_dim)
        actual_in_channels = in_channels - 1 + cell_embedding_dim  # -1: 去除 cell_id

        # heads 配置
        if not heads_schedule:
            if num_layers == 1:
                heads_schedule = [1]
            elif num_layers == 2:
                heads_schedule = [2, 1]
            elif num_layers == 3:
                heads_schedule = [2, 2, 1]  # 更保守的配置
            else:
                heads_schedule = [2] * (num_layers - 1) + [1]

        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        
        in_dim = actual_in_channels
        for i in range(num_layers):
            heads = heads_schedule[i] if i < len(heads_schedule) else 1
            self.convs.append(GATConv(in_dim, out_channels, heads=heads, concat=False, dropout=dropout))
            if i < num_layers - 1:
                self.bns.append(torch.nn.LayerNorm(out_channels))
            in_dim = out_channels
        self.dropout = dropout

    def forward(self, x, edge_index):
        import torch.nn.functional as F
        base = x[:, :-1]
        cell_ids = x[:, -1].long()
        cell_emb = self.cell_embedding(cell_ids)
        x = torch.cat([base, cell_emb], dim=1)
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = self.bns[i](x)
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x
        
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

class GNNInference:
    """GNN 推論類"""
    
    def __init__(self, model_path: str = "encoder.pt"):
        """
        初始化推論器
        
        Args:
            model_path: 訓練好的模型路徑
        """
        self.model_path = model_path
        self.encoder = None
        self.global_cell_types = None
        
        # 設定數據路徑
        self.root_dir = "/root/ruan_workspace/gtlvl_design"
        self.verilog_root = "/root/ruan_workspace/gtlvl_design"
        
        # 檢查模型文件
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        print(f"🎯 GNN 推論器初始化完成")
        print(f"   模型路徑: {model_path}")
    
    def _prepare_global_cell_types(self):
        """準備全局cell types"""
        if self.global_cell_types is None:
            print("🔍 收集全局 cell types...")
            benchmarks = ["c17", "c432", "c499", "c880", "c1355", "c1908", "c2670", "c3540", "c5315", "c6288", "s5378"]
            self.global_cell_types = collect_all_cell_types(
                benchmarks, self.root_dir, self.verilog_root
            )
            print(f"   收集到 {len(self.global_cell_types)} 種 cell types")
    
    def _load_meta(self):
        meta_path_candidates = [
            'encoder_meta.json',
            os.path.splitext(self.model_path)[0] + '_meta.json'
        ]
        for p in meta_path_candidates:
            if os.path.exists(p):
                try:
                    with open(p, 'r') as f:
                        meta = json.load(f)
                    print(f"📝 已載入模型中繼資料: {p}")
                    return meta
                except Exception as e:
                    print(f"⚠️  讀取 meta 失敗 {p}: {e}")
        print("⚠️  找不到 meta 檔，將嘗試以推測方式載入模型")
        return None

    def load_model(self, input_dim: int, hidden_dim: int = 128):
        """載入模型（自動判斷新舊 encoder 架構）"""
        meta = self._load_meta()
        num_layers = 3
        dropout = 0.1
        if meta:
            hidden_dim = meta.get('hidden_dim', hidden_dim)
            num_layers = meta.get('num_layers', num_layers)
            dropout = meta.get('dropout', dropout)
        
        # 獲取 cell 數量用於 embedding
        try:
            from graph_builder import get_or_create_cell_id_mapping
            cell_to_id = get_or_create_cell_id_mapping()
            num_cells = len(cell_to_id)
        except:
            num_cells = 854  # 預設值
            
        self.encoder = ConfigurableGATEncoder(input_dim, hidden_dim, num_layers=num_layers, 
                                            dropout=dropout, num_cells=num_cells)
        encoder_type = 'Configurable'
        try:
            state_dict = torch.load(self.model_path, map_location='cpu')
            self.encoder.load_state_dict(state_dict)
            self.encoder.eval()
            print(f"✅ 模型載入成功: 類型 {encoder_type}, 輸入維度 {input_dim}, 隱藏維度 {hidden_dim}")
        except Exception as e:
            raise RuntimeError(f"模型載入失敗: {e}")
    
    def inference_single(self, circuit_name: str) -> Tuple[torch.Tensor, List[str]]:
        """
        對單一電路進行推論
        
        Args:
            circuit_name: 電路名稱
            
        Returns:
            (embeddings, node_names): 節點嵌入和節點名稱
        """
        print(f"\n🔮 開始推論電路: {circuit_name}")
        
        # 準備全局cell types
        self._prepare_global_cell_types()
        
        # 構建圖數據
        case_path = os.path.join(self.root_dir, circuit_name, "bookshelf_run", "output", circuit_name)
        
        if not os.path.exists(case_path + ".nodes"):
            raise FileNotFoundError(f"電路文件不存在: {case_path}.nodes")
        
        graph_data = build_graph_from_case(
            case_path, 
            verilog_root=self.verilog_root, 
            global_cell_types=self.global_cell_types
        )
        
        print(f"   圖結構: {graph_data.x.shape[0]} 節點, {graph_data.x.shape[1]} 特徵, {graph_data.edge_index.shape[1]} 邊")
        
        # 載入模型 (如果還沒載入)
        if self.encoder is None:
            input_dim = graph_data.num_node_features
            self.load_model(input_dim)
        
        # 推論
        with torch.no_grad():
            embeddings = self.encoder(graph_data.x, graph_data.edge_index)
        
        print(f"   推論完成: 輸出形狀 {embeddings.shape}")
        
        return embeddings, graph_data.node_names
    
    def inference_batch(self, circuit_list: List[str]) -> Dict[str, Tuple[torch.Tensor, List[str]]]:
        """批次推論"""
        print(f"\n🎪 開始批次推論 {len(circuit_list)} 個電路")
        
        results = {}
        success_count = 0
        
        for i, circuit_name in enumerate(circuit_list, 1):
            try:
                print(f"\n[{i}/{len(circuit_list)}]", end=" ")
                embeddings, node_names = self.inference_single(circuit_name)
                results[circuit_name] = (embeddings, node_names)
                success_count += 1
                print(f"✅ {circuit_name} 成功")
            except Exception as e:
                print(f"❌ {circuit_name} 失敗: {str(e)}")
                results[circuit_name] = None
        
        print(f"\n🏆 批次推論完成: {success_count}/{len(circuit_list)} 成功")
        return results

class ResultAnalyzer:
    """結果分析器"""
    
    @staticmethod
    def analyze_importance(embeddings: torch.Tensor, node_names: List[str], top_k: int = 10):
        """節點重要性分析"""
        importance_scores = torch.norm(embeddings, dim=1)
        top_indices = torch.topk(importance_scores, k=min(top_k, len(node_names))).indices
        
        print(f"\n🏆 Top {min(top_k, len(node_names))} 重要節點:")
        print("-" * 60)
        print(f"{'排名':<4} {'節點名稱':<25} {'重要性分數':<12} {'特徵範數':<10}")
        print("-" * 60)
        
        for i, idx in enumerate(top_indices):
            node_name = node_names[idx.item()]
            score = importance_scores[idx.item()].item()
            norm = torch.norm(embeddings[idx]).item()
            print(f"{i+1:<4} {node_name:<25} {score:<12.4f} {norm:<10.4f}")
        
        return importance_scores
    
    @staticmethod
    def find_similar_nodes(embeddings: torch.Tensor, node_names: List[str], 
                          target_node: str, top_k: int = 5):
        """找相似節點"""
        if target_node not in node_names:
            print(f"❌ 節點 '{target_node}' 不存在")
            return
        
        target_idx = node_names.index(target_node)
        target_embedding = embeddings[target_idx].unsqueeze(0)
        
        similarities = torch.cosine_similarity(target_embedding, embeddings, dim=1)
        similarities[target_idx] = -1  # 排除自己
        
        top_indices = torch.topk(similarities, k=min(top_k, len(node_names)-1)).indices
        
        print(f"\n🔍 與節點 '{target_node}' 最相似的 {min(top_k, len(node_names)-1)} 個節點:")
        print("-" * 50)
        print(f"{'排名':<4} {'節點名稱':<25} {'相似性':<10}")
        print("-" * 50)
        
        for i, idx in enumerate(top_indices):
            node_name = node_names[idx.item()]
            similarity = similarities[idx.item()].item()
            print(f"{i+1:<4} {node_name:<25} {similarity:<10.4f}")
    
    @staticmethod
    def visualize_embeddings(embeddings: torch.Tensor, node_names: List[str], 
                           circuit_name: str, save_path: Optional[str] = None):
        """視覺化嵌入"""
        try:
            from sklearn.manifold import TSNE
            import matplotlib.pyplot as plt
            import numpy as np
            
            print(f"\n📊 正在生成 {circuit_name} 的嵌入視覺化...")
            
            # t-SNE 降維
            if len(embeddings) < 4:
                print("❌ 節點數量太少，無法進行 t-SNE 降維")
                return
                
            perplexity = min(30, len(embeddings) - 1)
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            embeddings_2d = tsne.fit_transform(embeddings.numpy())
            
            # 計算重要性用於著色
            importance_scores = torch.norm(embeddings, dim=1).numpy()
            
            # 繪圖
            plt.figure(figsize=(14, 10))
            scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                c=importance_scores, cmap='viridis', 
                                alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
            
            # 標註前10個重要節點
            top_10_indices = np.argsort(-importance_scores)[:min(10, len(node_names))]
            
            for i, idx in enumerate(top_10_indices):
                x, y = embeddings_2d[idx]
                plt.annotate(f"{i+1}:{node_names[idx]}", (x, y), 
                           xytext=(10, 10), textcoords='offset points', 
                           fontsize=9, alpha=0.8, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
            
            plt.colorbar(scatter, label='Node Importance Score')
            plt.title(f'Node Embeddings Visualization - {circuit_name}\n'
                     f'Total Nodes: {len(node_names)}, Embedding Dim: {embeddings.shape[1]}')
            plt.xlabel('t-SNE Dimension 1')
            plt.ylabel('t-SNE Dimension 2')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            if save_path is None:
                save_path = f"{circuit_name}_embeddings_visualization.png"
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"✅ 視覺化圖表已保存: {save_path}")
            
        except ImportError:
            print("❌ 需要安裝相關套件: pip install scikit-learn matplotlib")
        except Exception as e:
            print(f"❌ 視覺化失敗: {e}")

def save_results(results: Dict, output_path: str, circuit_name: str = None):
    """保存推論結果"""
    os.makedirs(output_path, exist_ok=True)
    
    if circuit_name and circuit_name in results:
        # 保存單個電路結果
        embeddings, node_names = results[circuit_name]
        file_path = os.path.join(output_path, f"{circuit_name}_inference_result.pt")
        
        torch.save({
            'circuit_name': circuit_name,
            'embeddings': embeddings,
            'node_names': node_names,
            'embedding_dim': embeddings.shape[1],
            'num_nodes': len(node_names)
        }, file_path)
        
        print(f"✅ {circuit_name} 結果已保存: {file_path}")
    else:
        # 保存所有結果
        for circuit_name, result in results.items():
            if result is not None:
                embeddings, node_names = result
                file_path = os.path.join(output_path, f"{circuit_name}_inference_result.pt")
                
                torch.save({
                    'circuit_name': circuit_name,
                    'embeddings': embeddings,
                    'node_names': node_names,
                    'embedding_dim': embeddings.shape[1],
                    'num_nodes': len(node_names)
                }, file_path)
                
                print(f"✅ {circuit_name} 結果已保存: {file_path}")

def main():
    parser = argparse.ArgumentParser(
        description='GNN 模型推論工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  # 基本推論
  python inference.py --circuit c17
  
  # 帶完整分析
  python inference.py --circuit c17 --analysis --visualize
  
  # 批次推論
  python inference.py --batch c17,c432,c499 --output results/
  
  # 自定義模型和參數
  python inference.py --circuit s27 --model my_encoder.pt --hidden-dim 256
  
  # 相似性分析
  python inference.py --circuit c17 --analysis --similar-to c17_n1
        """
    )
    
    # 基本參數
    parser.add_argument('--circuit', type=str, help='電路名稱 (如: c17, s27)')
    parser.add_argument('--batch', type=str, help='批次電路 (逗號分隔, 如: c17,c432,c499)')
    parser.add_argument('--model', type=str, default='encoder.pt', help='模型文件路徑')
    parser.add_argument('--hidden-dim', type=int, default=128, help='隱藏維度 (需與訓練時一致)')
    
    # 分析參數
    parser.add_argument('--analysis', action='store_true', help='進行節點重要性分析')
    parser.add_argument('--visualize', action='store_true', help='生成視覺化圖表')
    parser.add_argument('--top-k', type=int, default=10, help='顯示前k個重要節點')
    parser.add_argument('--similar-to', type=str, help='分析與指定節點的相似性')
    
    # 輸出參數
    parser.add_argument('--output', type=str, help='輸出目錄路徑')
    parser.add_argument('--save-viz', type=str, help='視覺化圖片保存路徑')
    
    args = parser.parse_args()
    
    # 參數檢查
    if not args.circuit and not args.batch:
        parser.error("請指定 --circuit 或 --batch 參數")
    
    if args.circuit and args.batch:
        parser.error("--circuit 和 --batch 不能同時使用")
    
    try:
        # 初始化推論器
        inferencer = GNNInference(args.model)
        analyzer = ResultAnalyzer()
        
        # 執行推論
        if args.circuit:
            # 單電路推論
            embeddings, node_names = inferencer.inference_single(args.circuit)
            results = {args.circuit: (embeddings, node_names)}
            
            # 分析
            if args.analysis:
                analyzer.analyze_importance(embeddings, node_names, args.top_k)
                
                if args.similar_to:
                    analyzer.find_similar_nodes(embeddings, node_names, args.similar_to)
            
            # 視覺化
            if args.visualize:
                save_path = args.save_viz or f"{args.circuit}_embeddings.png"
                analyzer.visualize_embeddings(embeddings, node_names, args.circuit, save_path)
        
        elif args.batch:
            # 批次推論
            circuit_list = [c.strip() for c in args.batch.split(',')]
            results = inferencer.inference_batch(circuit_list)
            
            # 批次分析
            if args.analysis:
                for circuit_name, result in results.items():
                    if result is not None:
                        embeddings, node_names = result
                        print(f"\n{'='*20} {circuit_name} 分析 {'='*20}")
                        analyzer.analyze_importance(embeddings, node_names, args.top_k)
            
            # 批次視覺化
            if args.visualize:
                for circuit_name, result in results.items():
                    if result is not None:
                        embeddings, node_names = result
                        save_path = f"{circuit_name}_embeddings.png"
                        analyzer.visualize_embeddings(embeddings, node_names, circuit_name, save_path)
        
        # 保存結果
        if args.output:
            save_results(results, args.output, args.circuit)
        
        print(f"\n🎉 推論完成!")
        
    except Exception as e:
        print(f"❌ 推論失敗: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
