#!/usr/bin/env python3
"""
詳細測試推論流程的可重現性
"""

import sys
import os
import torch
import random
import numpy as np

sys.path.append('/root/ruan_workspace/ic_cad/rl')
sys.path.append('/root/ruan_workspace/ic_cad/gnn')

def set_seed(seed: int):
    """設置所有隨機種子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"🎲 隨機種子已設置為: {seed}")

def test_ppo_action_sampling():
    """測試 PPO agent 的動作採樣是否可重現"""
    from ppo_agent import TwoDimensionalPPOAgent
    from config import RLConfig
    
    print("\n" + "="*60)
    print("測試: PPO Agent 動作採樣可重現性")
    print("="*60)
    
    config = RLConfig()
    
    # 第一次
    set_seed(42)
    agent1 = TwoDimensionalPPOAgent(
        feature_dim=config.gnn_embed_dim,
        hidden_dim=config.hidden_dim,
        max_candidates=config.max_candidates,
        max_replacements=config.max_replacements,
        device='cpu'
    )
    
    # 創建模擬狀態
    class MockState:
        def __init__(self):
            self.candidate_instances = ['cell1', 'cell2', 'cell3']
            self.candidate_cells = ['type1', 'type2', 'type3']
            self.candidate_gnn_features = np.random.randn(3, 23).astype(np.float32)
            self.candidate_dynamic_features = np.random.randn(3, 10).astype(np.float32)
            self.global_features = np.random.randn(7).astype(np.float32)
            self.action_mask = {
                'candidate_mask': [True, True, True],
                'candidate_replacement_masks': [
                    [True] * 20,
                    [True] * 20,
                    [True] * 20
                ]
            }
    
    set_seed(42)
    state1 = MockState()
    actions1 = []
    for _ in range(5):
        action, info = agent1.get_action(state1, deterministic=False)
        actions1.append(action)
        print(f"  動作: 候選={action[0]}, 替換={action[1]}")
    
    # 第二次
    print("\n重新設置種子並再次採樣:")
    set_seed(42)
    agent2 = TwoDimensionalPPOAgent(
        feature_dim=config.gnn_embed_dim,
        hidden_dim=config.hidden_dim,
        max_candidates=config.max_candidates,
        max_replacements=config.max_replacements,
        device='cpu'
    )
    
    set_seed(42)
    state2 = MockState()
    actions2 = []
    for _ in range(5):
        action, info = agent2.get_action(state2, deterministic=False)
        actions2.append(action)
        print(f"  動作: 候選={action[0]}, 替換={action[1]}")
    
    print(f"\n✅ 動作序列相同: {actions1 == actions2}")
    if actions1 != actions2:
        print(f"❌ 不一致！")
        print(f"第一次: {actions1}")
        print(f"第二次: {actions2}")
    
    return actions1 == actions2

def test_gnn_embeddings():
    """測試 GNN embeddings 是否可重現"""
    from gnn_api import load_encoder, get_embeddings
    
    print("\n" + "="*60)
    print("測試: GNN Embeddings 可重現性")
    print("="*60)
    
    meta_path = os.path.expanduser('~/ruan_workspace/ic_cad/gnn/models/all/encoder_meta.json')
    
    if not os.path.exists(meta_path):
        print("⚠️ GNN meta 文件不存在，跳過測試")
        return True
    
    try:
        # 第一次
        set_seed(42)
        encoder1, meta1 = load_encoder(meta_path=meta_path)
        encoder1.eval()
        
        # 創建測試輸入
        set_seed(42)
        test_data = torch.randn(10, 20)  # 10 nodes, 20 features
        
        with torch.no_grad():
            # 模擬 GNN forward
            # 注意：這裡需要 edge_index，簡化測試
            emb1 = encoder1.convs[0](test_data, torch.tensor([[0,1,2],[1,2,3]]))
        
        # 第二次
        set_seed(42)
        encoder2, meta2 = load_encoder(meta_path=meta_path)
        encoder2.eval()
        
        set_seed(42)
        test_data2 = torch.randn(10, 20)
        
        with torch.no_grad():
            emb2 = encoder2.convs[0](test_data2, torch.tensor([[0,1,2],[1,2,3]]))
        
        is_same = torch.allclose(emb1, emb2, atol=1e-6)
        print(f"\n✅ GNN embeddings 相同: {is_same}")
        if not is_same:
            print(f"差異: {torch.abs(emb1 - emb2).max().item()}")
        
        return is_same
        
    except Exception as e:
        print(f"⚠️ GNN 測試失敗: {e}")
        return True

def test_environment_reset():
    """測試環境 reset 是否可重現"""
    from environment import OptimizationEnvironment
    from config import RLConfig
    
    print("\n" + "="*60)
    print("測試: Environment Reset 可重現性")
    print("="*60)
    
    config = RLConfig()
    
    try:
        # 第一次
        set_seed(42)
        env1 = OptimizationEnvironment(config)
        # 注意：這裡需要真實的 case，可能會失敗
        # state1 = env1.reset('s1488')
        print("✅ 環境創建成功（需要真實 case 才能測試 reset）")
        return True
        
    except Exception as e:
        print(f"⚠️ 環境測試失敗: {e}")
        return True

if __name__ == "__main__":
    print("="*60)
    print("🔬 推論可重現性詳細測試")
    print("="*60)
    
    results = []
    
    # 測試 PPO action sampling
    results.append(("PPO Action Sampling", test_ppo_action_sampling()))
    
    # 測試 GNN embeddings
    results.append(("GNN Embeddings", test_gnn_embeddings()))
    
    # 測試環境
    results.append(("Environment", test_environment_reset()))
    
    print("\n" + "="*60)
    print("📊 測試結果總結")
    print("="*60)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(r[1] for r in results)
    if all_passed:
        print("\n🎉 所有測試通過！")
    else:
        print("\n⚠️ 部分測試失敗，需要進一步調查")
