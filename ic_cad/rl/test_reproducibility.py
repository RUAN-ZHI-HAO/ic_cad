#!/usr/bin/env python3
"""
測試推論的可重現性
"""

import sys
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

def test_random_sources():
    """測試各種隨機來源"""
    print("\n" + "="*60)
    print("測試 1: Python random")
    print("="*60)
    set_seed(42)
    r1 = [random.random() for _ in range(5)]
    print(f"第一次: {r1}")
    
    set_seed(42)
    r2 = [random.random() for _ in range(5)]
    print(f"第二次: {r2}")
    print(f"相同: {r1 == r2}")
    
    print("\n" + "="*60)
    print("測試 2: NumPy random")
    print("="*60)
    set_seed(42)
    n1 = np.random.rand(5).tolist()
    print(f"第一次: {n1}")
    
    set_seed(42)
    n2 = np.random.rand(5).tolist()
    print(f"第二次: {n2}")
    print(f"相同: {n1 == n2}")
    
    print("\n" + "="*60)
    print("測試 3: PyTorch random")
    print("="*60)
    set_seed(42)
    t1 = torch.rand(5).tolist()
    print(f"第一次: {t1}")
    
    set_seed(42)
    t2 = torch.rand(5).tolist()
    print(f"第二次: {t2}")
    print(f"相同: {t1 == t2}")
    
    print("\n" + "="*60)
    print("測試 4: PyTorch Categorical 採樣")
    print("="*60)
    from torch.distributions import Categorical
    
    set_seed(42)
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0])
    dist1 = Categorical(logits=logits)
    samples1 = [dist1.sample().item() for _ in range(10)]
    print(f"第一次採樣: {samples1}")
    
    set_seed(42)
    dist2 = Categorical(logits=logits)
    samples2 = [dist2.sample().item() for _ in range(10)]
    print(f"第二次採樣: {samples2}")
    print(f"相同: {samples1 == samples2}")
    
    print("\n" + "="*60)
    print("測試 5: PyTorch Dropout (eval mode)")
    print("="*60)
    import torch.nn as nn
    
    dropout = nn.Dropout(p=0.5)
    dropout.eval()  # 評估模式下應該不做任何操作
    
    set_seed(42)
    x = torch.ones(5)
    y1 = dropout(x)
    print(f"第一次 (eval): {y1.tolist()}")
    
    set_seed(42)
    y2 = dropout(x)
    print(f"第二次 (eval): {y2.tolist()}")
    print(f"相同: {torch.allclose(y1, y2)}")
    
    print("\n" + "="*60)
    print("✅ 所有隨機來源測試完成")
    print("="*60)

if __name__ == "__main__":
    test_random_sources()
