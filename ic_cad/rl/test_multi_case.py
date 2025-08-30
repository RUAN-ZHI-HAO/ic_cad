#!/usr/bin/env python3
"""測試多案例載入和切換"""

import sys
import os
sys.path.append('/root/ruan_workspace/ic_cad/rl')
sys.path.append('/root/ruan_workspace/ic_cad/gnn')

from utils_openroad import OpenRoadInterface
import logging

logging.basicConfig(level=logging.INFO)

def test_multi_case_loading():
    """測試多個案例的載入和切換"""
    print("🔧 初始化 OpenROAD 介面...")
    openroad = OpenRoadInterface(
        cell_groups_json="/root/ruan_workspace/ic_cad/gnn/cell_groups.json"
    )
    
    test_cases = ['s1488', 's298']
    
    for i, case in enumerate(test_cases):
        print(f"\n{'='*50}")
        print(f"🔄 測試載入案例 {i+1}: {case}")
        try:
            openroad.load_case(case)
            print(f"✅ 成功載入案例: {case}")
            
            # 測試基本操作
            if openroad.has_case(case):
                print(f"✅ 案例 {case} 已在記憶體中")
                design = openroad.get_design(case)
                block = design.getBlock()
                inst_count = len(list(block.getInsts()))
                print(f"📊 案例 {case} 有 {inst_count} 個實例")
            else:
                print(f"❌ 案例 {case} 不在記憶體中")
                
        except Exception as e:
            print(f"❌ 載入案例 {case} 失敗: {e}")
            return False
    
    # 測試快速切換
    print(f"\n{'='*50}")
    print("🔄 測試快速案例切換...")
    for i in range(3):
        for case in test_cases:
            print(f"🔄 第{i+1}輪，切換到案例: {case}")
            try:
                openroad.load_case(case)  # 應該很快，因為已經在記憶體中
                print(f"✅ 快速切換到案例: {case}")
            except Exception as e:
                print(f"❌ 切換到案例 {case} 失敗: {e}")
                return False
    
    print(f"\n{'='*50}")
    print("🎉 多案例測試通過！")
    return True

if __name__ == "__main__":
    success = test_multi_case_loading()
    if success:
        print("✅ 測試成功！多案例載入和切換正常工作")
        sys.exit(0)
    else:
        print("❌ 測試失敗！")
        sys.exit(1)
