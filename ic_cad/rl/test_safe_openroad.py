#!/usr/bin/env python3
"""測試修復後的 OpenROAD 介面安全性"""

import sys
import os
sys.path.append('/root/ruan_workspace/ic_cad/rl')
sys.path.append('/root/ruan_workspace/ic_cad/gnn')

from utils_openroad import OpenRoadInterface
import logging

logging.basicConfig(level=logging.INFO)

def test_safe_openroad():
    """測試修復後的安全 OpenROAD 操作"""
    print("🔧 測試安全的 OpenROAD 操作...")
    
    try:
        openroad = OpenRoadInterface(
            cell_groups_json="/root/ruan_workspace/ic_cad/gnn/cell_groups.json"
        )
        
        case = 's1488'
        print(f"🔄 載入案例: {case}")
        openroad.load_case(case)
        
        print(f"📊 測試 report_metrics...")
        metrics = openroad.report_metrics(case)
        print(f"✅ Metrics: TNS={metrics.tns}, WNS={metrics.wns}, Power={metrics.total_power}")
        
        print(f"🔍 測試 update_cell_information...")
        openroad.update_cell_information(case)
        
        cell_info = openroad.get_cell_information(case)
        print(f"✅ Cell information: {len(cell_info)} cells processed")
        
        # 測試幾個 cell 的資訊
        sample_cells = list(cell_info.keys())[:3]
        for cell_name in sample_cells:
            info = cell_info[cell_name]
            print(f"  📋 {cell_name}: type={info.cell_type}, power={info.total_power:.6f}, slack={info.worst_slack:.3f}")
        
        print("🎉 所有測試通過！")
        return True
        
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_safe_openroad()
    if success:
        print("✅ 安全性測試成功！")
        sys.exit(0)
    else:
        print("❌ 安全性測試失敗！")
        sys.exit(1)
