#!/usr/bin/env python3
"""
簡化版 OpenRoad Interface 使用範例
"""

from utils_openroad import OpenRoadInterface, OptimizationAction
import os

def main():
    print("🚀 簡化版 OpenRoad Interface 使用範例")
    
    # 創建介面實例
    interface = OpenRoadInterface()
    
    # 設定檔案路徑（您需要根據實際情況修改）
    case_name = "c17"  # 或您要處理的檔案名稱
    def_path = f"~/solution/testcases/{case_name}/{case_name}.def"
    sdc_path = f"~/solution/testcases/{case_name}/{case_name}.sdc"
    
    # 檢查檔案是否存在
    def_path_expanded = os.path.expanduser(def_path)
    sdc_path_expanded = os.path.expanduser(sdc_path)
    
    if not os.path.exists(def_path_expanded):
        print(f"❌ DEF 檔案不存在: {def_path_expanded}")
        print("請修改 def_path 變數為正確的檔案路徑")
        return
        
    if not os.path.exists(sdc_path_expanded):
        print(f"❌ SDC 檔案不存在: {sdc_path_expanded}")
        print("請修改 sdc_path 變數為正確的檔案路徑")
        return
    
    try:
        # 1. 載入設計檔案
        print(f"📂 載入設計檔案: {case_name}")
        interface.load_design(def_path, sdc_path, case_name)
        print("✅ 設計檔案載入成功")
        
        # 2. 取得初始效能報告
        print("\n📊 初始效能報告:")
        metrics = interface.report_metrics()
        
        # 3. 更新 cell 資訊
        print("🔄 更新 cell 資訊...")
        cell_info_list = interface.update_cell_information()
        print(f"✅ 已分析 {len(cell_info_list)} 個 cells")
        
        # 4. 取得動態特徵
        print("🧮 計算動態特徵...")
        features, cell_names = interface.get_dynamic_features()
        print(f"✅ 特徵矩陣大小: {features.shape}")
        
        # 5. 取得候選優化目標
        print("🎯 尋找候選優化目標...")
        candidates = interface.get_candidate_cells(top_delay=5, top_power=5, top_slew=5)
        print(f"✅ 找到 {len(candidates)} 個候選 cells:")
        
        for i, (inst_name, cell_type) in enumerate(candidates[:3]):
            cell_info = interface.get_cell_information(inst_name)
            if cell_info:
                print(f"  {i+1}. {inst_name} ({cell_type})")
                print(f"     - Power: {cell_info.total_power:.2e} W")
                print(f"     - Delay: {cell_info.delay:.2f} ps")
                print(f"     - Slew:  {cell_info.output_slew:.2f} ps")
        
        # 6. 測試 cell replacement（可選）
        if candidates and len(candidates) > 0:
            print(f"\n🔧 測試 cell replacement...")
            test_inst, test_cell = candidates[0]
            
            # 假設替換為不同的 cell type（您需要根據實際可用的 cell 修改）
            new_cell_type = "INVx1_ASAP7_75t_L"  # 範例 cell type
            
            action = OptimizationAction(
                action_type="replace_cell",
                target_cell=test_inst,
                new_cell_type=new_cell_type
            )
            
            print(f"   替換 {test_inst}: {test_cell} -> {new_cell_type}")
            success = interface.apply_action(action)
            
            if success:
                print("✅ Cell replacement 成功")
                print("\n📊 替換後效能報告:")
                new_metrics = interface.report_metrics()
                
                # 比較前後差異
                print(f"   Power 變化: {metrics.total_power:.2e} -> {new_metrics.total_power:.2e}")
                print(f"   TNS 變化:   {metrics.tns:.3f} -> {new_metrics.tns:.3f}")
                print(f"   WNS 變化:   {metrics.wns:.3f} -> {new_metrics.wns:.3f}")
            else:
                print("❌ Cell replacement 失敗")
        
        print("\n🎉 範例執行完成!")
        
    except Exception as e:
        print(f"❌ 執行過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
