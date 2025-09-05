#!/usr/bin/env python3
import json
import sys

def analyze_cell_areas(json_file_path):
    """
    分析 JSON 檔案中的 cell area 資訊
    """
    print(f"正在分析檔案: {json_file_path}")
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"錯誤：無法讀取檔案 {e}")
        return
    
    # 收集所有 cell 的 area 資訊
    all_areas = []
    sram_areas = []
    non_sram_areas = []
    
    for cell_name, cell_data in data.items():
        if isinstance(cell_data, dict) and 'attributes' in cell_data:
            attributes = cell_data['attributes']
            if 'area' in attributes:
                try:
                    area = float(attributes['area'])
                    all_areas.append((cell_name, area))
                    
                    # 檢查是否為 sram cell
                    if cell_name.lower().startswith('sram'):
                        sram_areas.append((cell_name, area))
                    else:
                        non_sram_areas.append((cell_name, area))
                        
                except (ValueError, TypeError):
                    print(f"警告：無法解析 {cell_name} 的 area 值: {attributes['area']}")
    
    print(f"\n📊 分析結果：")
    print(f"總共找到 {len(all_areas)} 個有 area 資訊的 cell")
    print(f"其中 SRAM cell: {len(sram_areas)} 個")
    print(f"其中非 SRAM cell: {len(non_sram_areas)} 個")
    
    # 分析非 SRAM cell 的面積
    if non_sram_areas:
        non_sram_only_areas = [area for _, area in non_sram_areas]
        min_non_sram = min(non_sram_only_areas)
        max_non_sram = max(non_sram_only_areas)
        
        print(f"\n🚫 排除 SRAM cell 後的結果：")
        print(f"最小面積: {min_non_sram}")
        print(f"最大面積: {max_non_sram}")
        
        # 找出最小和最大面積的 cell 名稱
        min_cell = [name for name, area in non_sram_areas if area == min_non_sram]
        max_cell = [name for name, area in non_sram_areas if area == max_non_sram]
        
        print(f"最小面積的 cell: {min_cell}")
        print(f"最大面積的 cell: {max_cell}")
    
    # 分析包含 SRAM cell 的面積
    if all_areas:
        all_only_areas = [area for _, area in all_areas]
        min_all = min(all_only_areas)
        max_all = max(all_only_areas)
        
        print(f"\n✅ 包含 SRAM cell 的結果：")
        print(f"最小面積: {min_all}")
        print(f"最大面積: {max_all}")
        
        # 找出最小和最大面積的 cell 名稱
        min_all_cell = [name for name, area in all_areas if area == min_all]
        max_all_cell = [name for name, area in all_areas if area == max_all]
        
        print(f"最小面積的 cell: {min_all_cell}")
        print(f"最大面積的 cell: {max_all_cell}")
    
    # 分析 SRAM cell 的面積範圍
    if sram_areas:
        sram_only_areas = [area for _, area in sram_areas]
        min_sram = min(sram_only_areas)
        max_sram = max(sram_only_areas)
        
        print(f"\n🔲 SRAM cell 的面積範圍：")
        print(f"最小面積: {min_sram}")
        print(f"最大面積: {max_sram}")
        
        # 列出所有 SRAM cell
        print(f"\n所有 SRAM cell:")
        for name, area in sorted(sram_areas, key=lambda x: x[1]):
            print(f"  {name}: {area}")

if __name__ == "__main__":
    json_file = "/root/ruan_workspace/ic_cad/parser/parsed_cells.json"
    analyze_cell_areas(json_file)
