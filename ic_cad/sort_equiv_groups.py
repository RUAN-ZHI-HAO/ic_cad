#!/usr/bin/env python3
import json
import re

def parse_cell_name(cell_name):
    """解析 cell 名稱，提取基本名稱、尺寸和 VT 類型"""
    # 移除 _ASAP7_75t_ 後綴和 VT 類型
    parts = cell_name.split('_ASAP7_75t_')
    if len(parts) != 2:
        return cell_name, 0, ''  # 特殊情況，如 sram
    
    base_with_size = parts[0]
    vt_type = parts[1]
    
    # 提取尺寸部分
    # 匹配 x 後面的數字和小數點
    size_match = re.search(r'x(\d*p?\d+(?:f|DC)?)$', base_with_size)
    if size_match:
        size_str = size_match.group(1)
        base_name = base_with_size[:size_match.start()]
        
        # 轉換尺寸字符串為數值
        # 移除後綴
        clean_size = size_str.replace('f', '').replace('DC', '')
        
        if 'p' in clean_size:
            # 處理小數，如 p33 -> 0.33, 1p5 -> 1.5, 5p33 -> 5.33
            if clean_size.startswith('p'):
                size_val = float('0.' + clean_size[1:])
            else:
                parts = clean_size.split('p')
                size_val = float(parts[0] + '.' + parts[1])
        else:
            # 處理整數
            size_val = float(clean_size)
        
        return base_name, size_val, vt_type
    else:
        return base_with_size, 0, vt_type

def sort_group(group):
    """排序一個群組內的 cells"""
    # 解析所有 cell 名稱
    parsed = []
    for cell in group:
        base, size, vt = parse_cell_name(cell)
        parsed.append((cell, base, size, vt))
    
    # 定義 VT 類型順序
    vt_order = {'SRAM': 0, 'R': 1, 'L': 2, 'SL': 3}
    
    # 排序：先按基本名稱，再按尺寸，最後按 VT 類型
    sorted_parsed = sorted(parsed, key=lambda x: (x[1], x[2], vt_order.get(x[3], 999)))
    
    return [cell for cell, _, _, _ in sorted_parsed]

def main():
    # 讀取原始檔案
    with open('/root/ruan_workspace/ic_cad/equiv_groups_new.json', 'r') as f:
        groups = json.load(f)
    
    # 排序每個群組
    sorted_groups = []
    for group in groups:
        if len(group) > 1:  # 只有多個元素的群組才需要排序
            sorted_group = sort_group(group)
            sorted_groups.append(sorted_group)
        else:
            sorted_groups.append(group)  # 單個元素的群組保持不變
    
    # 寫入排序後的檔案
    with open('/root/ruan_workspace/ic_cad/equiv_groups_new_sorted.json', 'w') as f:
        json.dump(sorted_groups, f, indent=2)
    
    print("排序完成！結果保存在 equiv_groups_new_sorted.json")
    
    # 顯示第一個群組的排序結果作為範例
    print("\n第一個群組的排序結果:")
    for i, cell in enumerate(sorted_groups[0][:20]):  # 只顯示前20個
        print(f"{i:2d}: {cell}")
    if len(sorted_groups[0]) > 20:
        print(f"... 還有 {len(sorted_groups[0]) - 20} 個")

if __name__ == "__main__":
    main()
