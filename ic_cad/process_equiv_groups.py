import json
from collections import defaultdict
import re

def extract_cell_base_type(cell_name):
    """從 cell 名稱中提取基本類型（去除數字大小但保留重要變體）"""
    # 移除後綴 _ASAP7_75t_SRAM/_R/_L/_SL
    base_name = re.sub(r'_ASAP7_75t_(SRAM|R|L|SL)$', '', cell_name)
    
    # 提取基本類型，保留重要變體如 f, DC
    # 先檢查是否有重要變體後綴
    if base_name.endswith('f'):
        # 對於帶f的，移除x和數字部分但保留f
        cell_type = re.sub(r'x\d+(\.\d+)?(p\d+)?f$', 'f', base_name)
    elif 'DC' in base_name:
        # 對於帶DC的，移除x和數字部分但保留DC
        cell_type = re.sub(r'x\d+(\.\d+)?(p\d+)?DC', 'DC', base_name)
    else:
        # 對於普通的，移除x和數字部分
        cell_type = re.sub(r'x\d+(\.\d+)?(p\d+)?$', '', base_name)
    
    return cell_type

def group_by_base_type(cells):
    """按基本類型分組，相同基本類型的不同大小放在一起，但不同變體分開"""
    groups = defaultdict(list)
    
    for cell in cells:
        # 獲取基本類型（保留重要變體）
        base_type = extract_cell_base_type(cell)
        groups[base_type].append(cell)
    
    return groups

def process_equiv_groups(input_file, output_file):
    """處理等價組，分離不同類型並按長度排序"""
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    new_groups = []
    
    for group in data:
        # 按基本類型分組（相同類型不同大小放一起，不同變體分開）
        sub_groups = group_by_base_type(group)
        
        # 將每個子組添加到新的組列表中
        for base_type, cells in sub_groups.items():
            # 按字母順序排序每個組內的 cells
            sorted_cells = sorted(cells)
            new_groups.append(sorted_cells)
    
    # 按組的長度排序（從長到短）
    new_groups.sort(key=len, reverse=True)
    
    # 寫入新文件
    with open(output_file, 'w') as f:
        json.dump(new_groups, f, indent=2)
    
    print(f"處理完成！")
    print(f"原始組數: {len(data)}")
    print(f"新組數: {len(new_groups)}")
    
    # 顯示一些統計信息
    length_counts = defaultdict(int)
    for group in new_groups:
        length_counts[len(group)] += 1
    
    print("\n按長度統計的組數:")
    for length in sorted(length_counts.keys(), reverse=True):
        print(f"長度 {length}: {length_counts[length]} 組")
    
    # 顯示前幾個最長的組
    print(f"\n前 5 個最長的組:")
    for i, group in enumerate(new_groups[:5]):
        print(f"第 {i+1} 組 (長度 {len(group)}): {group[0].split('_')[0]} 類型")

if __name__ == "__main__":
    input_file = "/root/ruan_workspace/ic_cad/equiv_groups_new_sorted.json"
    output_file = "/root/ruan_workspace/ic_cad/equiv_groups_separated_sorted.json"
    
    process_equiv_groups(input_file, output_file)
