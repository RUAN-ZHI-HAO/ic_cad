import json
import re

def extract_cell_base_type(cell_name):
    """從 cell 名稱中提取基本類型（去除數字大小但保留重要變體）"""
    # 移除後綴 _ASAP7_75t_SRAM/_R/_L/_SL
    base_name = re.sub(r'_ASAP7_75t_(SRAM|R|L|SL)$', '', cell_name)
    
    # 提取基本類型，保留重要變體如 f, DC
    # 對於 CKINVDC 這種情況，DC 是名稱的一部分而不是後綴
    if base_name.endswith('f'):
        # 對於帶f的，移除x和數字部分但保留f
        cell_type = re.sub(r'x\d+(\.\d+)?(p\d+)?f$', 'f', base_name)
    elif base_name.endswith('DC'):
        # 對於以DC結尾的，移除x和數字部分但保留DC
        cell_type = re.sub(r'x\d+(\.\d+)?(p\d+)?DC$', 'DC', base_name)
    elif 'DC' in base_name:
        # 對於包含DC的（如CKINVDC），保留整個DC部分
        # 移除 DC 後面的 x 和數字部分
        cell_type = re.sub(r'DCx\d+(\.\d+)?(p\d+)?$', 'DC', base_name)
    else:
        # 對於普通的，移除x和數字部分
        cell_type = re.sub(r'x\d+(\.\d+)?(p\d+)?$', '', base_name)
    
    return cell_type

# 測試一些 CKINVDC 的例子
test_cases = [
    "CKINVDCx5p33_ASAP7_75t_L",
    "CKINVDCx6p67_ASAP7_75t_R", 
    "CKINVDCx8_ASAP7_75t_SL",
    "CKINVDCx16_ASAP7_75t_SRAM",
    "CKINVDCx20_ASAP7_75t_L",
    "BUFx4_ASAP7_75t_L",
    "BUFx4f_ASAP7_75t_L"
]

for case in test_cases:
    result = extract_cell_base_type(case)
    print(f"{case} -> {result}")
