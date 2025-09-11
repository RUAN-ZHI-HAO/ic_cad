from utils_openroad import OpenRoadInterface, OptimizationAction, MetricsReport
import os
import json

def power_optimization(interface: OpenRoadInterface, cell_groups_map, cell_groups,  top_n: int = 10):
    _, power_list, _ = interface.get_candidate_cells(top_power=top_n)
    for inst in power_list:
        inst_name = inst["instance_name"]
        cell_type = inst["cell_type"]
        new_cell_type_index, cell_index = cell_groups_map.get(cell_type, (None, None))
        # print("inst_name:", inst_name, "cell_type:", cell_type)
        if cell_index > 0:
            cell_index -= 1
            new_cell_type = cell_groups[new_cell_type_index][cell_index]
        
        # print("cell_type:", cell_type, " new_cell_type:", new_cell_type)
        if "_R" in cell_type and "_SRAM" not in new_cell_type:
            continue
        elif "_L" in cell_type and "_R" not in new_cell_type:
            continue
        elif "_SL" in cell_type and "_L" not in new_cell_type:
            continue
        elif "_SRAM" in cell_type:
            continue

        # print(f"替換 {inst_name} 的 cell 從 {cell_type} 到 {new_cell_type}")

        action = OptimizationAction(action_type="replace_cell", target_cell=inst_name, new_cell_type=new_cell_type)
        interface.apply_action(action)

def delay_optimization(interface: OpenRoadInterface, cell_groups_map, cell_groups,  top_n: int = 10):
    delay, _, _ = interface.get_candidate_cells(top_delay=top_n)
    for inst in delay:
        inst_name = inst["instance_name"]
        cell_type = inst["cell_type"]
        new_cell_type_index, cell_index = cell_groups_map.get(cell_type, (None, None))
        # print("inst_name:", inst_name, "cell_type:", cell_type)
        if cell_index < len(cell_groups[new_cell_type_index]) - 1:
            cell_index += 1
            new_cell_type = cell_groups[new_cell_type_index][cell_index]
        
        # print("cell_type:", cell_type, " new_cell_type:", new_cell_type)
        if "_SRAM" in cell_type and "_R" not in new_cell_type:
            continue
        elif "_R" in cell_type and "_L" not in new_cell_type:
            continue
        elif "_L" in cell_type and "_SL" not in new_cell_type:
            continue
        elif "_SL" in cell_type:
            continue

        # print(f"替換 {inst_name} 的 cell 從 {cell_type} 到 {new_cell_type}")

        action = OptimizationAction(action_type="replace_cell", target_cell=inst_name, new_cell_type=new_cell_type)
        interface.apply_action(action)

def slew_optimization(interface: OpenRoadInterface, cell_groups_map, cell_groups,  top_n: int = 10):
    _, _, slew = interface.get_candidate_cells(top_slew=top_n)
    for inst in slew:
        inst_name = inst["instance_name"]
        cell_type = inst["cell_type"]
        new_cell_type_index, cell_index = cell_groups_map.get(cell_type, (None, None))
        # print("inst_name:", inst_name, "cell_type:", cell_type)
        if cell_index < len(cell_groups[new_cell_type_index]) - 1:
            cell_index += 1
            new_cell_type = cell_groups[new_cell_type_index][cell_index]
        
        # print("cell_type:", cell_type, " new_cell_type:", new_cell_type)
        if "_SRAM" in cell_type and "_R" not in new_cell_type:
            continue
        elif "_R" in cell_type and "_L" not in new_cell_type:
            continue
        elif "_L" in cell_type and "_SL" not in new_cell_type:
            continue
        elif "_SL" in cell_type:
            continue

        # print(f"替換 {inst_name} 的 cell 從 {cell_type} 到 {new_cell_type}")

        action = OptimizationAction(action_type="replace_cell", target_cell=inst_name, new_cell_type=new_cell_type)
        interface.apply_action(action)

def get_cell_type(cell_groups):
    cell_groups_map = {}
    for i in range(len(cell_groups)):
        cell_group = cell_groups[i]
        for j in range(len(cell_group)):
            cell = cell_group[j]
            cell_groups_map[cell] = [i, j]  # (cell_group_index, cell_index)
    return cell_groups_map

def main():
    """主要執行函數：使用簡化的 OpenRoadInterface"""
    pdk_root = os.path.expanduser("~/solution/testcases/ASAP7")
    design_path = os.path.expanduser("~/solution/testcases")
    output_path = "./output"
    os.makedirs(output_path, exist_ok=True)
    benchmark = "aes_cipher_top"

    with open(os.path.join("../equiv_groups_new.json")) as f:
        cell_groups = json.load(f)
        
    cell_groups_map = get_cell_type(cell_groups)

    # 創建簡化的介面
    interface = OpenRoadInterface(pdk_root=pdk_root)
    
    # 載入設計檔案
    def_path = f"{design_path}/{benchmark}/{benchmark}.def"
    sdc_path = f"{design_path}/{benchmark}/{benchmark}.sdc"
    
    print(f"🔄 載入設計: {benchmark}")
    interface.load_design(def_path, sdc_path, benchmark)
    
    # 初始分析
    print("📊 初始 STA 分析:")
    interface.report_metrics()
    
    # 執行優化
    print("🔧 執行優化...")
    interface.optimize()
    
    # 優化後分析
    print("📊 openroad 優化後 STA 分析:")
    interface.report_metrics()

    interface.update_cell_information()
    power_optimization(interface, cell_groups_map, cell_groups, top_n=100)
    # power 優化後分析
    print("📊 power 優化後 STA 分析:")
    interface.report_metrics()

    # interface.update_cell_information()
    # delay_optimization(interface, cell_groups_map, cell_groups, top_n=100)
    # # delay 優化後分析
    # print("📊 delay 優化後 STA 分析:")
    # interface.report_metrics()

    # interface.update_cell_information()
    # slew_optimization(interface, cell_groups_map, cell_groups, top_n=100)
    # # slew 優化後分析
    # print("📊 slew 優化後 STA 分析:")
    # interface.report_metrics()
    
    # 輸出結果
    print("✅ 寫入輸出檔案...")
    interface.write_output(output_path, benchmark)

if __name__ == "__main__":
    main()
