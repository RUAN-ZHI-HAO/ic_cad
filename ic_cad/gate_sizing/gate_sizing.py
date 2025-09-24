from utils_openroad import OpenRoadInterface, OptimizationAction, MetricsReport
import os
import json
import time
from openroad import Tech, Design, Timing
import openroad

def no_slack_cell_optimization(interface: OpenRoadInterface, cell_groups_map, cell_groups):
    interface.update_cell_information()
    change_count = 0
    for cell, info in interface.cell_information.items():
        
        cell_type = info.cell_type
        if info.worst_slack > 20:
            new_cell_type_index, cell_index = cell_groups_map.get(cell_type, (None, None))
            # print("inst_name:", inst_name, "cell_type:", cell_type)
            if cell_index > 0:
                cell_index -= 1
                new_cell_type = cell_groups[new_cell_type_index][cell_index]
            else:
                continue
            
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

            action = OptimizationAction(action_type="replace_cell", target_cell=cell, new_cell_type=new_cell_type)
            interface.apply_action(action)
            change_count += 1
    print(f"no slack Power optimization: total changes made = {change_count}")

def power_optimization(interface: OpenRoadInterface, cell_groups_map, cell_groups,  top_n: int = 10):
    interface.update_cell_information()
    _, power_list, _, _ = interface.get_candidate_cells(top_power=top_n)
    change_count = 0
    for inst in power_list:
        inst_name = inst["instance_name"]
        cell_type = inst["cell_type"]
        new_cell_type_index, cell_index = cell_groups_map.get(cell_type, (None, None))
        # print("inst_name:", inst_name, "cell_type:", cell_type)
        if cell_index > 0:
            cell_index -= 1
            new_cell_type = cell_groups[new_cell_type_index][cell_index]
        else:
            continue
        
        # print("cell_type:", cell_type, " new_cell_type:", new_cell_type)
        if "_R" in cell_type and "_SRAM" not in new_cell_type:
            continue
        elif "_L" in cell_type and "_R" not in new_cell_type:
            continue
        elif "_SL" in cell_type and "_L" not in new_cell_type:
            continue
        elif "_SRAM" in cell_type:
            continue

        # print(interface.state_norm["worst_slack"][0])
        # print(interface.cell_information[inst_name].worst_slack)
        if interface.cell_information[inst_name].worst_slack < interface.state_norm["worst_slack"][0] * 0.7:
            continue

        # print(f"替換 {inst_name} 的 cell 從 {cell_type} 到 {new_cell_type}")

        action = OptimizationAction(action_type="replace_cell", target_cell=inst_name, new_cell_type=new_cell_type)
        interface.apply_action(action)
        change_count += 1
    print(f"Power optimization: total changes made = {change_count}")

def delay_optimization(interface: OpenRoadInterface, cell_groups_map, cell_groups,  top_n: int = 10):
    interface.update_cell_information()
    delay, _, _, _ = interface.get_candidate_cells(top_delay=top_n)
    change_count = 0
    for inst in delay:
        inst_name = inst["instance_name"]
        cell_type = inst["cell_type"]
        new_cell_type_index, cell_index = cell_groups_map.get(cell_type, (None, None))
        # print("inst_name:", inst_name, "cell_type:", cell_type)
        if cell_index < len(cell_groups[new_cell_type_index]) - 1:
            cell_index += 1
            new_cell_type = cell_groups[new_cell_type_index][cell_index]
        else:
            continue
        
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
        change_count += 1
    print(f"Delay optimization: total changes made = {change_count}")

def slew_optimization(interface: OpenRoadInterface, cell_groups_map, cell_groups,  top_n: int = 10):
    interface.update_cell_information()
    _, _, slew, _ = interface.get_candidate_cells(top_slew=top_n)
    change_count = 0
    for inst in slew:
        inst_name = inst["instance_name"]
        cell_type = inst["cell_type"]
        new_cell_type_index, cell_index = cell_groups_map.get(cell_type, (None, None))
        # print("inst_name:", inst_name, "cell_type:", cell_type)
        if cell_index < len(cell_groups[new_cell_type_index]) - 1:
            cell_index += 1
            new_cell_type = cell_groups[new_cell_type_index][cell_index]
        else:
            continue
        
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
        change_count += 1
    print(f"Slew optimization: total changes made = {change_count}")

def slack_optimization(interface: OpenRoadInterface, cell_groups_map, cell_groups,  top_n: int = 10):
    interface.update_cell_information()
    _, _, _, slack = interface.get_candidate_cells(top_slack=top_n)
    change_count = 0
    for inst in slack:
        inst_name = inst["instance_name"]
        cell_type = inst["cell_type"]
        new_cell_type_index, cell_index = cell_groups_map.get(cell_type, (None, None))
        # print("inst_name:", inst_name, "cell_type:", cell_type)
        if cell_index < len(cell_groups[new_cell_type_index]) - 1:
            cell_index += 1
            new_cell_type = cell_groups[new_cell_type_index][cell_index]
        else:
            continue
        
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
        change_count += 1
    print(f"Slack optimization: total changes made = {change_count}")

def get_cell_type(cell_groups):
    cell_groups_map = {}
    for i in range(len(cell_groups)):
        cell_group = cell_groups[i]
        for j in range(len(cell_group)):
            cell = cell_group[j]
            cell_groups_map[cell] = [i, j]  # (cell_group_index, cell_index)
    return cell_groups_map

def detailed_slack_optimization(interface: OpenRoadInterface, cell_groups_map, cell_groups, top_n: int = 10):
    # 更新 cell information
    interface.update_cell_information()
    
    # 收集所有 cell 的資訊，按照 worst_slack 分組
    slack_groups = {}
    
    for inst_name, cell_info in interface.cell_information.items():
        worst_slack = cell_info.worst_slack
        delay = cell_info.delay
        
        if worst_slack not in slack_groups:
            slack_groups[worst_slack] = []
        
        slack_groups[worst_slack].append((inst_name, delay))
    
    # 對每個 slack 群組內的 cell 按 delay 排序 (降序，delay 越大越前面)
    for worst_slack in slack_groups:
        slack_groups[worst_slack].sort(key=lambda x: x[1], reverse=True)
    
    # 按照 worst_slack 排序 (升序，worst_slack 越小越前面)
    sorted_slack_keys = sorted(slack_groups.keys())
    
    cell_slack = []
    for worst_slack in sorted_slack_keys:
        slack_group = [inst_name for inst_name, delay in slack_groups[worst_slack]]
        cell_slack.append(slack_group)
    
    change_count = 0
    for cell_list in cell_slack[:top_n]:
        print("group size:", len(cell_list))
        for inst_name in cell_list:
            inst = interface.cell_information[inst_name]
            cell_type = inst.cell_type
            new_cell_type_index, cell_index = cell_groups_map.get(cell_type, (None, None))
            # print("inst_name:", inst_name, "cell_type:", cell_type)
            if cell_index < len(cell_groups[new_cell_type_index]) - 1:
                cell_index += 1
                new_cell_type = cell_groups[new_cell_type_index][cell_index]
            else:
                continue
            
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
            change_count += 1
    print(f"detailed slack optimization: total changes made = {change_count}")

    # return cell_slack

def main():
    """主要執行函數：使用簡化的 OpenRoadInterface"""
    # 記錄開始時間
    start_time = time.time()
    
    pdk_root = os.path.expanduser("~/solution/testcases/ASAP7")
    design_path = os.path.expanduser("~/solution/testcases")
    output_path = "./output"
    os.makedirs(output_path, exist_ok=True)
    
    # 設定 benchmark (請根據需要選擇)
    # benchmark = "des" 
    benchmark = "ac97_top"
    # benchmark = "aes"
    # benchmark = "aes_cipher_top"
    # benchmark = "ariane"
    # benchmark = "pci_bridge32"
    # benchmark = "s1196"



    with open(os.path.join("./equiv_groups_new.json")) as f:
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
    
    # interface.analyze_critical_paths(2)
    interface.update_cell_information()
    interface.cell_count = len(interface.cell_information)
    print(f"Total cell count: {interface.cell_count}")

    # g325、g42561、FE_OFC183_n_12937
    # for i in range(len(cell_groups)):
    #     print(f"Cell group {i}", end=" ")
    #     for cell in cell_groups[i]:
    #         print(cell, end=" ")
    #     print()
    # for cell in cell_groups[17]:
    #     interface.apply_action(OptimizationAction(action_type="replace_cell", target_cell="g325", new_cell_type=cell))
    #     interface.design.evalTclString("update_timing")
    #     interface.design.evalTclString("estimate_parasitics -placement")
    #     interface.update_cell_information()
    #     print(f"new cell type {cell}", "\tdelay:", interface.cell_information["g325"].delay, "\tpower:", interface.cell_information["g325"].total_power)
    # print()
    # for cell in cell_groups[4]:
    #     interface.apply_action(OptimizationAction(action_type="replace_cell", target_cell="g42561", new_cell_type=cell))
    #     interface.design.evalTclString("update_timing")
    #     interface.design.evalTclString("estimate_parasitics -placement")
    #     interface.update_cell_information()
    #     print(f"new cell type {cell}", "\tdelay:", interface.cell_information["g42561"].delay, "\tpower:", interface.cell_information["g42561"].total_power)
    # print()
    # for cell in cell_groups[2]:
    #     interface.apply_action(OptimizationAction(action_type="replace_cell", target_cell="FE_OFC183_n_12937", new_cell_type=cell))
    #     interface.design.evalTclString("update_timing")
    #     interface.design.evalTclString("estimate_parasitics -placement")
    #     interface.update_cell_information()
    #     print(f"new cell type {cell}", "\tdelay:", interface.cell_information["FE_OFC183_n_12937"].delay, "\tpower:", interface.cell_information["FE_OFC183_n_12937"].total_power)
    # print()


    # for cell, info in interface.cell_information.items():
    #     print(cell)
    #     print(info.worst_slack)
    
    
    # 執行優化
    # print("🔧 執行優化...")
    # interface.optimize()
    # # 優化後分析
    # print("📊 openroad 優化後 STA 分析:")
    # interface.report_metrics()

    detailed_slack_optimization(interface, cell_groups_map, cell_groups, top_n=3)
    # detailed_slack 優化後分析
    print("📊 detailed_slack 優化後 STA 分析:")
    interface.report_metrics()

    power_optimization(interface, cell_groups_map, cell_groups, top_n=int(interface.cell_count * 0.05))
    # power 優化後分析
    print("📊 power 優化後 STA 分析:")
    interface.report_metrics()

    for i in range(3):
        no_slack_cell_optimization(interface, cell_groups_map, cell_groups)
        # no_slack_cell 優化後分析
        print("📊 no_slack_cell 優化後 STA 分析:")
        interface.report_metrics()

    delay_optimization(interface, cell_groups_map, cell_groups, top_n=int(interface.cell_count * 0.05))
    # delay 優化後分析
    print("📊 delay 優化後 STA 分析:")
    interface.report_metrics()

    slew_optimization(interface, cell_groups_map, cell_groups, top_n=int(interface.cell_count * 0.05))
    # slew 優化後分析
    print("📊 slew 優化後 STA 分析:")
    interface.report_metrics()

    slack_optimization(interface, cell_groups_map, cell_groups, top_n=int(interface.cell_count * 0.05))
    # slack 優化後分析
    print("📊 slack 優化後 STA 分析:")
    interface.report_metrics()

    

    power_optimization(interface, cell_groups_map, cell_groups, top_n=int(interface.cell_count * 0.05))
    # power 優化後分析
    print("📊 power 優化後 STA 分析:")
    interface.report_metrics()

    for i in range(3):
        no_slack_cell_optimization(interface, cell_groups_map, cell_groups)
        # no_slack_cell 優化後分析
        print("📊 no_slack_cell 優化後 STA 分析:")
        interface.report_metrics()

    interface.design.evalTclString("detailed_placement")
    interface.design.evalTclString("check_placement")

    interface.design.evalTclString("report_power")
    interface.design.evalTclString("report_tns")
    interface.design.evalTclString("report_wns")
    
    # 輸出結果
    print("✅ 寫入輸出檔案...")
    interface.write_output(output_path, benchmark)
    
    # 計算並顯示總執行時間
    end_time = time.time()
    total_time = end_time - start_time
    print(f"⏱️ 程式執行時間: {total_time:.2f} 秒")

if __name__ == "__main__":
    main()
