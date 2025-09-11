from openroad import Tech, Design, Timing
import openroad
import os
import json
import re
from typing import Set
import rcx

# === 建立技術與設計物件 ===
def load_case(pdk_path, design_path, benchmark):
    tech = Tech()
    design = Design(tech)
    # design.evalTclString("define_corners tt") 
    # pdk_path = os.path.expanduser("~/solution/testcases/ASAP7")
    # design_path = os.path.expanduser("~/solution/testcases")

    # === 載入 PDK 資料 ===
    tech.readLef(f"{pdk_path}/techlef/asap7_tech_1x_201209.lef")

    lef_files = [
        "LEF/asap7sc7p5t_28_L_1x_220121a.lef",
        "LEF/asap7sc7p5t_28_R_1x_220121a.lef",
        "LEF/asap7sc7p5t_28_SL_1x_220121a.lef",
        "LEF/asap7sc7p5t_28_SRAM_1x_220121a.lef",
        "LEF/sram_asap7_16x256_1rw.lef",
        "LEF/sram_asap7_32x256_1rw.lef",
        "LEF/sram_asap7_64x256_1rw.lef",
        "LEF/sram_asap7_64x64_1rw.lef"
    ]
    for lef_file in lef_files:
        tech.readLef(f"{pdk_path}/{lef_file}")

    lib_files = [  
        "LIB/asap7sc7p5t_AO_LVT_TT_nldm_211120.lib",  
        "LIB/asap7sc7p5t_AO_RVT_TT_nldm_211120.lib",  
        "LIB/asap7sc7p5t_AO_SLVT_TT_nldm_211120.lib",  
        "LIB/asap7sc7p5t_AO_SRAM_TT_nldm_211120.lib",  
        "LIB/asap7sc7p5t_INVBUF_LVT_TT_nldm_220122.lib",  
        "LIB/asap7sc7p5t_INVBUF_RVT_TT_nldm_220122.lib",  
        "LIB/asap7sc7p5t_INVBUF_SLVT_TT_nldm_220122.lib",  
        "LIB/asap7sc7p5t_INVBUF_SRAM_TT_nldm_220122.lib",  
        "LIB/asap7sc7p5t_OA_LVT_TT_nldm_211120.lib",  
        "LIB/asap7sc7p5t_OA_RVT_TT_nldm_211120.lib",  
        "LIB/asap7sc7p5t_OA_SLVT_TT_nldm_211120.lib",  
        "LIB/asap7sc7p5t_OA_SRAM_TT_nldm_211120.lib",  
        "LIB/asap7sc7p5t_SEQ_LVT_TT_nldm_220123.lib",  
        "LIB/asap7sc7p5t_SEQ_RVT_TT_nldm_220123.lib",  
        "LIB/asap7sc7p5t_SEQ_SLVT_TT_nldm_220123.lib",  
        "LIB/asap7sc7p5t_SEQ_SRAM_TT_nldm_220123.lib",  
        "LIB/asap7sc7p5t_SIMPLE_LVT_TT_nldm_211120.lib",  
        "LIB/asap7sc7p5t_SIMPLE_RVT_TT_nldm_211120.lib",  
        "LIB/asap7sc7p5t_SIMPLE_SLVT_TT_nldm_211120.lib",  
        "LIB/asap7sc7p5t_SIMPLE_SRAM_TT_nldm_211120.lib",  
        "LIB/sram_asap7_16x256_1rw.lib",  
        "LIB/sram_asap7_32x256_1rw.lib",  
        "LIB/sram_asap7_64x256_1rw.lib",  
        "LIB/sram_asap7_64x64_1rw.lib"  
    ]  
    for lib_file in lib_files:
        tech.readLiberty(f"{pdk_path}/{lib_file}")

    # === 載入設計與 RC 資訊 ===
    design.readDef(f"{design_path}/{benchmark}/{benchmark}.def")
    design.evalTclString(f"read_sdc {design_path}/{benchmark}/{benchmark}.sdc")
    design.evalTclString(f"source {pdk_path}/setRC.tcl")
    design.evalTclString("estimate_parasitics -placement")  

    return design, tech

# === 優化流程函式 ===
def replace_cells():
    db = openroad.get_db()
    block = openroad.get_db_block()
    def try_swap(inst_name, new_cell):
        inst = block.findInst(inst_name)
        if inst:
            new_master = db.findMaster(new_cell)
            if new_master:
                inst.swapMaster(new_master)
                print(f"[INFO] Replaced {inst_name} → {new_cell}")
            else:
                print(f"[WARN] Master cell not found: {new_cell}")
        else:
            print(f"[WARN] Instance not found: {inst_name}")

    # try_swap("_1_", "NAND2x1_ASAP7_75t_L")
    # try_swap("_2_", "AO22x2_ASAP7_75t_L")
    # try_swap("_3_", "OA21x4_ASAP7_75t_L")

def insert_buffers():
    design.evalTclString("buffer_ports -inputs -outputs")

def optimize_and_report():
    design.evalTclString("repair_design")
    design.evalTclString("repair_timing")
    design.evalTclString("report_worst_slack -max -digits 3")
    design.evalTclString("report_tns -digits 3")
    design.evalTclString("report_power")

def write_output():
    design.evalTclString("write_def c17_optimized.def")
    design.evalTclString("write_changelog changelist.log")

def replace_instance_cell(inst_name: str, new_master: str):
    db = openroad.get_db()
    block = openroad.get_db_block()
    inst = block.findInst(inst_name)
    if not inst:
        print(f"[ERROR] Instance {inst_name} not found.")
        return
    master = db.findMaster(new_master)
    if not master:
        print(f"[ERROR] Master cell {new_master} not found.")
        return
    inst.swapMaster(master)
    print(f"[INFO] Replaced instance {inst_name} with master {new_master}.")

def insert_buffer(load_pins: list, lib_buf: str, new_buf_inst: str, new_net: str):
    pins_str = " ".join(load_pins)
    cmd = f"insert_buffer {{{pins_str}}} {lib_buf} {new_buf_inst} {new_net}"
    print(f"[INFO] Executing: {cmd}")
    design.evalTclString(cmd)

def analyze_sta_summary():
    """
    執行 STA 分析並回傳 power、tns、wns(float)
    """
    def extract_number(result_str):
        for line in result_str.splitlines():
            for token in line.strip().split():
                try:
                    return float(token)
                except ValueError:
                    continue
        return 0.0

    print("🔍 Analyzing STA summary...")

    # power_str = design.evalTclString("report_power -corner default -digits 3")
    design.evalTclString("update_timing")
    timing = Timing(design)
    total_power = 0.0
    for inst in design.getBlock().getInsts():  
        for corner in timing.getCorners():  
            try:
                static_power = timing.staticPower(inst, corner)  
                dynamic_power = timing.dynamicPower(inst, corner)  
                total_power += static_power + dynamic_power  
            except Exception as e:
                logger.debug(f"功耗計算失敗 {inst.getName()}: {e}")
                continue
    tns = float(design.evalTclString("total_negative_slack -max"))
    wns = float(design.evalTclString("worst_negative_slack -max"))  

    # power = extract_number(power_str)
    # tns = extract_number(tns_str)
    # wns = extract_number(wns_str)

    print(f"📊 Power: {total_power} uW")
    print(f"📉 TNS  : {tns} ns")
    print(f"📉 WNS  : {wns} ns")

    # return power, tns, wns

def find_equivalent_cells(tech, design, cell_name):  
    """  
    查找指定 cell 的所有等效 cell  
      
    Args:  
        tech: Tech 物件  
        design: Design 物件    
        cell_name: 要查詢的 cell 名稱  
    """  
    from openroad import Timing  

    # 建立 Timing 物件  
    timing = Timing(design)  
      
    # 初始化等效 cell 對應關係  
    timing.makeEquivCells()  
      
    # 取得資料庫  
    db = tech.getDB()  
      
    # 尋找指定的 master  
    target_master = db.findMaster(cell_name)  
      
    if not target_master:  
        print(f"找不到 cell: {cell_name}")  
        return

    same_cell = set()

    # 取得等效 cell  
    equiv_cells = timing.equivCells(target_master)  
      
    # print(f"等效於 '{cell_name}' 的 cell:")  
    # print("=" * 50)  
      
    if not equiv_cells:  
        # print("沒有找到等效的 cell")  
        return  
      
    for equiv_cell in equiv_cells:  
        same_cell.add(equiv_cell.getName())
        # print(f"  - {equiv_cell.getName()}")  
    
    return same_cell
    # print(f"\n總共找到 {len(equiv_cells)} 個等效 cell")

def parse_cells(text: str, valid: Set[str]) -> Set[str]:
    pattern = re.compile(r'^\s*\*?([A-Za-z0-9_]+)\s+[0-9.]+', re.M)
    return {m for m in pattern.findall(text or "") if m in valid}

def create_cell_groups(tech, design, include_singletons=False):  
    # 1) 取全部 master 名稱  
    masters = []  
    for lib in tech.getDB().getLibs():  
        for master in lib.getMasters():  
            name = master.getName()  
            if name:  
                masters.append(str(name))  
    masters = sorted(set(masters))  
    valid = set(masters)  
    groups = []  
    visited = set()  
      
    # 2) 逐個 seed，沒分過組才查  
    for seed in masters:  
        if seed in visited:  
            continue  
              
        # 獲取等價 cells  
        # out = design.evalTclString(f'tee -quiet -variable result {{ report_equiv_cells -all {seed} }}; Set result') 
        # print(out) 
        # eq = parse_cells(out, valid)  
        # print(eq)
        eq = find_equivalent_cells(tech, design, seed)
          
        # 移除已經被分組的 cells  
        eq = eq - visited  
          
        # 如果還有剩餘的 cells，才建立群組  
        if eq and (include_singletons or len(eq) > 1):  
            groups.append(sorted(eq))  
            visited.update(eq)  
      
    # 大組在前  
    groups.sort(key=lambda g: (-len(g), g[0]))  
    return groups

def save_groups_to_json(groups, path: str):
    """
    儲存成 JSON，一個群組(list)一行。
    例如：
    [
      ["NAND2xp33_ASAP7_75t_L", "NAND2xp5_ASAP7_75t_L"],
      ["BUFx1_ASAP7_75t_L", "BUFx2_ASAP7_75t_L"]
    ]
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write("[\n")
        for i, g in enumerate(groups):
            line = json.dumps(g, ensure_ascii=False)
            if i < len(groups) - 1:
                f.write(f"  {line},\n")
            else:
                f.write(f"  {line}\n")
        f.write("]\n")


def analyze_critical_paths():  
    """  
    分析 TNS 貢獻最多的前五條路徑中的所有 cells  
    """  
    # 報告前5條最差的timing paths  
    # result = design.evalTclString("report_checks -path_delay max -format full_clock_expanded -fields {input_pin slew capacitance} -digits 3 -group_count 5")  
    # print("前5條關鍵路徑:")  
    # print(result)  
      
    # 也可以使用更詳細的報告  
    detailed_result = design.evalTclString("report_checks -path_delay max -format full -fields {input_pin output_pin slew capacitance delay arrival required} -digits 3 -group_count 1")  
    print("\n詳細時序報告:")  
    print(detailed_result)  
      
    # return result

def update_cell_information():
    sta = tech.getSta()
    design.evalTclString("update_timing")

    print(', '.join(dir(sta)))
    print(', '.join(dir(design)))
    print(', '.join(dir(tech)))

    try:
        timing = Timing(design) 
    except Exception as e:
        logger.error(f"無法創建 Timing 物件: {e}")
        return
        
    cell_info_list = []  # 初始化 cell_info_list
    
    # graph_delay_calc = sta.graphDelayCalc()  
    corners = timing.getCorners()  
    default_corner = corners[0] if corners else None  
 
    for inst in design.getBlock().getInsts():  
        inst_name = inst.getName()  
        master = inst.getMaster()  
        cell_type = master.getName()  
        
        total_power = 0  
        static_power_total = 0  
        dynamic_power_total = 0
        worst_slack = float('inf')  
        
        try:
            # 功耗分析 - 更安全的實現
            for corner in timing.getCorners():  
                try:
                    static_power = timing.staticPower(inst, corner)  
                    dynamic_power = timing.dynamicPower(inst, corner)  
                    static_power_total += static_power  
                    dynamic_power_total += dynamic_power
                    total_power += static_power + dynamic_power
                except Exception as e:
                    logger.debug(f"功耗計算失敗 {inst_name}: {e}")
                    continue
        except Exception as e:
            logger.debug(f"功耗分析失敗 {inst_name}: {e}")

        # print(f"total_power : {total_power} static_power_total :{static_power_total} dynamic_power_total : {dynamic_power_total}")

        # 簡化的 VT 類型解析（從 cell 名稱）  
        vt_type = "L"  # 默認值  
        if "_SRAM" in cell_type:  
            vt_type = "SRAM"  
        elif "_SL" in cell_type:
            vt_type = "SL"
        elif "_R" in cell_type:  
            vt_type = "R"  
        elif "_L" in cell_type:  
            vt_type = "L"  
        
        # 電氣特性分析 - 更安全的實現
        output_cap = 0  
        input_slew = 0  
        fanout_count = 0  
        is_endpoint = False
        
        try:
            for iTerm in inst.getITerms():  
                if not iTerm.getNet():  
                    continue  

                # 檢查是否為 timing endpoint  
                try:  
                    if timing.isEndpoint(iTerm):  
                        is_endpoint = True  
                except Exception as e:  
                    logger.debug(f"Endpoint 檢查失敗: {e}")
                
                try:  
                    # 簡化的 Slack 分析 - 使用預設 corner  
                    rise_slack = timing.getPinSlack(iTerm, Timing.Rise, Timing.Max)  
                    fall_slack = timing.getPinSlack(iTerm, Timing.Fall, Timing.Max)  
                    pin_worst_slack = min(rise_slack, fall_slack)  
                    worst_slack = min(worst_slack, pin_worst_slack)  
                except Exception as e:  
                    logger.debug(f"Slack 分析失敗 {inst_name}/{iTerm.getMTerm().getName()}: {e}")
                
                try:
                    # 電氣特性  
                    mterm = iTerm.getMTerm()  
                    if mterm.getIoType() == "OUTPUT":  
                        try:

                            # 使用 corner-specific 的 capacitance 計算  
                            for corner in timing.getCorners():  
                                port_cap = timing.getPortCap(iTerm, corner, Timing.Max)  
                                if iTerm.getNet():  
                                    net_cap = timing.getNetCap(iTerm.getNet(), corner, Timing.Max)  
                                    total_cap = port_cap + net_cap  
                                    output_cap = max(output_cap, total_cap)  
                            
                            # 如果 corner-specific 方法失敗，使用原始方法  
                            if output_cap == 0:  
                                output_cap = timing.getMaxCapLimit(mterm) 

                        except Exception as e:
                            logger.debug(f"Cap limit 取得失敗: {e}")
                            output_cap = 0
                            
                        # 計算 fanout 數量  
                        net = iTerm.getNet()  
                        if net:  
                            fanout_count = len(list(net.getITerms())) - 1   
                            
                    elif mterm.getIoType() == "INPUT":  
                        # 完全跳過 getMaxSlewLimit 調用，因為它會導致段錯誤
                        try:  
                            max_slew = timing.getMaxSlewLimit(mterm)  
                            input_slew = max(input_slew, max_slew)  
                        except Exception as e:  
                            logger.debug(f"Slew limit 取得失敗 {inst_name}/{mterm.getName()}: {e}")
                            input_slew = 0
                except Exception as e:
                    logger.debug(f"電氣特性分析失敗 {inst_name}: {e}")
                    continue
        except Exception as e:
            logger.warning(f"ITerm 分析失敗 {inst_name}: {e}")


        drive_resistance = 0.0  
        try:  
            for mterm in master.getMTerms():  
                if mterm.getIoType() == "OUTPUT":  
                    fanout_terms = timing.getTimingFanoutFrom(mterm)  
                    if fanout_terms:  
                        # 基於實際的 timing arc 數量計算  
                        drive_resistance = 1.0 / max(1, len(fanout_terms))  
                        break  # 找到第一個 output port 即可  
        except Exception as e:  
            # 簡化的驅動強度計算  
            drive_resistance = 0.0  
            if "xp2" in cell_type:  
                drive_resistance = 0.2
            elif "xp25" in cell_type:  
                drive_resistance = 0.25
            elif "xp33" in cell_type:  
                drive_resistance = 0.33  
            elif "xp5" in cell_type:  
                drive_resistance = 0.5  
            elif "xp67" in cell_type:  
                drive_resistance = 0.67 
            elif "xp75" in cell_type:  
                drive_resistance = 0.75
            elif "x1" in cell_type:  
                drive_resistance = 1.0  
            elif "x1p5" in cell_type:  
                drive_resistance = 1.5
            elif "x2" in cell_type:  
                drive_resistance = 2.0
            elif "x3" in cell_type:  
                drive_resistance = 3.0  
            elif "x4" in cell_type:  
                drive_resistance = 4.0
            elif "x6" in cell_type:  
                drive_resistance = 6.0

            logger.warning(f"Liberty-based 驅動強度計算失敗: {e}") 
                    
        
        try:
            # 同時保留原本的 tuple 格式用於排序
            cell_info_list.append((  
                inst_name,                # Instance  0
                cell_type,                # Cell Type    1
                total_power,              # Total Power  2
                static_power_total,       # Leakage Power  3
                dynamic_power_total,      # Dynamic Power  4
                worst_slack,              # Slack  5
                drive_resistance,         # Drive Strength  6
                vt_type,                  # VT Type   7
                fanout_count,             # Fanout   8
                output_cap,               # Out Cap  9
                input_slew,               # In Slew  10
                master.getWidth(),        # Width  11
                master.getHeight(),       # Height  12
                master.getArea(),          # Area  13
                is_endpoint                # Endpoint  14
            ))

        except Exception as e:
            logger.warning(f"創建 CellInformation 失敗 {inst_name}: {e}")
            continue
            
    # 按功耗排序並顯示  
    if cell_info_list:  # 確保 list 不為空
        score_func = self.create_equal_weight_score(cell_info_list)
        cell_info_list.sort(key=score_func, reverse=True)

    return cell_info_list

def test():
    timing = Timing(design)
    print(dir(timing))

    # input_pin = design.evalTclString("get_pins _248_/A")
    # output_pin = design.evalTclString("get_pins _248_/Y")
    # output_net = design.evalTclString("get_nets -hierarchical -of_objects _248_/Y")
    # print(input_pin)
    # print(output_pin)
    # print(output_net)

    inst = design.getBlock().findInst("_248_")
    print(dir(inst))
    input_pin = inst.findITerm("A")  
    output_pin = inst.findITerm("Y") 

    arrival_rise = timing.getPinArrival(input_pin, Timing.Rise)  
    arrival_fall = timing.getPinArrival(input_pin, Timing.Fall)
    input_arrival = min(arrival_rise, arrival_fall)
    output_arrival_rise = timing.getPinArrival(output_pin, Timing.Rise)
    output_arrival_fall = timing.getPinArrival(output_pin, Timing.Fall)
    output_arrival = max(output_arrival_rise, output_arrival_fall)
    slack = timing.getPinSlack(output_pin, Timing.Rise, Timing.Max)

    slew = timing.getPinSlew(output_pin, Timing.Max)

    print(f"Slack: {slack}")
    print(f"Arrival Rise: {arrival_rise}")
    print(f"Arrival Fall: {arrival_fall}")
    print(f"delay: {output_arrival - input_arrival}")
    print(f"Slew: {slew}")

    # design.evalTclString(f"report_dcalc -from {input_pin} -to {output_pin} -corner tt -max -digits 3")

    # design.evalTclString(f"report_slews -corner tt {output_pin}")

    # design.evalTclString(f"report_net -corner tt -digits 4 _216_")

    pin_commands = [  
        # "get_pins *",  # 獲取所有 pins  
        "get_pins _248_/*",  # 獲取特定 instance 的 pins  
        "[get_property [get_pins _248_/Y] capacitance]",  # 獲取 pin 電容  
        "[get_property [get_pins _248_/Y] slew]"  # 獲取 pin slew  
    ]

    # pins = design.evalTclString("get_pins _1_/*")

    # print(dir(pins))

    # for method in pin_commands:
    #     print(design.evalTclString(method))  # 初始化 result 為空列表

# === 主流程 ===
def main():
    # print(dir(openroad.get_db()))
    # print(dir(design.getDb()))
    pdk_path = os.path.expanduser("~/solution/testcases/ASAP7")
    design_path = os.path.expanduser("~/solution/testcases")
    design, tech = load_case(pdk_path, design_path, "c17")
    
    # analyze_sta_summary()
    # design.evalTclString('global_placement -timing_driven -density 1')
    # design.evalTclString('detailed_placement')
    # design.evalTclString('repair_timing -setup')
    # test()
    # analyze_critical_paths()
    # find_equivalent_cells(tech, design, "O2A1O1Ixp33_ASAP7_75t_L")

    groups = create_cell_groups(tech, design, include_singletons=True)
    save_groups_to_json(groups, "equiv_groups_new.json")

    # design.evalTclString("report_cell_usage")

    # print(design.evalTclString('report_equiv_cells -all "O2A1O1Ixp33_ASAP7_75t_L"'))

    # total_cells = 0  
    # for lib in tech.getDB().getLibs():  
    #     lib_cell_count = 0  
    #     for master in lib.getMasters():  
    #         lib_cell_count += 1  
    #     print(f"Library {lib.getName()}: {lib_cell_count} cells")  
    #     total_cells += lib_cell_count  
    # print(f"Total library cells loaded: {total_cells}")

    # lef_masters = Set()  
    # for lib in tech.getDB().getLibs():  
    #     for master in lib.getMasters():  
    #         lef_masters.add(master.getName())  
    
    # # 獲取所有 Liberty cells (需要通過 STA 接口)  
    # liberty_cells = Set()  
    # # 這需要通過 dbNetwork 來訪問 Liberty 信息  
    
    # # 找出差異  
    # missing_in_liberty = lef_masters - liberty_cells  
    # print(f"LEF 中有但 Liberty 中沒有的 cells: {missing_in_liberty}")

    # replace_instance_cell("_551_", "OAI21xp5_ASAP7_75t_SRAM")
    # print("✅ Replacing cells...")
    # replace_cells()
    # print("✅ Inserting buffers...")
    # insert_buffers()
    # print("✅ Optimizing & reporting...")
    # design.evalTclString("repair_design")
    # MAX_BUFFER_PERCENT = 10.0
    # design.evalTclString(f"repair_timing -setup -hold -max_buffer_percent {MAX_BUFFER_PERCENT} -skip_pin_swap -skip_gate_cloning -skip_buffer_removal")
    # analyze_sta_summary()
    # optimize_and_report()
    # print("✅ Writing output files...")
    # write_output()

main()
