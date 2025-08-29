from openroad import Tech, Design
import openroad
import os
import json
import re
from typing import Set

# === 建立技術與設計物件 ===
tech = Tech()
pdk_path = os.path.expanduser("~/solution/testcases/ASAP7")
design_path = os.path.expanduser("~/solution/testcases")

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

# for lib_file in lib_files:  
#     print(f"Loading: {lib_file}")  
#     tech.readLiberty (f"{pdk_path}/{lib_file}")  
#     print(f"Successfully loaded: {lib_file}")


benchmark = "s1488"
# === 載入設計與 RC 資訊 ===
design = Design(tech)
design.readDef(f"{design_path}/{benchmark}/{benchmark}_placed.def")
design.evalTclString(f"read_sdc {design_path}/{benchmark}/{benchmark}_orig_gtlvl.sdc")
design.evalTclString(f"source {pdk_path}/setRC.tcl")
design.evalTclString("estimate_parasitics -placement") 
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

    power_str = design.evalTclString("report_power -corner default -digits 3")
    tns_str = design.evalTclString("report_tns -digits 3")
    wns_str = design.evalTclString("report_wns -digits 3")

    power = extract_number(power_str)
    tns = extract_number(tns_str)
    wns = extract_number(wns_str)

    # print(f"📊 Power: {power} uW")
    # print(f"📉 TNS  : {tns} ns")
    # print(f"📉 WNS  : {wns} ns")

    return power, tns, wns

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



# === 主流程 ===
def main():
    analyze_sta_summary()

    # find_equivalent_cells(tech, design, "O2A1O1Ixp33_ASAP7_75t_L")

    groups = create_cell_groups(tech, design, include_singletons=True)
    save_groups_to_json(groups, "equiv_groups.json")

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
    analyze_sta_summary()
    # optimize_and_report()
    print("✅ Writing output files...")
    write_output()

main()
