# utils_openroad.py
"""
OpenRoad Integration Utils
整合 OpenROAD 的常駐 session 工具
"""
import os
import re
import logging
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

from openroad import Tech, Design, Timing
import openroad

from cell_replacement_manager import CellReplacementManager

logger = logging.getLogger(__name__)

# -------- Data Classes --------
@dataclass
class MetricsReport:
    tns: float            # Total Negative Slack
    wns: float            # Worst Negative Slack
    total_power: float    # Total Power

@dataclass
class OptimizationAction:
    """動作參數：依 action_type 使用對應欄位
       - insert_buffer：target_net=pin 集合字串、buffer_type
       - replace_cell：target_net=instance 名、new_cell_type
    """
    action_type: str
    target_cell: str
    new_cell_type: Optional[str] = None
    position: Optional[Tuple[float, float]] = None  # 目前未用，可留空

@dataclass
class CellInformation:
    """Cell 相關資訊"""
    cell_type: str
    total_power: float
    static_power_total: float
    worst_slack: float
    drive_resistance: float
    vt_type: str
    fanout_count: int
    output_cap: float
    input_slew: float
    width: float
    height: float
    area: float

# -------- OpenROAD Interface --------
class OpenRoadInterface:
    """OpenROAD 常駐 session 介面：每個 case 只載入一次"""
    def __init__(self,
                 work_dir: str = "/tmp/openroad_work",
                 pdk_root: str = "~/solution/testcases/ASAP7",
                 design_root: str = "~/solution/testcases",
                 max_buffer_percent: float = 10.0,
                 auto_repair_each_step: bool = True,
                 cell_groups_json: str = "/root/cell_groups.json"):
        self.work_dir = os.path.abspath(os.path.expanduser(work_dir))
        self.pdk_root = os.path.abspath(os.path.expanduser(pdk_root))
        self.design_root = os.path.abspath(os.path.expanduser(design_root))
        self.max_buffer_percent = max_buffer_percent
        self.auto_repair_each_step = auto_repair_each_step
        os.makedirs(self.work_dir, exist_ok=True)

        # case_name -> {"metrics": MetricsReport, "design": Design, "cell_information": Dict[str, CellInformation]}
        self.sessions: Dict[str, Dict[str, object]] = {}
        self.tech = self._load_tech_and_libs()
        
        # Cell replacement manager
        self.cell_replacement_manager = CellReplacementManager(cell_groups_json)

    # ---- Helpers ----
    def has_case(self, case_name: str) -> bool:
        return case_name in self.sessions

    def get_design(self, case_name: str) -> Design:
        return self.sessions[case_name]["design"]  # type: ignore[return-value]
    
    def get_cell_information(self, case_name: str, instance_name: str = None) -> Union[Dict[str, CellInformation], CellInformation, None]:
        """取得 cell information
        Args:
            case_name: case 名稱
            instance_name: 如果指定，返回特定 instance 的資訊；否則返回所有 cell 的字典
        """
        cell_info_dict = self.sessions[case_name]["cell_information"]
        if instance_name:
            return cell_info_dict.get(instance_name)
        return cell_info_dict

    # ---- PDK / LEF / LIB ----
    def _load_tech_and_libs(self) -> Tech:
        tech = Tech()
        tech.readLef(f"{self.pdk_root}/techlef/asap7_tech_1x_201209.lef")
        for lef in [
            "LEF/asap7sc7p5t_28_L_1x_220121a.lef",
            "LEF/asap7sc7p5t_28_R_1x_220121a.lef",
            "LEF/asap7sc7p5t_28_SL_1x_220121a.lef",
            "LEF/asap7sc7p5t_28_SRAM_1x_220121a.lef",
            "LEF/sram_asap7_16x256_1rw.lef",
            "LEF/sram_asap7_32x256_1rw.lef",
            "LEF/sram_asap7_64x256_1rw.lef",
            "LEF/sram_asap7_64x64_1rw.lef",
        ]:
            tech.readLef(f"{self.pdk_root}/{lef}")
        for lib in [
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
        ]:
            tech.readLiberty(f"{self.pdk_root}/{lib}")
        return tech

    # ---- Case Load (once) ----
    def load_case(self, case_name: str):
        if case_name in self.sessions:
            return
        def_path = os.path.join(self.design_root, case_name, f"{case_name}_placed.def")
        sdc_path = os.path.join(self.design_root, case_name, f"{case_name}_orig_gtlvl.sdc")
        if not os.path.isfile(def_path):
            raise FileNotFoundError(f"DEF 不存在: {def_path}")
        if not os.path.isfile(sdc_path):
            raise FileNotFoundError(f"SDC 不存在: {sdc_path}")

        design = Design(self.tech)
        design.readDef(def_path)
        design.evalTclString(f"read_sdc {sdc_path}")
        design.evalTclString(f"source {self.pdk_root}/setRC.tcl")
        design.evalTclString("estimate_parasitics -placement")

        self.sessions[case_name] = {"design": design, "metrics": MetricsReport(0.0, 0.0, 0.0), "cell_information": {}}

    # ---- Apply Action ----
    def apply_action(self, case_name: str, action: OptimizationAction) -> bool:
        design: Design = self.sessions[case_name]["design"]
        try:
            if action.action_type == "replace_cell":
                # target_cell 為 instance 名稱
                db = design.getDb()
                block = design.getBlock()
                inst = block.findInst(action.target_cell)
                # print("cell_name :", action.new_cell_type)

                original_master = inst.getMaster()  
                original_cell_name = original_master.getName()  
                # print(f"Original cell: {original_cell_name}")

                print(f"original cell : {original_cell_name} new cell : {action.new_cell_type}")
                if not inst:
                    logger.error(f"Instance {action.target_cell} not found.")
                    return False
                master = db.findMaster(action.new_cell_type)
                if not master:
                    logger.error(f"Master cell {action.new_cell_type} not found.")
                    return False
                print(inst.swapMaster(master))
                logger.info(f"Replaced instance {action.target_cell} with master {action.new_cell_type}.")
            
            elif action.action_type == "auto_replace_cell":
                design.evalTclString('set_dont_touch [get_cells "*"]')
                design.evalTclString(f"unset_dont_touch [get_cells {action.target_cell}]")
                design.evalTclString("repair_timing -setup -skip_buffering -skip_buffer_removal -skip_gate_cloning -skip_pin_swap")
            
            return True

        except Exception as e:
            logger.exception(f"apply_action 失敗：{e}")
            
            return False

    # ---- Report ----
    def report_metrics(self, case_name: str) -> MetricsReport:
        design: Design = self.sessions[case_name]["design"]
        metrics: MetricsReport = self.sessions[case_name]["metrics"]
        total_power = 0.0

        try:  
            # 創建 Timing 物件  
            timing = Timing(design)  

            for inst in design.getBlock().getInsts():  
                for corner in timing.getCorners():  
                    static_power = timing.staticPower(inst, corner)  
                    dynamic_power = timing.dynamicPower(inst, corner)  
                    total_power += static_power + dynamic_power  
            
            # 使用 evalTclString 獲取 TNS 和 WNS  
            tns = float(design.evalTclString("total_negative_slack -max"))  
            wns = float(design.evalTclString("worst_negative_slack -max"))  
            
        except Exception as e:  
            print(f"Error parsing STA results: {e}")  
            return MetricsReport(0.0, 0.0, 0.0)
        
        print(f"Power: {total_power} W")  
        print(f"TNS  : {tns} ns")  
        print(f"WNS  : {wns} ns")

        metrics.tns = tns
        metrics.wns = wns
        metrics.total_power = total_power

        return metrics

    def create_equal_weight_score(self, cell_info_list):  
        # 提取並標準化各項指標  
        powers = [x[2] for x in cell_info_list]  # 功耗值  
        slacks = [abs(x[4]) for x in cell_info_list]  # 使用絕對值的 slack  
        
        max_power = max(powers) if powers else 1  
        max_slack = max(slacks) if slacks else 1  
        
        # 設置相等權重  
        power_weight = 0.5  
        slack_weight = 0.5  
        
        def optimization_score(x):  
            power_norm = x[2] / max_power  
            slack_norm = abs(x[4]) / max_slack  
            
            return power_weight * power_norm + slack_weight * slack_norm  
        
        return optimization_score

    def update_cell_information(self, case_name: str):
        design = self.sessions[case_name]["design"]
        timing = Timing(design) 
        cell_info_list = []  # 初始化 cell_info_list
        
        for inst in design.getBlock().getInsts():  
            inst_name = inst.getName()  
            master = inst.getMaster()  
            cell_type = master.getName()  
            
            total_power = 0  
            static_power_total = 0  
            worst_slack = float('inf')  
            
            # 功耗分析  
            for corner in timing.getCorners():  
                static_power = timing.staticPower(inst, corner)  
                dynamic_power = timing.dynamicPower(inst, corner)  
                total_power += static_power + dynamic_power  
                static_power_total += static_power  
                
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
            
            # 電氣特性分析  
            output_cap = 0  
            input_slew = 0  
            fanout_count = 0  
            
            for iTerm in inst.getITerms():  
                if not iTerm.getNet():  
                    continue  
                # Slack 分析  
                rise_max_slack = timing.getPinSlack(iTerm, Timing.Rise, Timing.Max)  
                fall_max_slack = timing.getPinSlack(iTerm, Timing.Fall, Timing.Max)  
                pin_worst_slack = min(rise_max_slack, fall_max_slack)  
                worst_slack = min(worst_slack, pin_worst_slack)  
                
                # 電氣特性  
                mterm = iTerm.getMTerm()  
                if mterm.getIoType() == "OUTPUT":  
                    output_cap = timing.getMaxCapLimit(mterm)
                    # 計算 fanout 數量  
                    net = iTerm.getNet()  
                    if net:  
                        fanout_count = len(list(net.getITerms())) - 1   
                elif mterm.getIoType() == "INPUT":  
                    try:  
                        input_slew = timing.getMaxSlewLimit(mterm)  
                    except:  
                        pass  

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
            
            if worst_slack != float('inf'):  
                # 創建 CellInformation 物件
                cell_info = CellInformation(
                    cell_type=cell_type,
                    total_power=total_power,
                    static_power_total=static_power_total,
                    worst_slack=worst_slack,
                    drive_resistance=drive_resistance,
                    vt_type=vt_type,
                    fanout_count=fanout_count,
                    output_cap=output_cap,
                    input_slew=input_slew,
                    width=master.getWidth(),
                    height=master.getHeight(),
                    area=master.getArea()
                )
                
                # 存儲到字典中，以 instance 名稱為 key
                self.sessions[case_name]["cell_information"][inst_name] = cell_info
                
                # 同時保留原本的 tuple 格式用於排序
                cell_info_list.append((  
                    inst_name,                # Instance  0
                    cell_type,                # Cell Type    1
                    total_power,              # Total Power  2
                    static_power_total,       # Leakage Power  3
                    worst_slack,              # Slack  4
                    drive_resistance,         # Drive Strength  5
                    vt_type,                  # VT Type   6
                    fanout_count,             # Fanout   7
                    output_cap,               # Out Cap  8
                    input_slew,               # In Slew  9
                    master.getWidth(),        # Width  10
                    master.getHeight(),       # Height  11
                    master.getArea()          # Area  12
                ))  
        
        # 按功耗排序並顯示  
        if cell_info_list:  # 確保 list 不為空
            score_func = self.create_equal_weight_score(cell_info_list)
            cell_info_list.sort(key=score_func, reverse=True)

        return cell_info_list

    def get_candidate_cells(self, case_name: str, top_slack: int = 50, top_power: int = 50) -> List[Tuple[str, str]]:
        """
        獲取候選 cell 集合：Top-P 最壞 slack + Top-H 最大功耗的 cells
        
        Args:
            case_name: case 名稱
            top_slack: 取 slack 最壞的前 N 個 cells
            top_power: 取功耗最大的前 N 個 cells
            
        Returns:
            候選 cell 的 (instance_name, cell_type) 對列表
        """
        cell_info_dict = self.sessions[case_name]["cell_information"]
        
        if not cell_info_dict:
            # 如果還沒有 cell information，先更新
            self.update_cell_information(case_name)
            cell_info_dict = self.sessions[case_name]["cell_information"]
        
        # 按 worst slack 排序 (越小越壞)
        cells_by_slack = sorted(
            cell_info_dict.items(), 
            key=lambda x: x[1].worst_slack
        )
        
        # 按 power 排序 (越大越耗電)
        cells_by_power = sorted(
            cell_info_dict.items(),
            key=lambda x: x[1].total_power,
            reverse=True
        )
        
        # 取 top candidates 的 instance names
        slack_candidates = [name for name, _ in cells_by_slack[:top_slack]]
        power_candidates = [name for name, _ in cells_by_power[:top_power]]
        
        # 合併並去重 instance names
        candidate_instances = list(set(slack_candidates + power_candidates))
        
        # 返回 (instance_name, cell_type) 對
        candidate_pairs = []
        for inst_name in candidate_instances:
            cell_info = cell_info_dict[inst_name]
            candidate_pairs.append((inst_name, cell_info.cell_type))
        
        return candidate_pairs

    def get_dynamic_features(self, case_name: str) -> np.ndarray:
        """
        計算每個 cell 的動態特徵
        
        Returns:
            [N, F_dyn] 的 numpy array，每一行是一個 cell 的動態特徵
        """
        cell_info_dict = self.sessions[case_name]["cell_information"]
        
        if not cell_info_dict:
            self.update_cell_information(case_name)
            cell_info_dict = self.sessions[case_name]["cell_information"]
        
        # 構建特徵矩陣
        features = []
        cell_names = []
        
        for inst_name, cell_info in cell_info_dict.items():
            # 動態特徵向量 (可以根據需要調整)
            feature_vector = [
                cell_info.worst_slack,              # local slack
                cell_info.total_power,              # power consumption
                cell_info.static_power_total,       # static power
                cell_info.drive_resistance,         # drive strength
                cell_info.fanout_count,             # fanout count
                cell_info.output_cap,               # output capacitance
                cell_info.input_slew,               # input slew
                cell_info.area,                     # cell area
                # VT type (one-hot encoding)
                0.5 if cell_info.vt_type == "L" else 0.0,      # LVT
                0.75 if cell_info.vt_type == "R" else 0.0,      # RVT  
                0.25 if cell_info.vt_type == "SL" else 0.0,     # SLVT
                1.0 if cell_info.vt_type == "SRAM" else 0.0,   # SRAM
            ]
            
            features.append(feature_vector)
            cell_names.append(inst_name)
        
        return np.array(features, dtype=np.float32), cell_names

        # print("Top 10 Power Consuming Cells - Complete Optimization Analysis:")  
        # print("Rank | Instance | Cell Type    | Power   | Leakage | Slack  | Drive | VT  | Fanout | Cap   | Slew  | Width | Height | Area")  
        # print("-" * 140)  

        # for i, (inst_name, cell_type, power, leakage, slack, drive, vt, fanout, cap, slew, width, height, area) in enumerate(cell_info_list[:10]):  
        #     print(f"{i+1:4d} | {inst_name:8s} | {cell_type:12s} | {power:.2e} | {leakage:.2e} | {slack:.3f} | {drive:.1f} | {vt:4s} | {fanout:6d} | {cap:.3e} | {slew:.3e} | {width:5d} | {height:6d} | {area:4d}")

if __name__ == "__main__":
    # 測試用：確保 OpenROAD 可以正常載入
    benchmark = "c17"
    interface = OpenRoadInterface()
    interface.load_case(benchmark)
    cell = interface.update_cell_information(benchmark)
    print(cell)
    print()
    print(interface.apply_action(benchmark, OptimizationAction(action_type = "replace_cell", target_cell = "_1_", new_cell_type = "NAND2x1_ASAP7_75t_L")))
    cell = interface.update_cell_information(benchmark)
    print(cell)
    print()
    report = interface.report_metrics(benchmark)
    print(f"TNS: {report.tns}, WNS: {report.wns}, Power: {report.total_power}")
