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
    delay: float
    drive_resistance: float
    vt_type: float  # 改為數值型別
    fanout_count: int
    output_cap: float
    output_slew: float
    area: float
    is_endpoint: bool

# -------- OpenROAD Interface --------
class OpenRoadInterface:
    """OpenROAD 常駐 session 介面：每個 case 只載入一次"""
    def __init__(self,
                 work_dir: str = "/tmp/openroad_work",
                 pdk_root: str = "~/solution/testcases/ASAP7",
                 design_root: str = "~/solution/testcases",
                 max_buffer_percent: float = 10.0,
                 auto_repair_each_step: bool = True,
                 cell_groups_json: str = "/root/ruan_workspace/ic_cad/gnn/cell_groups.json"):
        self.work_dir = os.path.abspath(os.path.expanduser(work_dir))
        self.pdk_root = os.path.abspath(os.path.expanduser(pdk_root))
        self.design_root = os.path.abspath(os.path.expanduser(design_root))
        self.max_buffer_percent = max_buffer_percent
        self.auto_repair_each_step = auto_repair_each_step
        os.makedirs(self.work_dir, exist_ok=True)

        # case_name -> {"design": Design, "tech": Tech, "metrics": MetricsReport, "cell_information": Dict[str, CellInformation]}
        self.sessions: Dict[str, Dict[str, object]] = {}
        # 不再在初始化時載入共享 tech，而是為每個案例創建獨立的 tech
        # self.tech = self._load_tech_and_libs()  # 註解掉
        
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

    # ---- PDK / LEF / LIB ---- (已棄用，現在每個案例獨立載入)
    def _load_tech_and_libs(self) -> Tech:
        """已棄用：現在為每個案例創建獨立的 Tech 物件"""
        logger.warning("⚠️  _load_tech_and_libs 已棄用，現在每個案例使用獨立的 Tech 物件")
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
            design = self.sessions[case_name]["design"]  
            design.evalTclString("update_timing")  
            # design.evalTclString("estimate_parasitics -placement")
            logger.info(f"📋 案例 {case_name} 已在記憶體中，直接使用")
            return
            
        logger.info(f"🔄 載入新案例: {case_name}")
        def_path = os.path.join(self.design_root, case_name, f"{case_name}.def")
        sdc_path = os.path.join(self.design_root, case_name, f"{case_name}.sdc")
        if not os.path.isfile(def_path):
            raise FileNotFoundError(f"DEF 不存在: {def_path}")
        if not os.path.isfile(sdc_path):
            raise FileNotFoundError(f"SDC 不存在: {sdc_path}")

        try:
            # 為每個案例創建獨立的 Tech 和 Design 物件
            case_tech = Tech()
            case_tech.readLef(f"{self.pdk_root}/techlef/asap7_tech_1x_201209.lef")
            for lef in [
                "LEF/asap7sc7p5t_28_L_1x_220121a.lef",
                "LEF/asap7sc7p5t_28_R_1x_220121a.lef", 
                "LEF/asap7sc7p5t_28_SL_1x_220121a.lef",
                "LEF/asap7sc7p5t_28_SRAM_1x_220121a.lef"
            ]:
                case_tech.readLef(f"{self.pdk_root}/{lef}")
            
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
                case_tech.readLiberty(f"{self.pdk_root}/{lib}")
            
            # 創建獨立的 Design 物件
            design = Design(case_tech)
            design.readDef(def_path)
            design.evalTclString(f"read_sdc {sdc_path}")
            design.evalTclString(f"source {self.pdk_root}/setRC.tcl")
            design.evalTclString("estimate_parasitics -placement")

            self.sessions[case_name] = {
                "design": design, 
                "tech": case_tech,  # 保存 tech 引用避免被垃圾回收
                "metrics": MetricsReport(0.0, 0.0, 0.0), 
                "state_norm": {"power": [float("inf"), float("-inf")], "capacitance": [float("inf"), float("-inf")], "slew": [float("inf"), float("-inf")], "area": [43740, 700380], "fanout_count": [float("inf"), float("-inf")], "delay": [float("inf"), float("-inf")]},  # 初始範圍
                "cell_information": {}
            }
            logger.info(f"✅ 成功載入案例: {case_name}")
            
        except Exception as e:
            logger.error(f"❌ 載入案例 {case_name} 失敗: {e}")
            # 如果載入失敗，清理可能的部分狀態
            if case_name in self.sessions:
                del self.sessions[case_name]
            raise

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

                print(f"original cell : {original_cell_name} new cell : {action.new_cell_type} target cell : {action.target_cell}")
                if not inst:
                    logger.error(f"Instance {action.target_cell} not found.")
                    return False
                master = db.findMaster(action.new_cell_type)
                if not master:
                    logger.error(f"Master cell {action.new_cell_type} not found.")
                    return False
                inst.swapMaster(master)

                design.evalTclString("update_timing") 

                # logger.info(f"Replaced instance {action.target_cell} with master {action.new_cell_type}.")
            
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
            design.evalTclString("update_timing")
            # design.evalTclString("estimate_parasitics -placement")
            timing = Timing(design)  

            # 更安全的功耗計算
            try:
                for inst in design.getBlock().getInsts():  
                    for corner in timing.getCorners():  
                        try:
                            static_power = timing.staticPower(inst, corner)  
                            dynamic_power = timing.dynamicPower(inst, corner)  
                            total_power += static_power + dynamic_power  
                        except Exception as e:
                            logger.debug(f"功耗計算失敗 {inst.getName()}: {e}")
                            continue
            except Exception as e:
                logger.warning(f"功耗分析失敗: {e}")
                total_power = 0.0
            
            # 使用 evalTclString 獲取 TNS 和 WNS  
            try:
                tns = float(design.evalTclString("total_negative_slack -max"))  
            except Exception as e:
                logger.warning(f"TNS 計算失敗: {e}")
                tns = 0.0
                
            try:
                wns = float(design.evalTclString("worst_negative_slack -max"))  
            except Exception as e:
                logger.warning(f"WNS 計算失敗: {e}")
                wns = 0.0
            
        except Exception as e:  
            logger.error(f"Timing 分析失敗: {e}")  
            return MetricsReport(0.0, 0.0, 0.0)
        
        print(f"Power: {total_power} W")  
        print(f"TNS  : {tns} ns")  
        print(f"WNS  : {wns} ns")
        print()

        metrics.tns = tns
        metrics.wns = wns
        metrics.total_power = total_power

        return metrics

    def create_equal_weight_score(self, cell_info_list):  
        # 提取並標準化各項指標  
        powers = [x[2] for x in cell_info_list]  # 功耗值  
        delay = [x[3] for x in cell_info_list]  # 使用絕對值的 delay
        slew = [x[8] for x in cell_info_list]  # 使用絕對值的 output slew
        
        max_power = max(powers) if powers else 1  
        max_delay = max(delay) if delay else 1  
        max_slew = max(slew) if slew else 1

        # 設置相等權重  
        power_weight = 0.5  
        delay_weight = 0.5  
        slew_weight = 0.5

        def optimization_score(x):  
            power_norm = x[2] / max_power  
            delay_norm = x[3] / max_delay  
            slew_norm = x[8] / max_slew

            return power_weight * power_norm + delay_weight * delay_norm + slew_weight * slew_norm  
        
        return optimization_score

    def update_cell_information(self, case_name: str):
        design = self.sessions[case_name]["design"]
        tech = design.getTech()
        sta = tech.getSta()
        design.evalTclString("update_timing")

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
            worst_slack = float('inf')  
            
            try:
                # 功耗分析 - 更安全的實現
                for corner in timing.getCorners():  
                    try:
                        static_power = timing.staticPower(inst, corner)  
                        dynamic_power = timing.dynamicPower(inst, corner)  
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
                vt_type = 0.0  
            elif "_SL" in cell_type:
                vt_type = 0.25
            elif "_R" in cell_type:  
                vt_type = 0.5 
            elif "_L" in cell_type:  
                vt_type = 1.0
            
            # 電氣特性分析 - 更安全的實現
            output_cap = 0  
            output_slew = 0  
            fanout_count = 0  
            delay = 0.0
            is_endpoint = False
            
            try:
                input_arrival_rise = 0.0
                input_arrival_fall = 0.0
                input_arrival = 0.0
                output_arrival_rise = 0.0
                output_arrival_fall = 0.0
                output_arrival = 0.0
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
                        # 電氣特性  
                        mterm = iTerm.getMTerm()  
                        pin_name = mterm.getName()
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
                                
                            try:
                                output_pin = inst.findITerm(pin_name)
                                output_arrival_rise = timing.getPinArrival(output_pin, Timing.Rise)
                                output_arrival_fall = timing.getPinArrival(output_pin, Timing.Fall)
                                output_arrival = max(output_arrival_rise, output_arrival_fall)
                            except Exception as e:
                                logger.debug(f"Pin arrival 取得失敗 {inst_name}/{pin_name}: {e}")
                                output_arrival = 0.0

                            try:
                                output_slew = timing.getPinSlew(output_pin, Timing.Max)
                            except Exception as e:
                                logger.debug(f"Pin slew 取得失敗 {inst_name}/{pin_name}: {e}")
                                output_slew = 0.0

                            # 計算 fanout 數量  
                            net = iTerm.getNet()  
                            if net:  
                                fanout_count = len(list(net.getITerms())) - 1   
                                
                        elif mterm.getIoType() == "INPUT":  
                            try:  
                                input_pin = inst.findITerm(pin_name)
                                input_arrival_rise = timing.getPinArrival(input_pin, Timing.Rise)
                                input_arrival_fall = timing.getPinArrival(input_pin, Timing.Fall)
                                input_arrival = min(input_arrival, input_arrival_rise, input_arrival_fall)
                            except Exception as e:  
                                logger.debug(f"Pin arrival 取得失敗 {inst_name}/{pin_name}: {e}")
                                input_arrival = 0.0

                    except Exception as e:
                        logger.debug(f"電氣特性分析失敗 {inst_name}: {e}")
                        continue

                delay = output_arrival - input_arrival

            except Exception as e:
                logger.warning(f"ITerm 分析失敗 {inst_name}: {e}")

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
            
            try:
                # 做一些數值修正
                output_cap *= 1e15  
                output_slew *= 1e12
                delay *= 1e12

                # 更新最大/最小範圍
                norm = self.sessions[case_name]["state_norm"]
                norm["power"][0] = min(norm["power"][0], total_power)
                norm["power"][1] = max(norm["power"][1], total_power)
                norm["capacitance"][0] = min(norm["capacitance"][0], output_cap)
                norm["capacitance"][1] = max(norm["capacitance"][1], output_cap)
                norm["slew"][0] = min(norm["slew"][0], output_slew)
                norm["slew"][1] = max(norm["slew"][1], output_slew)
                norm["fanout_count"][0] = min(norm["fanout_count"][0], fanout_count)
                norm["fanout_count"][1] = max(norm["fanout_count"][1], fanout_count)
                norm["delay"][0] = min(norm["delay"][0], delay)
                norm["delay"][1] = max(norm["delay"][1], delay)

                # 創建 CellInformation 物件 - 更安全的實現
                cell_info = CellInformation(
                    cell_type=cell_type,
                    total_power=total_power,
                    delay=delay,
                    drive_resistance=drive_resistance,
                    vt_type=vt_type,
                    fanout_count=fanout_count,
                    output_cap=output_cap,
                    output_slew=output_slew,
                    area=master.getArea(),
                    is_endpoint=is_endpoint 
                )
                
                # 存儲到字典中，以 instance 名稱為 key
                self.sessions[case_name]["cell_information"][inst_name] = cell_info

                # 同時保留原本的 tuple 格式用於排序
                cell_info_list.append((  
                    inst_name,                # Instance  0
                    cell_type,                # Cell Type    1
                    total_power,              # Total Power  2
                    delay,                    # Delay  3
                    drive_resistance,         # Drive Strength  4
                    vt_type,                  # VT Type   5
                    fanout_count,             # Fanout   6
                    output_cap,               # Out Cap  7
                    output_slew,              # Out Slew  8
                    master.getArea(),         # Area  9
                    is_endpoint               # Endpoint  10
                ))

            except Exception as e:
                logger.warning(f"創建 CellInformation 失敗 {inst_name}: {e}")
                continue
                
        # 按功耗排序並顯示  
        if cell_info_list:  # 確保 list 不為空
            score_func = self.create_equal_weight_score(cell_info_list)
            cell_info_list.sort(key=score_func, reverse=True)

        return cell_info_list

    def get_candidate_cells(self, case_name: str, top_delay: int = 15, top_power: int = 15, top_slew: int = 15) -> List[Tuple[str, str]]:
        """
        獲取候選 cell 集合：Top-P 最壞 delay + Top-H 最大功耗的 cells + Top-S 最大輸出 slew
        這些 cell 將作為優化的目標
        
        Args:
            case_name: case 名稱
            top_delay: 取 delay 最壞的前 N 個 cells
            top_power: 取功耗最大的前 N 個 cells
            top_slew: 取輸出 slew 最大的前 N 個 cells
            
        Returns:
            候選 cell 的 (instance_name, cell_type) 對列表
        """
        cell_info_dict = self.sessions[case_name]["cell_information"]
        
        if not cell_info_dict:
            # 如果還沒有 cell information，先更新
            self.update_cell_information(case_name)
            cell_info_dict = self.sessions[case_name]["cell_information"]

        # 按 worst delay 排序 (越大越壞)
        cells_by_delay = sorted(
            cell_info_dict.items(),
            key=lambda x: x[1].delay,
            reverse=True
        )
        
        # 按 power 排序 (越大越耗電)
        cells_by_power = sorted(
            cell_info_dict.items(),
            key=lambda x: x[1].total_power,
            reverse=True
        )

        cells_by_output_slew = sorted(
            cell_info_dict.items(),
            key=lambda x: x[1].output_slew,
            reverse=True
        )
        
        # 取 top candidates 的 instance names
        delay_candidates = [name for name, _ in cells_by_delay[:top_delay]]
        power_candidates = [name for name, _ in cells_by_power[:top_power]]
        slew_candidates = [name for name, _ in cells_by_output_slew[:top_slew]]

        # 合併並去重 instance names
        candidate_instances = list(set(delay_candidates + power_candidates + slew_candidates))

        # 返回 (instance_name, cell_type) 對
        candidate_pairs = []
        for inst_name in candidate_instances:
            cell_info = cell_info_dict[inst_name]
            candidate_pairs.append((inst_name, cell_info.cell_type))
        
        return candidate_pairs

    def normalize_feature(self, values: float, min_val: float, max_val: float, negative: bool) -> float:
        """將特徵值標準化到 [0, 1] 範圍"""
        if max_val - min_val == 0:
            return 0.0
        
        if negative:
            return 2 * (values - min_val) / (max_val - min_val) - 1
        else:
            return (values - min_val) / (max_val - min_val)

    def get_dynamic_features(self, case_name: str) -> np.ndarray:
        """
        計算每個 cell 的動態特徵
        
        Returns:
            [N, F_dyn] 的 numpy array，每一行是一個 cell 的動態特徵
        """
        cell_info_dict = self.sessions[case_name]["cell_information"]
        norm = self.sessions[case_name]["state_norm"]
        
        if not cell_info_dict:
            self.update_cell_information(case_name)
            cell_info_dict = self.sessions[case_name]["cell_information"]
        
        # 構建特徵矩陣
        features = []
        cell_names = []
        
        for inst_name, cell_info in cell_info_dict.items():
            # 動態特徵向量 (可以根據需要調整)
            total_power = self.normalize_feature(cell_info.total_power, norm["power"][0], norm["power"][1], negative=False)
            delay = self.normalize_feature(cell_info.delay, norm["delay"][0], norm["delay"][1], negative=False)  # 假設 delay 範圍與 slew 相似，且越小越好
            drive_resistance = self.normalize_feature(cell_info.drive_resistance, 0.2, 6.0, negative=False)  # 假設驅動強度範圍在 0.2 到 6.0
            fanout_count = self.normalize_feature(cell_info.fanout_count, norm["fanout_count"][0], norm["fanout_count"][1], negative=False)
            output_cap = self.normalize_feature(cell_info.output_cap, norm["capacitance"][0], norm["capacitance"][1], negative=False)
            output_slew = self.normalize_feature(cell_info.output_slew, norm["slew"][0], norm["slew"][1], negative=False)
            area = self.normalize_feature(cell_info.area, norm["area"][0], norm["area"][1], negative=False)
            
            # VT type 已經是數值，直接使用
            vt_type_value = cell_info.vt_type
            
            feature_vector = [
                total_power,                                            # power consumption
                delay,                                                  # delay
                drive_resistance,                                       # drive strength
                vt_type_value,                                          # VT type (numerical)
                fanout_count,                                           # fanout count
                output_cap,                                             # output capacitance
                output_slew,                                            # input slew
                area,                                                   # cell area
                1.0 if cell_info.is_endpoint else 0.0,                 # is timing endpoint
            ]
            
            features.append(feature_vector)
            cell_names.append(inst_name)
        
        

        # print("Top 10 Power Consuming Cells - Complete Optimization Analysis:")  
        # print("Rank | Instance | Cell Type    | Power   | Leakage | Slack  | Drive | VT  | Fanout | Cap   | Slew  | Width | Height | Area")  
        # print("-" * 140)  

        # for i, (inst_name, cell_type, power, leakage, slack, drive, vt, fanout, cap, slew, width, height, area) in enumerate(cell_info_list[:10]):  
        #     print(f"{i+1:4d} | {inst_name:8s} | {cell_type:12s} | {power:.2e} | {leakage:.2e} | {slack:.3f} | {drive:.1f} | {vt:4s} | {fanout:6d} | {cap:.3e} | {slew:.3e} | {width:5d} | {height:6d} | {area:4d}")

        return np.array(features, dtype=np.float32), cell_names

if __name__ == "__main__":
    # 測試用：確保 OpenROAD 可以正常載入
    benchmark = "c17"
    interface = OpenRoadInterface()
    interface.load_case(benchmark)
    interface.report_metrics(benchmark)
    interface.update_cell_information(benchmark)
    cells = interface.get_dynamic_features(benchmark)
    for cell in cells:
        print(cell)

    # cells = interface.get_candidate_cells(benchmark, top_slack=15, top_power=5)
    # print(len(cells))
    # for i in cells:
    #     print(i)

    # cell = interface.update_cell_information(benchmark)
    # print(cell)
    # print()
    # print(interface.apply_action(benchmark, OptimizationAction(action_type = "replace_cell", target_cell = "_1_", new_cell_type = "NAND2x1_ASAP7_75t_L")))
    # cell = interface.update_cell_information(benchmark)
    # print(cell)
    # print()
    # report = interface.report_metrics(benchmark)
    # print(f"TNS: {report.tns}, WNS: {report.wns}, Power: {report.total_power}")
