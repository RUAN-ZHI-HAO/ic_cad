import json
import torch
from torch_geometric.data import Data
import os
import sys

# === 動態路徑設定：允許相對於專案根的 parser 匯入 ===
PROJECT_ROOT = os.environ.get('IC_CAD_ROOT', '/root/ruan_workspace/ic_cad')
PARSER_DIR = os.path.join(PROJECT_ROOT, 'parser')
if PARSER_DIR not in sys.path:
    sys.path.insert(0, PARSER_DIR)
from my_parser import parse_nodes, parse_nets, parse_pl, set_net_pin

# === Cell ID 映射管理 ===
def get_or_create_cell_id_mapping(cell_groups_path="/root/ruan_workspace/ic_cad/gnn/cell_groups.json", 
                                 mapping_path="/root/ruan_workspace/ic_cad/gnn/cell_id_mapping.json"):
    """
    獲取或創建 cell ID 映射
    基於 cell_groups.json 建立所有 cell 的索引映射
    """
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r') as f:
            mapping = json.load(f)
        return mapping
    
    # 從 cell_groups.json 創建映射
    if os.path.exists(cell_groups_path):
        with open(cell_groups_path, 'r') as f:
            cell_groups = json.load(f)
        
        # 收集所有獨特的 cells
        all_cells = set()
        for group in cell_groups:
            all_cells.update(group)
        
        # 添加特殊 cell types
        all_cells.add("terminal_NI")
        all_cells.add("unknown")
        
        # 建立索引映射
        cell_to_id = {cell: idx for idx, cell in enumerate(sorted(all_cells))}
        
        # 保存映射
        with open(mapping_path, 'w') as f:
            json.dump(cell_to_id, f, indent=2)
        
        print(f"📋 創建 cell ID 映射：{len(cell_to_id)} 個 cells")
        return cell_to_id
    
    # 如果沒有 cell_groups.json，使用空映射
    return {"terminal_NI": 0, "unknown": 1}

# === 收集所有設計的cell類型 ===
def collect_all_cell_types(benchmarks, root_dir, verilog_root, library_json_path=None):
    """
    收集所有 cell types - 現在使用靜態 cell_id_mapping 來確保一致性
    """
    # 使用靜態 cell_id_mapping 來確保與 GNN 模型的一致性
    static_mapping = get_or_create_cell_id_mapping()
    global_cell_types = sorted(static_mapping.keys())
    print(f"🌍 使用靜態映射收集到 {len(global_cell_types)} 個全域 cell types")
    return global_cell_types

# === 從 Verilog 擷取 instance to cell type 對應 ===
def extract_cell_type_mapping_from_verilog(verilog_path):
    mapping = {}
    # print(f"📋 Starting to parse Verilog file: {verilog_path}")
    
    with open(verilog_path, 'r') as f:
        content = f.read()
    
    # 使用正則表達式來匹配 Verilog 實例
    import re
    
    # 多種匹配模式來處理不同的格式
    patterns = [
        # Pattern 1: 標準格式 CELL_TYPE INSTANCE_NAME (
        r'^\s*([A-Za-z][A-Za-z0-9_]*)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(',
        # Pattern 2: 處理可能有額外空格的情況
        r'^\s*([A-Za-z0-9_]+(?:x\d+)?_[A-Za-z0-9_]+)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(',
        # Pattern 3: 處理整行的實例宣告
        r'([A-Za-z0-9_]+(?:x\d+)?_[A-Za-z0-9_]+)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\([^;]*\)\s*;'
    ]
    
    lines = content.split('\n')
    found_instances = set()  # 避免重複計算
    
    for pattern_idx, pattern in enumerate(patterns):
        # print(f"🔍 Trying pattern {pattern_idx + 1}: {pattern}")
        pattern_matches = 0
        
        for line_no, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('//'):
                continue
                
            match = re.match(pattern, line)
            if match:
                cell_type = match.group(1)
                instance_name = match.group(2)
                
                # 排除關鍵字和模組定義
                if cell_type not in ['module', 'wire', 'input', 'output', 'assign', 'endmodule', 'reg']:
                    # 只有在之前沒有找到這個實例時才添加
                    if instance_name not in found_instances:
                        mapping[instance_name] = cell_type
                        found_instances.add(instance_name)
                        pattern_matches += 1
                        # if pattern_matches <= 3: # 只顯示前3個範例
                        #     print(f"   Line {line_no}: {instance_name} -> {cell_type}")
        
        # print(f"   Pattern {pattern_idx + 1} found {pattern_matches} new matches")
        
        # 如果已經找到了實例，就不需要嘗試其他模式了
        if len(mapping) > 0 and pattern_idx == 0:
            # print("   ✅ Found matches with first pattern, skipping others")
            break
    
    # print(f"Total extracted cell types: {len(mapping)}")
    
    # if len(mapping) == 0:
    #     print("⚠️  No cell type mappings found! Showing first 10 lines for debugging:")
    #     for i, line in enumerate(lines[:10], 1):
    #         print(f"   {i:2}: {line}")
    
    return mapping

# === Graph 構建主函數 ===
def build_graph_from_case(case_dir: str, verilog_root: str, global_cell_types=None, library_json_path=None):
    case_name = os.path.basename(os.path.normpath(case_dir))

    # 獲取 cell ID 映射
    cell_to_id = get_or_create_cell_id_mapping()

    # 1. Parse 所有資料
    nodes = parse_nodes(os.path.join(case_dir, f"{case_name}.nodes"))
    nets = parse_nets(os.path.join(case_dir, f"{case_name}.nets"))
    pl = parse_pl(os.path.join(case_dir, f"{case_name}.pl"))
    set_net_pin(nets, pl)

    # 使用可配置的 library 路徑
    if library_json_path is None:
        library_json_path = "/root/ruan_workspace/ic_cad/parser/parsed_cells.json"
    
    with open(library_json_path, 'r') as f:
        lib = json.load(f)
    # 2. 擷取 cell type map from verilog
    verilog_path = os.path.join(verilog_root, case_name, f"{case_name}_orig_gtlvl.v")
    # print(f"Reading Verilog file: {verilog_path}")
    
    if not os.path.exists(verilog_path):
        print(f"⚠️  Verilog file not found: {verilog_path}")
        cell_type_map = {}
    else:
        cell_type_map = {}
    if verilog_path:
        cell_type_map = extract_cell_type_mapping_from_verilog(verilog_path)

    # 3. 計算網路連接度統計 (用於後續特徵計算)
    node_degree = {}
    node_fanin = {}
    node_fanout = {}
    net_sizes = {}
    
    # 初始化
    for name in nodes.keys():
        node_degree[name] = 0
        node_fanin[name] = 0
        node_fanout[name] = 0
    
    # 計算連接度和扇入扇出
    for net_name, net_pins in nets.items():
        net_size = len(net_pins)
        net_sizes[net_name] = net_size
        
        # 分析網路拓撲：基於 pin type 來分類
        driving_logic_gates = []  # 在這個網路中驅動信號的邏輯門 (pin type = 'O')
        receiving_logic_gates = []  # 在這個網路中接收信號的邏輯門 (pin type = 'I')
        driving_terminals = []  # 在這個網路中驅動信號的 terminal (輸入端口)
        
        for pin in net_pins:
            cell_name = pin['cell']
            pin_type = pin.get('type', '')
            is_terminal = nodes[cell_name].get('terminal', False)
            
            if cell_name in node_degree:
                node_degree[cell_name] += 1
            
            if is_terminal:
                # Terminal：只有輸出端口 (pin_type='I') 不算作驅動源
                if pin_type == 'O':  # 輸入端口驅動網路
                    driving_terminals.append(cell_name)
            elif cell_name in cell_type_map:
                # 邏輯門
                if pin_type == 'O':  # 邏輯門驅動這個網路
                    driving_logic_gates.append(cell_name)
                elif pin_type == 'I':  # 邏輯門接收這個網路的信號
                    receiving_logic_gates.append(cell_name)
        
        # 去重（一個元件可能有多個pin在同一個網路中）
        driving_logic_gates = list(set(driving_logic_gates))
        receiving_logic_gates = list(set(receiving_logic_gates))
        driving_terminals = list(set(driving_terminals))
        
        # 計算總的驅動源數量（邏輯門 + terminals）
        total_drivers = len(driving_logic_gates) + len(driving_terminals)
        
        # 更新 fanin/fanout：
        # - 驅動邏輯門的 fanout += 接收信號的邏輯門數量 (不包括 terminal)
        # - 接收邏輯門的 fanin += 所有驅動信號的數量 (邏輯門 + terminal)
        for driver in driving_logic_gates:
            node_fanout[driver] += len(receiving_logic_gates)
        
        for receiver in receiving_logic_gates:
            node_fanin[receiver] += total_drivers

    # 3.5 計算每個節點的輸出負載電容
    node_output_load_cap = {}
    
    # 初始化所有節點的輸出負載電容
    for name in nodes.keys():
        node_output_load_cap[name] = 0.0
    
    # print("🔍 開始計算輸出負載電容...")
    
    for net_name, net_pins in nets.items():
        # 分類網路中的節點
        logic_gates = []
        terminals = []
        
        for pin in net_pins:
            cell_name = pin['cell']
            is_terminal = nodes[cell_name].get('terminal', False)
            
            if is_terminal:
                terminals.append(cell_name)
            elif cell_name in cell_type_map:
                logic_gates.append(cell_name)
        
        # 只處理包含邏輯門的網路
        if not logic_gates:
            continue
            
        # print(f"\n📍 網路 {net_name}: Logic gates={logic_gates}, Terminals={terminals}")
        
        # 找出這個網路中的邏輯門驅動者（type='O'）和接收者（type='I'）
        logic_gate_drivers = []
        logic_gate_receivers = []
        
        for pin in net_pins:
            cell_name = pin['cell']
            pin_type = pin.get('type', '')
            
            # if cell_name in logic_gates:
            if pin_type == 'O':  # 邏輯門驅動這個網路
                logic_gate_drivers.append(cell_name)
            elif pin_type == 'I':  # 邏輯門接收這個網路的信號
                logic_gate_receivers.append(cell_name)
        
        # print(f"   邏輯門驅動者: {logic_gate_drivers}")
        # print(f"   邏輯門接收者: {logic_gate_receivers}")
        
        # 只為邏輯門驅動者計算輸出負載
        if not logic_gate_drivers:
            # print(f"   ⚠️  沒有邏輯門驅動者，跳過")
            continue
            
        # 檢查是否驅動輸出端口
        # output_terminals = []
        # for pin in net_pins:
        #     cell_name = pin['cell']
        #     pin_type = pin.get('type', '')
        #     is_terminal = nodes[cell_name].get('terminal', False)
            
        #     if is_terminal and pin_type == 'I':  # terminal接收信號 = 輸出端口
        #         output_terminals.append(cell_name)
        
        # if output_terminals:
        #     print(f"   ⚡ 驅動輸出端口 {output_terminals}，負載電容 = 0")
        #     continue
        
        # 計算邏輯門接收者的總輸入電容 - 只計算這個網路連接的特定引腳
        total_load_cap = 0.0
        for receiver in logic_gate_receivers:
            receiver_type = cell_type_map.get(receiver)
            if receiver_type and receiver_type in lib:
                cell_info = lib[receiver_type]
                pins_info = cell_info.get('pins', {})
                
                # 找出這個 receiver 在當前網路中連接的引腳
                receiver_pins_in_net = [p for p in net_pins if p['cell'] == receiver and p.get('type') == 'I']
                pin_cap_sum = 0.0
                for pin_info in receiver_pins_in_net:
                    pin_name = pin_info.get('pin_name', '')
                    if pin_name and pin_name in pins_info:
                        pin_cap = float(pins_info[pin_name].get('capacitance', 0))
                        pin_cap_sum += pin_cap
                        # print(f"       {receiver}.{pin_name}: {pin_cap:.5f}")
                    else:
                        # 如果沒有具體的 pin 名稱，使用平均輸入電容作為估計
                        input_pins = [p for p in pins_info.values() if p.get('direction') == 'input']
                        if input_pins:
                            avg_input_cap = sum(float(p.get('capacitance', 0)) for p in input_pins) / len(input_pins)
                            pin_cap_sum += avg_input_cap
                            # print(f"       {receiver}.<avg_input>: {avg_input_cap:.5f}")
                
                total_load_cap += pin_cap_sum
                # print(f"     {receiver} ({receiver_type}): {pin_cap_sum:.5f}")
        
        # print(f"   總負載電容: {total_load_cap:.5f}")
        
        # 分配給所有驅動者
        for driver in logic_gate_drivers:
            node_output_load_cap[driver] += total_load_cap
            # print(f"   分配給 {driver}: +{total_load_cap:.5f} (累計: {node_output_load_cap[driver]:.5f})")
    
    # print("\n📊 最終輸出負載電容:")
    # for name in sorted(node_output_load_cap.keys()):
    #     if name in cell_type_map:  # 只顯示邏輯門
    #         print(f"   {name}: {node_output_load_cap[name]:.5f}")

        # 4. 準備 enhanced node features
    node_names = list(nodes.keys())
    node_features = []
    node_name_to_idx = {}
    
    # 使用全域cell類型列表 (如果提供)
    if global_cell_types is None:
        # 收集當前設計的cell類型
        all_cell_types = set()
        for name in node_names:
            cell_type = cell_type_map.get(name)
            if cell_type is None:
                if nodes[name].get("terminal", False):
                    cell_type = "terminal_NI"
                else:
                    cell_type = "unknown"
            all_cell_types.add(cell_type)
        all_cell_types = sorted(all_cell_types)
        # print(f"Found {len(all_cell_types)} unique cell types: {all_cell_types}")
    else:
        all_cell_types = global_cell_types
    
    cell_type_to_idx = {ct: i for i, ct in enumerate(all_cell_types)}

    for i, name in enumerate(node_names):
        node = nodes[name]
        cell_type = cell_type_map.get(name)
        
        if cell_type is None:
            if node.get("terminal", False):
                cell_type = "terminal_NI"
            else:
                print(f"⚠️  Cannot find cell type for {name}, assign unknown")
                cell_type = "unknown"

        # === 基本物理特徵 ===
        if cell_type == "terminal_NI" or cell_type == "unknown":
            area = 0.0
            leakage = 0.0
            # 輸入電容統計特徵 (5維)
            input_cap_sum = 0.0
            input_cap_max = 0.0
            input_cap_min = 0.0
            input_cap_avg = 0.0
            input_cap_std = 0.0
            num_pins = 0.0
            drive_strength = 0.0
        else:
            cell_info = lib.get(cell_type, {})
            area = float(cell_info.get('attributes', {}).get('area', 0))
            
            # 改進的leakage計算 - 使用總 leakage power
            leakage_info = cell_info.get('attributes', {}).get('leakage_power', {}).get('VDD', {}).get('when', {})
            if isinstance(leakage_info, dict) and leakage_info:
                # 計算所有條件下的總 leakage power
                leakage = sum(float(v.get('value', 0)) for v in leakage_info.values())
            else:
                leakage = 0.0
            
            # 分別計算輸入電容統計特徵
            pins_info = cell_info.get('pins', {})
            
            # 收集輸入引腳的電容值
            input_caps = []
            num_pins = len(pins_info)
            
            for pin_name, pin_info in pins_info.items():
                cap = float(pin_info.get('capacitance', 0))
                direction = pin_info.get('direction', '')
                if direction == 'input':
                    input_caps.append(cap)
            
            # 計算輸入電容的統計特徵
            if input_caps:
                import numpy as np
                input_cap_sum = sum(input_caps)
                input_cap_max = max(input_caps)
                input_cap_min = min(input_caps)
                input_cap_avg = input_cap_sum / len(input_caps)
                input_cap_std = float(np.std(input_caps))
            else:
                # 沒有輸入引腳的情況
                input_cap_sum = 0.0
                input_cap_max = 0.0
                input_cap_min = 0.0
                input_cap_avg = 0.0
                input_cap_std = 0.0
            
            # 估計驅動強度 (基於cell名稱中的數字)
            import re
            strength_match = re.search(r'x(\d+)', cell_type)
            drive_strength = float(strength_match.group(1)) if strength_match else 1.0

        # === 位置特徵 ===
        is_terminal = float(node.get('terminal', False))
        x = pl.get(name, {}).get('x', 0)
        y = pl.get(name, {}).get('y', 0)
        
        # 歸一化位置 (相對於設計中心)
        all_x = [pl.get(n, {}).get('x', 0) for n in node_names]
        all_y = [pl.get(n, {}).get('y', 0) for n in node_names]
        center_x = sum(all_x) / len(all_x)
        center_y = sum(all_y) / len(all_y)
        
        rel_x = x - center_x
        rel_y = y - center_y
        distance_from_center = (rel_x**2 + rel_y**2)**0.5

        # === 拓撲特徵 ===
        degree = float(node_degree.get(name, 0))
        fanin = float(node_fanin.get(name, 0))
        fanout = float(node_fanout.get(name, 0))
        
        # === Cell類型ID（用於embedding） ===
        cell_id = float(cell_to_id.get(cell_type, cell_to_id.get("unknown", 1)))

        # === 功能類型特徵 ===
        # 基本邏輯門
        basic_gates = ['AND', 'OR', 'NOT', 'NAND', 'NOR', 'XOR', 'XNOR', 'INV', 'BUF']
        # 複合邏輯門 (AND-OR, OR-AND 組合)
        complex_gates = ['AO', 'OA', 'AOI', 'OAI', 'MUX', 'ADDER', 'DECODER', 'ENCODER']
        # 記憶體元件
        memory_gates = ['FF', 'DFF', 'LATCH', 'SRAM', 'REG']
        
        is_logic_gate = 1.0 if any(gate in cell_type.upper() for gate in basic_gates + complex_gates) else 0.0
        is_memory = 1.0 if any(mem in cell_type.upper() for mem in memory_gates) else 0.0
        is_complex = 1.0 if any(comp in cell_type.upper() for comp in complex_gates) else 0.0

        # === 輸出負載電容 ===
        output_load_cap = node_output_load_cap.get(name, 0.0)

        # 組合所有特徵 (22 基礎特徵 + 1 cell ID = 23，embedding 將在 GNN 中處理)
        feature_vector = [
            # 物理特徵 (9): area, leakage, 輸入電容統計(5), output_load_cap, num_pins, drive_strength
            area, leakage, input_cap_sum, input_cap_max, input_cap_min, input_cap_avg, input_cap_std, output_load_cap, num_pins, drive_strength,
            # 位置特徵 (6) 
            is_terminal, x, y, rel_x, rel_y, distance_from_center,
            # 拓撲特徵 (3)
            degree, fanin, fanout,
            # 功能類型特徵 (3)
            is_logic_gate, is_memory, is_complex,
            # Cell ID (1) - 將在 GNN 中轉換為 32 維 embedding
            cell_id
        ]

        node_features.append(feature_vector)
        node_name_to_idx[name] = i
        
        # 詳細顯示特徵向量組成 (只顯示前3個節點作為範例)
        # if i < 3:
        #     print(f"\n📊 節點 '{name}' (索引 {i}) 特徵詳情:")
        #     print(f"   Cell Type: {cell_type}")
        #     print(f"   物理特徵 (9): area={area:.5f}, leakage={leakage:.5f}")
        #     print(f"      輸入電容統計: sum={input_cap_sum:.5f}, max={input_cap_max:.5f}, min={input_cap_min:.5f}, avg={input_cap_avg:.5f}, std={input_cap_std:.5f}")
        #     print(f"      其他: output_load_cap={output_load_cap:.5f}, pins={num_pins}, drive={drive_strength}")
        #     print(f"   位置特徵 (6): terminal={is_terminal}, x={x}, y={y}, rel_x={rel_x:.1f}, rel_y={rel_y:.1f}, dist={distance_from_center:.1f}")
        #     print(f"   拓撲特徵 (3): degree={degree}, fanin={fanin}, fanout={fanout}")
        #     print(f"   功能特徵 (3): logic={is_logic_gate}, memory={is_memory}, complex={is_complex}")
        #     print(f"   Cell ID (1): {cell_id} (對應 cell_type: {cell_type})")
        #     print(f"   完整向量 ({len(feature_vector)}): {[f'{x:.5f}' for x in feature_vector[:10]]}{'...' if len(feature_vector) > 10 else ''}")
        # elif i == 3:
        #     print(f"   ... (省略其餘 {len(node_names)-3} 個節點的詳細輸出)")

    x = torch.tensor(node_features, dtype=torch.float)
    
    # print(f"Enhanced node features shape: {x.shape}")
    # print(f"Feature breakdown: Physical(9) + Position(6) + Topology(3) + Function(3) + CellID(1) = {x.shape[1]} total (23 features)")
    # print(f"Note: Cell ID will be converted to 32-dim embedding in GNN, resulting in 22+32=54 total dimensions")

    # 5. 建立 edge_index with edge features
    # 5. 建立 edge_index with edge features
    edge_list = []
    edge_attr_list = []
    
    for net_name, net_pins in nets.items():
        net_size = len(net_pins)
        pins = [n['cell'] for n in net_pins if n['cell'] in node_name_to_idx]
        
        # 計算net的統計資訊 (改用degree作為簡化)
        net_importance = 1.0 / net_size if net_size > 0 else 0.0
        
        for i in range(len(pins)):
            for j in range(i + 1, len(pins)):
                src = node_name_to_idx[pins[i]]
                dst = node_name_to_idx[pins[j]]
                
                # 計算edge特徵 - 位置座標欄位索引：
                # feature layout:
                # 0 area, 1 leakage, 2-6 input cap stats, 7 output_load_cap, 8 num_pins, 9 drive_strength,
                # 10 is_terminal, 11 x, 12 y, 13 rel_x, 14 rel_y, 15 dist_center, ...
                src_x = node_features[src][11]
                src_y = node_features[src][12]
                dst_x = node_features[dst][11]
                dst_y = node_features[dst][12]
                
                distance = ((src_x - dst_x)**2 + (src_y - dst_y)**2)**0.5
                
                edge_features = [
                    float(net_size),        # 網路大小
                    distance,               # 物理距離
                    net_importance,         # 網路重要性
                    1.0                     # 連接強度
                ]
                
                edge_list.append([src, dst])
                edge_list.append([dst, src])  # undirected
                edge_attr_list.extend([edge_features, edge_features])  # 無向邊，兩個方向使用相同特徵

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float) if edge_attr_list else None

    # 6. 包成 PyG 的 Data 結構
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.node_names = node_names
    
    # print(f"Graph built: {len(node_names)} nodes, {edge_index.shape[1]} edges")
    # if edge_attr is not None:
    #     print(f"Edge features shape: {edge_attr.shape}")

    return data
