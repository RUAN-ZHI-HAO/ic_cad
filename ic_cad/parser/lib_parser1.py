import os
import json
import re

def extract_pins_structured(cell_body):
    pins = {}
    for match in re.finditer(r'pin\s*\(([^)]+)\)\s*\{', cell_body):
        pin_name = match.group(1).strip()
        if pin_name in ("VDD", "VSS"):
            continue

        start = match.end()
        brace = 1
        end = start
        while end < len(cell_body) and brace > 0:
            if cell_body[end] == '{':
                brace += 1
            elif cell_body[end] == '}':
                brace -= 1
            end += 1
        pin_body = cell_body[start:end - 1]
        pin_data = {}

        # --- 1. 找出所有 timing/internal_power block 區間
        blocks = []
        for block_type in ["timing", "internal_power"]:
            for t_match in re.finditer(r'%s\s*\(\)\s*\{' % block_type, pin_body):
                t_start = t_match.end()
                brace = 1
                t_end = t_start
                while t_end < len(pin_body) and brace > 0:
                    if pin_body[t_end] == '{':
                        brace += 1
                    elif pin_body[t_end] == '}':
                        brace -= 1
                    t_end += 1
                blocks.append((t_match.start(), t_end))
        # sort by start
        blocks.sort()

        # --- 2. 只針對非 block 區間 parse key:value 和 key:(X, Y)
        cursor = 0
        for block_start, block_end in blocks + [(len(pin_body), len(pin_body))]:
            outside = pin_body[cursor:block_start]
            # key: value
            for k, v in re.findall(r'([a-zA-Z0-9_]+)\s*:\s*([^;{}\n]+);', outside):
                pin_data[k.strip()] = v.strip().strip('"')
            # key: (X, Y) 只允許兩個數值的 range
            for k, v in re.findall(r'([a-zA-Z0-9_]+)\s*\(\s*([^)]+)\s*\)\s*;', outside):
                items = [float(x.strip()) for x in v.split(',') if x.strip()]
                pin_data[k.strip()] = items
            cursor = block_end

        # --- 3. timing block (支援多個 timing 區塊)
        timing_blocks = []
        for t_match in re.finditer(r'timing\s*\(\)\s*\{', pin_body):
            t_start = t_match.end()
            brace = 1
            t_end = t_start
            while t_end < len(pin_body) and brace > 0:
                if pin_body[t_end] == '{':
                    brace += 1
                elif pin_body[t_end] == '}':
                    brace -= 1
                t_end += 1
            timing_block = pin_body[t_start:t_end - 1]
            timing_data = {}
            # timing key:value
            # print(timing_block)
            # print(re.search(r'when\s*:\s*"([^"]+)";', timing_block))
            for k, v in re.findall(r'([a-zA-Z0-9_]+)\s*:\s*([^;{}\n]+);', timing_block):
                timing_data[k.strip()] = v.strip().strip('"')
            # timing 2D table
            for submatch in re.finditer(r'(cell_rise|cell_fall|rise_transition|fall_transition)\s*\(([^)]+)\)\s*\{(.*?)\}', timing_block, re.DOTALL):
                subname, tmpl, subblock = submatch.group(1), submatch.group(2), submatch.group(3)
                idx1 = re.search(r'index_1\s*\(\s*"([^"]+)"\s*\)', subblock)
                idx1_list = [float(x.strip()) for x in idx1.group(1).split(',')] if idx1 else []
                idx2 = re.search(r'index_2\s*\(\s*"([^"]+)"\s*\)', subblock)
                idx2_list = [float(x.strip()) for x in idx2.group(1).split(',')] if idx2 else []
                values_match = re.search(r'values\s*\(\s*(.*?)\s*\);', subblock, re.DOTALL)
                if values_match:
                    raw_values = values_match.group(1)
                    lines = re.findall(r'"([^"]+)"', raw_values)
                    values_2d = []
                    for line in lines:
                        values_2d.append([float(x.strip()) for x in line.split(',') if x.strip()])
                else:
                    values_2d = []
                timing_data[subname] = {
                    tmpl: {
                        "index_1": idx1_list,
                        "index_2": idx2_list,
                        "values": values_2d
                    }
                }
            
            # 使用 when 條件作為 key，如果沒有 when 則用索引
            when_condition = timing_data.get('when', f'timing_{len(timing_blocks)}')
            timing_blocks.append({when_condition: timing_data})
        
        # 將所有 timing 區塊合併到一個字典中
        if timing_blocks:
            pin_data['timing'] = {}
            for timing_block in timing_blocks:
                pin_data['timing'].update(timing_block)

        # --- 4. internal_power (多個)
        internal_power_dict = {}
        for p_match in re.finditer(r'internal_power\s*\(\)\s*\{', pin_body):
            p_start = p_match.end()
            brace = 1
            p_end = p_start
            while p_end < len(pin_body) and brace > 0:
                if pin_body[p_end] == '{':
                    brace += 1
                elif pin_body[p_end] == '}':
                    brace -= 1
                p_end += 1
            ip_block = pin_body[p_start:p_end - 1]
            ip_data = {}
            for k, v in re.findall(r'([a-zA-Z0-9_]+)\s*:\s*([^;{}\n]+);', ip_block):
                ip_data[k.strip()] = v.strip().strip('"')
            # 只允許 key: (X, Y) 的 range，不抓 index_1/values
            for k, v in re.findall(r'([a-zA-Z0-9_]+)\s*\(\s*([^)]+)\s*\)\s*;', ip_block):
                if k.strip().endswith("_range"):  # 只抓 range
                    items = [float(x.strip()) for x in v.split(',') if x.strip()]
                    ip_data[k.strip()] = items
            for submatch in re.finditer(r'(rise_power|fall_power)\s*\(([^)]+)\)\s*\{(.*?)\}', ip_block, re.DOTALL):
                subname, tmpl, subblock = submatch.group(1), submatch.group(2), submatch.group(3)
                idx1 = re.search(r'index_1\s*\(\s*"([^"]+)"\s*\)', subblock)
                idx1_list = [float(x.strip()) for x in idx1.group(1).split(',')] if idx1 else []
                idx2 = re.search(r'index_2\s*\(\s*"([^"]+)"\s*\)', subblock)
                idx2_list = [float(x.strip()) for x in idx2.group(1).split(',')] if idx2 else []
                values_match = re.search(r'values\s*\(\s*(.*?)\s*\);', subblock, re.DOTALL)
                if values_match:
                    raw_values = values_match.group(1)
                    lines = re.findall(r'"([^"]+)"', raw_values)
                    values_2d = []
                    for line in lines:
                        values_2d.append([float(x.strip()) for x in line.split(',') if x.strip()])
                else:
                    values_2d = []
                ip_data[subname] = {
                    tmpl: {
                        "index_1": idx1_list,
                        "index_2": idx2_list,
                        "values": values_2d
                    }
                }
            pg_pin = ip_data.get('related_pg_pin', f'unnamed_{len(internal_power_dict)}')
            internal_power_dict[pg_pin] = ip_data
        pin_data['internal_power'] = internal_power_dict

        pins[pin_name] = pin_data
    return pins


def cell_extract_attributes(cell_body: str) -> dict:
    result = {}

    # 1. 擷取 pin 區塊之前的內容，只保留最前面的單行 key:value
    first_block_match = re.search(r'\b(pg_pin|pin|leakage_power)\s*\(', cell_body)
    if first_block_match:
        first_block_start = first_block_match.start()
        trimmed = cell_body[:first_block_start]
    else:
        trimmed = cell_body

    simple_attrs = re.findall(r'([a-zA-Z0-9_]+)\s*:\s*([^;{}]+);', trimmed)
    for k, v in simple_attrs:
        result[k.strip()] = v.strip().strip('"')

    # 2. pg_pin 區塊
    result["pg_pin"] = {}
    for match in re.finditer(r'pg_pin\s*\(([^)]+)\)\s*\{(.*?)\}', cell_body, re.DOTALL):
        name = match.group(1).strip()
        body = match.group(2)
        attrs = dict(re.findall(r'([a-zA-Z0-9_]+)\s*:\s*([^;{}]+);', body))
        result["pg_pin"][name] = {
            k.strip(): v.strip().strip('"') for k, v in attrs.items() if k.strip() in ("direction", "pg_type")
        }

    # 3. leakage_power 區塊
    result["leakage_power"] = {}
    for match in re.finditer(r'leakage_power\s*\(\)\s*\{(.*?)\}', cell_body, re.DOTALL):
        body = match.group(1)
        value_match = re.search(r'value\s*:\s*([^;{}]+);', body)
        when_match = re.search(r'when\s*:\s*"([^"]+)";', body)
        pg_pin_match = re.search(r'related_pg_pin\s*:\s*([a-zA-Z0-9_]+);', body)

        if value_match and pg_pin_match:
            pin = pg_pin_match.group(1).strip()
            value = value_match.group(1).strip()

            if pin not in result["leakage_power"]:
                result["leakage_power"][pin] = {"when": {}}

            if when_match:
                when_cond = when_match.group(1).strip()
            else:
                when_cond = "default"

            result["leakage_power"][pin]["when"][when_cond] = {"value": value}

    return result



def extract_blocks(text, keyword):
    blocks = []
    pattern = rf'{keyword}\s*\(([^)]+)\)\s*\{{'
    pos = 0
    while True:
        match = re.search(pattern, text[pos:])
        # print(match)
        if not match:
            break
        name = match.group(1).strip()
        start = pos + match.end()
        brace_level = 1
        end = start
        while end < len(text) and brace_level > 0:
            if text[end] == '{':
                brace_level += 1
            elif text[end] == '}':
                brace_level -= 1
            end += 1
        block = text[start:end-1]
        blocks.append((name, block))
        pos = end
    return blocks


# 你自己的 .lib parser function，這邊示意
def parse_lib_file(filepath):
    result = {}

    with open(filepath, 'r') as f:
        content = f.read()

    # 拿到所有 cell 區塊
    cell_blocks = extract_blocks(content, "cell")

    for cell_name, cell_body in cell_blocks:
        attributes = cell_extract_attributes(cell_body)
        cell_data = {
            "attributes": attributes,
            "pins": extract_pins_structured(cell_body)
        }

        result[cell_name] = cell_data

    return result

# 指定資料夾路徑
lib_folder = "../ICCAD25/ASAP7/LIB"

# 用來收集所有 cell 的 dict
all_cells = {}

# 一個一個讀取 .lib 檔案
for filename in os.listdir(lib_folder):
    # print(filename)
    # if filename != 'asap7sc7p5t_SIMPLE_RVT_TT_nldm_211120.lib':
    #     continue
    if filename == "merged_nosram.lib":
        continue
    if filename.endswith(".lib"):
        filepath = os.path.join(lib_folder, filename)
        # print(filepath)
        print(f"Parsing {filename}...")
        cells = parse_lib_file(filepath)

        # 加入所有 cell（不記檔名，直接塞進 all_cells）
        for cell_name, cell_info in cells.items():
            if cell_name in all_cells:
                print(f"⚠️ Warning: cell {cell_name} already exists! 被覆蓋")
            all_cells[cell_name] = cell_info

# 全部 cell 寫入一個 JSON 檔案
with open("parsed_cells.json", "w") as f:
    json.dump(all_cells, f, indent=2)

print("✅ 完成！所有 cell 已寫入 parsed_cells.json")
