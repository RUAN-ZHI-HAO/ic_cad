# ==== bookshelf_gen.tcl ====
# 用 RosettaStone 將 DEF 轉 ODB，再轉 Bookshelf
# 執行：openroad -exit bookshelf_gen.tcl

######################## 基本設定 ########################
# 多個 benchmark 以空格分隔：{s27 s298 …}
# set benchmarks {c17 c432 c499 c880 c1355 c1908 c2670 c3540 c5315 c6288 c7552 s27 s298 s344 s382 s386 s400 s420 s444 s510 s526 s641 s713 s820 s832 s838 s953 s1196 s1238 s1423 s1488 s1494 s5378 s9234 s13207 s15850 s35932 s38417 s38584}
set benchmarks {c7752}
# 根目錄（請改成你實際位置，再簡單做一次 file normalize）
set root_dir [file normalize "."]          ;# 目前目錄
set design_root [file normalize "../gtlvl_design"]
set lib_dir     [file normalize "../ic_cad/ICCAD25/ASAP7"]
set rosettastone [file normalize "/workspace/docker_ruan/RosettaStone"]

# tech & cell LEF
set tech_lef "$lib_dir/techlef/asap7_tech_1x_201209.lef"
set cell_lefs [list \
    "$lib_dir/LEF/asap7sc7p5t_28_L_1x_220121a.lef" \
    "$lib_dir/LEF/asap7sc7p5t_28_R_1x_220121a.lef" \
    "$lib_dir/LEF/asap7sc7p5t_28_SL_1x_220121a.lef" \
    "$lib_dir/LEF/asap7sc7p5t_28_SRAM_1x_220121a.lef" \
    "$lib_dir/LEF/sram_asap7_16x256_1rw.lef" \
    "$lib_dir/LEF/sram_asap7_32x256_1rw.lef" \
    "$lib_dir/LEF/sram_asap7_64x64_1rw.lef" \
    "$lib_dir/LEF/sram_asap7_64x256_1rw.lef" \
]

# PDK 代號
set platform_tag "asap7sc7p5t"

# openroad 內建 Python；若要用 venv 換掉絕對路徑即可
set python_exec [list [exec which openroad] -exit -python]
###########################################################

# ---------- 小工具 ----------
proc write_file {path data} {
    set fh [open $path w]; puts $fh $data; close $fh
}

# ========= 逐設計轉檔 =========
foreach bench $benchmarks {
    puts "\n===== 轉換設計：$bench ====="

    set design_def [file normalize "$design_root/$bench/${bench}_placed.def"]
    set out_dir    [file normalize "$design_root/$bench/bookshelf_run"]
    
    # 檢查 DEF 文件是否存在
    if {![file exists $design_def]} {
        puts stderr "ERROR: DEF 文件不存在: $design_def"
        continue
    }
    
    file mkdir $out_dir

    # ---------- 產生 temp_make_odb.py ----------
    set make_py "$out_dir/temp_make_odb.py"

    # tech_lef + cell_lef → Python list 文字
    set py_lef_list "    r\"$tech_lef\",\n"
    foreach lef $cell_lefs { append py_lef_list "    r\"$lef\",\n" }

    set make_code [format {import odb, os
platform = "%s"
design   = "%s"
def_file = r"%s"
out_dir  = r"%s"

lef_list = [
%s]

db = odb.dbDatabase.create()
for lf in lef_list:
    odb.read_lef(db, lf)

# 獲取 tech 並讀取 DEF 文件
tech = db.getTech()
odb.read_def(tech, def_file)

os.makedirs(out_dir, exist_ok=True)
odb_path = f"{out_dir}/{platform}_{design}.odb"
odb.write_db(db, odb_path)
print(f"ODB written to: {odb_path}")
} $platform_tag $bench $design_def $out_dir $py_lef_list]
    write_file $make_py $make_code

    # ---------- 執行 DEF → ODB ----------
    # 為每個設計創建臨時 TCL 腳本
    set temp_tcl "$out_dir/temp_make_odb.tcl"
    set tcl_code [format {
# 讀取 LEF 文件
read_lef {%s}
foreach lef {%s} {
    read_lef $lef
}

# 讀取 DEF 文件
read_def {%s}

# 寫入 ODB 文件
write_db {%s}
exit
} $tech_lef $cell_lefs $design_def $out_dir/${platform_tag}_${bench}.odb]
    
    write_file $temp_tcl $tcl_code
    
    # 執行 OpenROAD 腳本
    set cmd [list openroad -exit $temp_tcl]
    if {[catch {eval exec $cmd} err]} {
        puts stderr "ERROR(make_odb): $err"
        file delete -force $temp_tcl
        continue
    }
    
    puts "✅ $bench ODB 產生完成！"
    file delete -force $temp_tcl

    # 取得 .odb
    set odb_file [lindex [glob -nocomplain $out_dir/*.odb] 0]
    if {$odb_file eq ""} {
        puts stderr "ERROR: 找不到 .odb，跳過 $bench"
        continue
    }

    # ---------- 產生 temp_convert.py ----------
    set conv_py "$out_dir/temp_convert.py"
    set conv_code [format {import odb, os, sys, re
sys.path.append(r"%s/odbComm")
from convert_odb2bookshelf import OdbToBookshelf

odb_file = r"%s"
out_dir  = r"%s"
os.makedirs(out_dir, exist_ok=True)

# 切換到輸出目錄
os.chdir(out_dir)

# 創建必要的目錄結構
os.makedirs("output/%s", exist_ok=True)

db = odb.dbDatabase.create(); odb.read_db(db, odb_file)

# 使用設計名稱作為 bookshelf 名稱
design_name = "%s"
converter = OdbToBookshelf(opendbpy=odb, opendb=db,
                          cellPadding=0, modeFormat="ISPD05",
                          layerCapacity=r"%s/odbComm/layeradjust_empty.tcl")

# 手動調用轉換函數，跳過 namemap 步驟
converter.WriteAux(design_name)
converter.WriteScl(design_name)
converter.WriteNodes(design_name)
converter.WriteRoute(design_name)
converter.WriteWts(design_name)
converter.WriteNets(design_name)
converter.WritePl(design_name)
converter.WriteShapes(design_name)

# 後處理檔案以匹配大會格式
def fix_nodes_format(filename):
    """修正 .nodes 檔案格式"""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    with open(filename, 'w') as f:
        for line in lines:
            # 移除註解行
            if line.startswith('#'):
                continue
            # 修正 NumNodes/NumTerminals 格式
            line = re.sub(r'NumNodes\s*:\s*(\d+)', r'NumNodes : \1', line)
            line = re.sub(r'NumTerminals\s*:\s*(\d+)', r'NumTerminals : \1', line)
            # 修正 terminal 標記
            if 'terminal' in line and not 'terminal_NI' in line:
                line = line.replace('terminal', 'terminal_NI')
            f.write(line)

def fix_scl_format(filename):
    """修正 .scl 檔案格式"""
    with open(filename, 'r') as f:
        content = f.read()
    
    # 移除註解行
    lines = content.split('\n')
    filtered_lines = []
    for line in lines:
        if not line.strip().startswith('#'):
            # 修正各種格式
            line = re.sub(r'NumRows\s*:\s*(\d+)', r'NumRows : \1', line)
            line = re.sub(r'Coordinate\s*:\s*(\d+)', r'Coordinate   : \1', line)
            line = re.sub(r'Height\s*:\s*(\d+)', r'Height       : \1', line)
            line = re.sub(r'Sitewidth\s*:\s*(\d+)', r'Sitewidth    : \1', line)
            line = re.sub(r'Sitespacing\s*:\s*(\d+)', r'Sitespacing  : \1', line)
            # 修正 Siteorient: R0 -> 0, R1 -> 1 等
            line = re.sub(r'Siteorient\s*:\s*R0', r'Siteorient   : 0', line)
            line = re.sub(r'Siteorient\s*:\s*R1', r'Siteorient   : 1', line)
            line = re.sub(r'Siteorient\s*:\s*(\w+)', r'Siteorient   : \1', line)
            # 修正 Sitesymmetry: True -> 1, False -> 0
            line = re.sub(r'Sitesymmetry\s*:\s*True', r'Sitesymmetry : 1', line)
            line = re.sub(r'Sitesymmetry\s*:\s*False', r'Sitesymmetry : 0', line)
            line = re.sub(r'SubrowOrigin\s*:\s*(\d+)\s*NumSites\s*:\s*(\d+)', r'SubrowOrigin : \1  NumSites : \2', line)
            filtered_lines.append(line)
    
    with open(filename, 'w') as f:
        f.write('\n'.join(filtered_lines))

def fix_pl_format(filename):
    """修正 .pl 檔案格式 - 確保 primary I/O 有 /FIXED_NI"""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    with open(filename, 'w') as f:
        for line in lines:
            # 移除註解行
            if line.startswith('#'):
                continue
            # 確保 primary I/O 端口有 /FIXED_NI 標記
            if ': N' in line and '/FIXED_NI' not in line:
                # 檢查是否為 primary I/O (通常是簡短名稱，不含底線開頭)
                parts = line.split()
                if len(parts) >= 4:
                    name = parts[0]
                    # Primary I/O 通常不以底線開頭，且名稱較簡短
                    if not name.startswith('_') and not 'reg' in name.lower() and len(name) <= 10:
                        line = line.rstrip() + ' /FIXED_NI\n'
            f.write(line)

def fix_nets_format(filename):
    """修正 .nets 檔案格式"""
    with open(filename, 'r') as f:
        content = f.read()
    
    # 移除註解行並修正格式
    lines = content.split('\n')
    filtered_lines = []
    for line in lines:
        if not line.strip().startswith('#'):
            # 修正 NetDegree 格式，移除多餘空格
            if line.startswith('NetDegree'):
                line = re.sub(r'NetDegree\s*:\s*(\d+)\s+(\w+)', r'NetDegree : \1 \2', line)
            
            # 修正 pin 座標格式
            elif re.match(r'^\s+\w+\s+[IO]\s*:', line):
                # 檢查是否已有完整格式
                if not re.search(r':\s*[\d\.-]+\s+[\d\.-]+\s*:\s*[\d\.]+\s+[\d\.]+\s+p_', line):
                    # 需要補充格式
                    parts = line.split(':')
                    if len(parts) >= 2:
                        # 提取 pin 名稱和方向
                        pin_match = re.match(r'^\s*(\w+)\s+([IO])', line)
                        if pin_match:
                            pin_name = pin_match.group(1)
                            direction = pin_match.group(2)
                            coords = parts[1].strip()
                            
                            # 修正時鐘相關的方向和類型
                            if pin_name in ['CK', 'clk', 'CLK']:
                                # Primary clock 應該是 Output，類型是 p_
                                direction = 'O'
                                pin_type = 'p_'
                            elif direction == 'I' and ('CLK' in line or 'clk' in line or '_reg' in line):
                                # 連接到時鐘的 register pins
                                pin_type = 'p_CLK'
                            elif '_reg' in pin_name and direction == 'O':
                                pin_type = 'p_QN'
                            elif direction == 'I' and ('_' in pin_name or 'u0_' in pin_name):
                                pin_type = 'p_B1'
                            elif direction == 'I':
                                pin_type = 'p_A'
                            else:
                                pin_type = 'p_'
                            
                            # 重建格式，使其與大會一致
                            spaces = '  ' if pin_name in ['CK', 'clk', 'CLK'] or len(pin_name) <= 3 else '  '
                            line = f"{spaces}{pin_name} {direction} : {coords} : 0.0 0.0 {pin_type}"
            filtered_lines.append(line)
    
    with open(filename, 'w') as f:
        f.write('\n'.join(filtered_lines))

def fix_pl_format(filename):
    """修正 .pl 檔案格式 - 確保 primary I/O 有 /FIXED_NI"""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    with open(filename, 'w') as f:
        for line in lines:
            # 移除註解行
            if line.startswith('#'):
                continue
            # 確保 primary I/O 端口有 /FIXED_NI 標記
            if ': N' in line and '/FIXED_NI' not in line:
                # 檢查是否為 primary I/O (通常是簡短名稱，不含底線或複雜結構)
                parts = line.split()
                if len(parts) >= 4:
                    name = parts[0]
                    if not name.startswith('_') and not 'reg' in name.lower():
                        line = line.rstrip() + ' /FIXED_NI\n'
            f.write(line)

# 修正各個檔案格式
output_path = f"output/{design_name}"
fix_nodes_format(f"{output_path}/{design_name}.nodes")
fix_nets_format(f"{output_path}/{design_name}.nets")
fix_pl_format(f"{output_path}/{design_name}.pl")
fix_scl_format(f"{output_path}/{design_name}.scl")

print("Bookshelf conversion completed successfully!")
print("Files formatted to match competition standard.")
} $rosettastone $odb_file $out_dir $bench $bench $rosettastone]
    write_file $conv_py $conv_code

    # ---------- 執行 ODB → Bookshelf ----------
    set cmd2 [concat $python_exec [list $conv_py]]
    if {[catch {eval exec $cmd2} err2]} {
        puts stderr "ERROR(convert): $err2"
        continue
    }

    puts "✅ $bench 轉換完成！結果在 $out_dir/output/$bench"
    
    # ---------- 清理臨時文件 ----------
    file delete -force $make_py
    file delete -force $conv_py
    file delete -force $odb_file
    puts "🧹 臨時文件已清理"
}

puts "\n🎉 全部設計處理完畢！"
exit
