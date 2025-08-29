
# 完整的 OpenROAD PPA 分析腳本
puts "=== 開始 OpenROAD PPA 分析 ==="

# 讀取 Liberty 檔案
puts "讀取 Liberty 檔案..."
if {[file exists "../ASAP7/LIB"]} {
    foreach libFile [glob "../ASAP7/LIB/*nldm*.lib"] {
        puts "載入 LIB: $libFile"
        read_liberty $libFile
    }
} else {
    puts "警告: 找不到 Liberty 檔案目錄 ../ASAP7/LIB"
}

# 讀取 LEF 檔案  
puts "讀取 LEF 檔案..."
if {[file exists "../ASAP7/techlef/asap7_tech_1x_201209.lef"]} {
    read_lef ../ASAP7/techlef/asap7_tech_1x_201209.lef
} else {
    puts "警告: 找不到技術 LEF 檔案"
}

if {[file exists "../ASAP7/LEF"]} {
    foreach lefFile [glob "../ASAP7/LEF/*.lef"] {
        puts "載入 LEF: $lefFile"  
        read_lef $lefFile
    }
} else {
    puts "警告: 找不到 LEF 檔案目錄 ../ASAP7/LEF"
}

# 讀取設計檔案
puts "讀取設計檔案..."
if {[file exists "s27.def"]} {
    read_def s27.def
    puts "已載入 DEF: s27.def"
} else {
    puts "錯誤: 找不到 DEF 檔案 s27.def"
    exit 1
}

if {[file exists "s27.sdc"]} {
    read_sdc s27.sdc  
    puts "已載入 SDC: s27.sdc"
} else {
    puts "警告: 找不到 SDC 檔案 s27.sdc"
}

# 載入 RC 參數
puts "載入 RC 寄生參數..."
if {[file exists "../ASAP7/setRC.tcl"]} {
    source ../ASAP7/setRC.tcl
} else {
    puts "警告: 找不到 RC 檔案 ../ASAP7/setRC.tcl"
}

# 估算寄生參數
puts "估算寄生參數..."
estimate_parasitics -placement

# 執行指定的分析命令
puts "執行分析命令..."

set tns_file "temp_tns.txt"
report_tns > $tns_file
puts "TNS 報告已寫入 $tns_file"


puts "=== OpenROAD 分析完成 ==="
