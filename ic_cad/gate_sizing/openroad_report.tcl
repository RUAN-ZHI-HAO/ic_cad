# set seed_def $::env(SEED_DEF)
# set sol_def $::env(SOL_DEF)
set def_file $::env(DEF)
set prefix $::env(PREFIX)
set sdc_file $::env(SDC)
set LIB_DIR "/root/solution/testcases/ASAP7"

# 初始化技術文件的函數
proc load_technology {LIB_DIR} {
    puts "<<<<<<<<<< 初始化技術文件 >>>>>>>>>>"

    puts "-----讀取 LIB 檔案-----"
    foreach libFile [glob "$LIB_DIR/LIB/*nldm*.lib"] {
        read_liberty $libFile
    }

    puts "-----讀取 LEF 檔案-----"
    read_lef $LIB_DIR/techlef/asap7_tech_1x_201209.lef
    foreach lefFile [glob "$LIB_DIR/LEF/*.lef"] {
        read_lef $lefFile
    }
}

# 初始化技術文件的函數
proc report_all {prefix} {    
    puts "\n<<<<<<<<<< 回報 $prefix 結果 >>>>>>>>>>"    
      
    # 安全地估算寄生參數    
    if {[catch {estimate_parasitics -placement} err]} {    
        puts "警告：寄生參數估算失敗：$err"    
    }    
    # 安全地檢查布局   
    if {[catch {check_placement} err]} {    
        puts "警告：布局檢查失敗：$err"  
        return 1  
    }    
      
    # 直接獲取數值，添加調試信息  
    puts "獲取時序數據..."  
    set tns_value [sta::total_negative_slack -max]    
    set wns_value [sta::worst_slack -max]    
    puts "TNS: $tns_value, WNS: $wns_value"  
        
    # 獲取利用率和面積，添加調試  
    puts "獲取面積數據..."  
    set raw_util [rsz::utilization]  
    set raw_area [rsz::design_area]  
    puts "Raw utilization: $raw_util, Raw area: $raw_area"  
      
    set util_value [format %.1f [expr $raw_util * 100]]    
    set area_value [sta::format_area $raw_area 0]    
        
    # 獲取功耗，需要指定 corner  
    set power_value "N/A"  
    if {[catch {  
        set corner [sta::cmd_corner]  
        set power_result [sta::design_power $corner]  
        set power_value [lindex $power_result 3]  
    } err]} {  
        puts "功耗分析失敗：$err"  
    }
        
    puts "${prefix} 分析完成!!"    
    puts "${prefix}_TNS: $tns_value"    
    puts "${prefix}_WNS: $wns_value"    
    puts "${prefix}_Total_Power: $power_value"    
    puts "${prefix}_AREA: $area_value"    
    puts "${prefix}_UTIL: ${util_value}%"    
}

# 處理函數：分析兩個設計
proc analyze_design {LIB_DIR sdc_file def_file prefix} {
    # 重新載入技術檔案  
    load_technology $LIB_DIR  

    puts "\n<<<<<<<<<< 分析 $prefix 設計 >>>>>>>>>>"

    puts "-----讀取 DEF 檔案: $def_file-----"
    read_def $def_file

    puts "-----讀取 SDC 檔案: $sdc_file-----"
    read_sdc $sdc_file

    puts "載入RC寄生參數模型..."
    source $LIB_DIR/setRC.tcl

    report_all $prefix
}

# 分析 seed & sol 設計
analyze_design $LIB_DIR $sdc_file $def_file $prefix