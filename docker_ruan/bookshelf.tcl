# ==== bookshelf_gen.tcl ====
# 功能：將已完成 placement 的 DEF 透過 RosettaStone 轉出 Bookshelf
# 執行：openroad -exit bookshelf_gen.tcl

############## 使用者需填寫的變數 ################
# 多個 benchmark 以空格分隔，例如：{s27 s298}
set benchmarks {s27}

# <bench>.def 的根目錄：<design_root>/<bench>/<bench>.def
set design_root "../gtlvl_design"

# PDK 目錄（ASAP7 為例）
set lib_dir      "../ic_cad/ICCAD25/ASAP7"

# RosettaStone 路徑
set rosettastone "/workspace/docker_ruan/RosettaStone"

# tech LEF
set tech_lef "$lib_dir/techlef/asap7_tech_1x_201209.lef"

# cell / macro LEF（每行需雙引號，才能展開 $lib_dir 變數）
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

# PDK 代號，決定輸出檔名前綴
set platform_tag "asap7sc7p5t"

# <<< 你可能想改成自己的 venv >>> ----------------
# 預設使用 openroad 內建的 Python（含 odb）
set or_bin       [exec which openroad]
set python_exec  [list $or_bin -python]
# 若改用 venv，請註解上面兩行並改成：
# set python_exec "/workspace/docker_ruan/venv/bin/python"
###################################################

# ========== 逐設計轉檔 ==========
foreach bench $benchmarks {
    puts "\n===== 轉換設計：$bench ====="

    # 1) 路徑組合
    set design_def "$design_root/$bench/$bench.def"
    set out_dir    "$design_root/$bench/bookshelf_run"
    file mkdir $out_dir

    # 2) DEF → ODB
    set cmd [concat $python_exec [list \
        $rosettastone/odbComm/make_odb.py \
        --platform $platform_tag \
        --def $design_def \
        --lef $tech_lef]]
    foreach lef $cell_lefs { lappend cmd --lef $lef }
    lappend cmd --out_dir $out_dir

    puts ">> 執行: [join $cmd { }]"
    if {[catch {eval exec $cmd} err]} {
        puts stderr "ERROR(make_odb): $err"
        continue
    }
    puts "✅ $bench ODB 產生完成！"

    # 3) 取得 .odb
    set odb_file [lindex [glob -nocomplain $out_dir/*.odb] 0]
    if {$odb_file eq ""} {
        puts stderr "ERROR: 找不到 .odb，跳過 $bench"
        continue
    }

    # 4) ODB → Bookshelf
    set cmd2 [concat $python_exec [list \
        $rosettastone/odbComm/convert_odb2bookshelf.py \
        --odb $odb_file \
        --out_dir $out_dir/bookshelf \
        --mode ISPD04]]
    puts ">> 執行: [join $cmd2 { }]"
    if {[catch {eval exec $cmd2} err2]} {
        puts stderr "ERROR(convert): $err2"
        continue
    }

    puts "✅ $bench 轉換完成！結果在 $out_dir/bookshelf"
}

puts "\n🎉 全部設計處理完畢！"
exit
