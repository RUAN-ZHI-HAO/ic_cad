#!/bin/bash

# benchmarks=(c17 c432 c499 c880 c1355 c1908 c2670 c3540 c5315 c6288 c7552 s27 s298 s344 s382 s386 s400 s420 s444 s510 s526 s641 s713 s820 s832 s838 s953 s1196 s1238 s1423 s1488 s1494 s5378 s9234 s13207 s15850 s35932 s38417 s38584)
benchmarks=(c7752)
benchmarks_dir="/workspace/ic_cad/open_ic_design/ISCAS/Verilog"
output_dir="/workspace/gtlvl_design"
pdk_dir="/workspace/ic_cad/ICCAD25/ASAP7"
lib_dir="$pdk_dir/LIB"
lef_dir="$pdk_dir/LEF"
merged_lib="$lib_dir/merged_nosram.lib"

tech_lef="$pdk_dir/techlef/asap7_tech_1x_201209.lef"
# tech_tracks_lef="$pdk_dir/techlef/asap7_tracks.lef"

# 建立合併的 .lib 檔案（排除 SRAM）
if [ ! -f "$merged_lib" ]; then
    echo "⏳ Merging non-SRAM .lib files into $merged_lib ..."
    echo "library(merged_nosram) {" > "$merged_lib"

    find "$lib_dir" -name "*.lib" | grep -v "sram" | while read lib; do
        echo "  /* from $lib */" >> "$merged_lib"
        awk '
        BEGIN {bracket=0; start=0}
        /library\s*\(.*\)\s*\{/ {start=1; bracket=1; next}
        {
            if (start) {
                if (/\{/) bracket++;
                if (/\}/) bracket--;
                if (bracket == 0) { start=0; next }
                print
            }
        }
        ' "$lib" >> "$merged_lib"
    done

    echo "}" >> "$merged_lib"
    echo "✅ Merged .lib created: $merged_lib"
fi

for file in "${benchmarks[@]}"; do
    mkdir -p "$output_dir/$file"

    cat > mapping_$file.tcl <<EOF
read_liberty -ignore_miss_func -ignore_miss_dir -ignore_miss_data_latch $merged_lib
read_verilog $benchmarks_dir/$file.v
prep -flatten
proc
synth -top $file -flatten
dfflibmap -liberty $merged_lib
opt
abc -liberty $merged_lib
techmap
splitnets -ports
opt_clean
write_verilog -noattr $output_dir/$file/${file}_orig_gtlvl.v
EOF

    yosys -s mapping_$file.tcl | tee $output_dir/$file/yosys_log.txt
    rm mapping_$file.tcl

    # 產生 SDC
    echo "🔧 Generating SDC file for $file..."
    python3 sdc.py $output_dir/$file/${file}_orig_gtlvl.v $output_dir/$file/${file}_orig_gtlvl.sdc --clock clk

    # 產生 OpenROAD TCL
    cat > def_$file.tcl <<EOF
read_lef $tech_lef
# read_lef $tech_tracks_lef

# 選項 1: 讀取所有 LEF 檔案（包含 SRAM）- 取消下面註解來啟用
$(find "$lef_dir" -name "*.lef" | sed 's/^/read_lef /')

read_liberty $merged_lib
read_verilog $output_dir/$file/${file}_orig_gtlvl.v
link_design $file
read_sdc $output_dir/$file/${file}_orig_gtlvl.sdc

define_pin_shape_pattern -layer M3 -x_step 0.036 -y_step 0.036 -region "*" -size {0.018 0.018}

# initialize_floorplan -utilization 0.5 -core_space 2 -site asap7sc7p5t
initialize_floorplan -utilization 0.3 -core_space 30 -site asap7sc7p5t
# initialize_floorplan -utilization 0.1 -core_space 85 -site asap7sc7p5t

make_tracks M2 -x_offset 0 -x_pitch 0.036 -y_offset 0 -y_pitch 0.036  
make_tracks M3 -x_offset 0 -x_pitch 0.036 -y_offset 0 -y_pitch 0.036

# place_pins
place_pins -hor_layers M2 -ver_layers M3 -min_distance 2 -corner_avoidance 1 -annealing

global_placement
detailed_placement
check_placement

write_def $output_dir/$file/${file}_placed.def
EOF

    # 執行 OpenROAD
    echo "🚀 Running OpenROAD placement for $file..."
    openroad -exit def_$file.tcl | tee $output_dir/$file/openroad_log.txt
    rm def_$file.tcl

done

# 選項 2: 只讀取標準 LEF 檔案（排除 SRAM）- 目前啟用
# $(find "$lef_dir" -name "*.lef" | grep -v -i sram | sed 's/^/read_lef /')