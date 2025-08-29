import odb, os
platform = "asap7sc7p5t"
design   = "s27"
def_file = r"/workspace/docker_ruan/gtlvl_design/s27/s27.def"
out_dir  = r"/workspace/docker_ruan/gtlvl_design/s27/bookshelf_run"

lef_list = [
    r"/workspace/docker_ruan/ic_cad/ICCAD25/ASAP7/techlef/asap7_tech_1x_201209.lef",
    r"/workspace/docker_ruan/ic_cad/ICCAD25/ASAP7/LEF/asap7sc7p5t_28_L_1x_220121a.lef",
    r"/workspace/docker_ruan/ic_cad/ICCAD25/ASAP7/LEF/asap7sc7p5t_28_R_1x_220121a.lef",
    r"/workspace/docker_ruan/ic_cad/ICCAD25/ASAP7/LEF/asap7sc7p5t_28_SL_1x_220121a.lef",
    r"/workspace/docker_ruan/ic_cad/ICCAD25/ASAP7/LEF/asap7sc7p5t_28_SRAM_1x_220121a.lef",
    r"/workspace/docker_ruan/ic_cad/ICCAD25/ASAP7/LEF/sram_asap7_16x256_1rw.lef",
    r"/workspace/docker_ruan/ic_cad/ICCAD25/ASAP7/LEF/sram_asap7_32x256_1rw.lef",
    r"/workspace/docker_ruan/ic_cad/ICCAD25/ASAP7/LEF/sram_asap7_64x64_1rw.lef",
    r"/workspace/docker_ruan/ic_cad/ICCAD25/ASAP7/LEF/sram_asap7_64x256_1rw.lef",
]

db = odb.dbDatabase.create()
for lf in lef_list:
    odb.read_lef(db, lf)

# OpenROAD v2.0-22xxx 之後 read_def 需給 dbTech 而非 dbDatabase
odb.read_def(db.getTech(), def_file)

os.makedirs(out_dir, exist_ok=True)
odb.write_db(db, f"{out_dir}/{platform}_{design}.odb")

