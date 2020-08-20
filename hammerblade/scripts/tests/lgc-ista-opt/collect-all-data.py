import sys
import os
import json
import copy
import subprocess

sys.path.append('/scratch/users/zz546/hb-pytorch/hammerblade/scripts/')

from compare_aten_op import compare, average_aten_op

# ======================================================
# Helper
# ======================================================

def fancy_print(route):
  for waypoint in route:
    print(waypoint)

def compare_wrapper(full, chunk, stats):
  assert len(full) == len(chunk)
  ops = []
  for i in range(len(full)):
    ops.append(compare(full[i], chunk[i], stats))
  aten_op = average_aten_op(ops)
  return aten_op

# ======================================================
# Read kernels
# ======================================================

with open('lgc_ista.json',) as f:
  route = json.load(f)

fancy_print(route)
print()

print("total number of jobs: " + str(len(route)))

# ======================================================
# Main loop
# ======================================================

kernels = []

for i in range(len(route)):
  hb_name = "lgc_ista_hb_%d" % i
  name = "lgc_ista_data_new_format_%d" % i
  # work dir
  sh_cmd = "mkdir -p " + name
  print(sh_cmd)
  os.system(sh_cmd)
  for j in range(5):
    xeon_name = "lgc_ista_xeon_%d_%d" % (i, j)
    # collect full stack
    sh_cmd = "cp " + xeon_name + "/out.std " + name + ("/full-%02d.std" % j)
    print(sh_cmd)
    os.system(sh_cmd)
  # collect cosim chunk stack for cross check
  sh_cmd = "cp " + hb_name + "/out.std " + name + "/chunk-cosim.std"
  print(sh_cmd)
  os.system(sh_cmd)
  # run uw's profiling data tool
  sh_cmd = "(cd " + hb_name + "; python /home/zz546/bsg_bladerunner/bsg_manycore/software/py/vanilla_parser/stats_parser.py --stats vanilla_stats.csv)"
  print(sh_cmd)
  os.system(sh_cmd)
  # collect output
  sh_cmd = "cp " + hb_name + "/stats/manycore_stats.log " + name + "/"
  print(sh_cmd)
  os.system(sh_cmd)

  # ======================================================
  # Actual data processing
  # ======================================================
  full = ["{0}/full-{1:02d}.std".format(name, j) for j in range(5)]
  chunk = ["{0}/full-{1:02d}.std".format(name, j) for j in range(5)]
  stats = "{0}/manycore_stats.log".format(name)
  kernels.append(compare_wrapper(full, chunk, stats))
  print(kernels[-1].draw_cpu_log())
  print()
  print(kernels[-1])
  # ======================================================
  # End of actual data procesing
  # ======================================================

  # run postprocessing script for cross check
  sh_cmd = "python /scratch/users/zz546/hb-pytorch/hammerblade/scripts/compare_aten_op.py --full {0}/full-00.std --chunk {0}/chunk-cosim.std --manycore-stats {0}/manycore_stats.log > {0}/cross_check.txt 2>&1".format(name)
  print(sh_cmd)
  os.system(sh_cmd)

# ======================================================
# Print all kernels at the end
# ======================================================
total_xeon = 0
total_host = 0
total_device = 0

buf = """
+-----------------------+---------------+--------------------+--------------------+-----------------+---------------------+-----------------+-------------------+
|        ATen OP        |     Input     |     Full  Size     |     Chunk Size     |    Xeon Time    |    HB Total Time    |    Host Time    |    Device Time    |
+-----------------------+---------------+--------------------+--------------------+-----------------+---------------------+-----------------+-------------------+"""

for k in kernels:
  total_xeon += k.xeon_time
  total_host += k.hb_host_time
  total_device += k.hb_device_time
  buf += k.fancy_print()

template = "\n| {func:<22}| {tensor:<14}| {full:<19}| {chunk:<19}|{xeon:>16} |{hb:>20} |{host:>16} |{device:>18} |"

buf += template.format(
                func = "Total",
                tensor = "",
                full = "",
                chunk = "",
                xeon = "{:.2f}".format(total_xeon / 1000.0),
                hb = "{:.2f}".format((total_host + total_device) / 1000.0),
                host = "{:.2f}".format(total_host / 1000.0),
                device = "{:.2f}".format(total_device / 1000.0))
buf += """
+-----------------------+---------------+--------------------+--------------------+-----------------+---------------------+-----------------+-------------------+"""
print(buf)
