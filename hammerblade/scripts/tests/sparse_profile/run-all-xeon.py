import sys
import os
import json
import copy
import subprocess

def fancy_print(route):
  for waypoint in route:
    print(waypoint)

with open('cmd.json',) as f:
  route = json.load(f)

fancy_print(route)
print()

print("total number of jobs: " + str(len(route)))

for i in [37]:
  for j in range(10):
    cmd = copy.deepcopy(route)
    cmd[i]['offload'] = True
    fancy_print(cmd)
    print()
    name = "recsys_kernel_%03d_xeon_%02d" % (i, j)
    sh_cmd = "mkdir -p " + name
    print(sh_cmd)
    os.system(sh_cmd)
    with open(name + "/cmd.json", 'w') as outfile:
      json.dump(cmd, outfile, indent=4, sort_keys=True)
    sh_cmd = "ln -s /work/global/lc873/work/sdh/playground/recsys_data/pytorch-apps/recsys/data " + name + "/data"
    print(sh_cmd)
    os.system(sh_cmd)
    script = "(cd " + name + "; python /work/global/lc873/work/sdh/playground/recsys_data/pytorch-apps/recsys/recsys.py --batch-size 256 --nbatch 1 --training > out.std 2>&1)"
    with open(name + "/run.sh", 'w') as outfile:
      outfile.write(script)
    print("starting cosim job ...")
    # run the job 3 times
    for runs in range(3):
      cosim_run = subprocess.Popen(["sh", name + "/run.sh"], env=os.environ)
      cosim_run.wait()
