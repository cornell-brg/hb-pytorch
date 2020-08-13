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

for i in range(len(route)):
  cmd = copy.deepcopy(route)
  cmd[i]['offload'] = True
  fancy_print(cmd)
  print()
  name = "test_kernel_%03d_hb" % i
  sh_cmd = "mkdir -p " + name
  print(sh_cmd)
  os.system(sh_cmd)
  with open(name + "/cmd.json", 'w') as outfile:
    json.dump(cmd, outfile, indent=4, sort_keys=True)
  sh_cmd = "ln -s ../test.py " + name + "/test.py"
  print(sh_cmd)
  os.system(sh_cmd)
  script = "(cd " + name + "; pycosim test.py > out.std 2>&1)"
  with open(name + "/run.sh", 'w') as outfile:
    outfile.write(script)
  print("starting cosim job ...")
  cosim_run = subprocess.Popen(["sh", name + "/run.sh"], env=os.environ)

