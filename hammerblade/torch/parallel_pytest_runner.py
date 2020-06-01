# import essential modules
import sys
import os
import subprocess
import pathlib

# figure out current directory
current_path = str(pathlib.Path(__file__).parent.absolute())

# add current and tests paths
sys.path.append(current_path + "/tests")
sys.path.append(current_path)

# construct target list
targets = []

# Get test list from the command line if provided
if len(sys.argv) > 1:
    for t in sys.argv[1:]:
        targets.append(current_path + "/tests/" + t)
else:
    # collect pytest files. somehow it can't do so automatically
    # inside COSIM
    import glob
    targets = glob.glob(current_path + "/tests/test_*.py")

targets.sort()

print()
print(" files collected by pytest runner:")
print()
for t in targets:
    print(" " + t)
print()

# create a tmp test dir
sh_cmd = "mkdir -p pytorch_cosim_unittest"
print(sh_cmd)
os.system(sh_cmd)

# default batch size == 32 jobs
batch_size = 32

# loop through the test files
batch = 0
while batch < len(targets):
    jobs = []
    for i in range(batch, batch + batch_size):
        if i < len(targets):
            # get the test filename
            target = targets[i]
            test_name = (target.split("/")[-1]).split(".")[0]
            print(test_name)
            # create a folder for holding the script
            sh_cmd = "mkdir -p pytorch_cosim_unittest/" + test_name
            print(sh_cmd)
            os.system(sh_cmd)
            # generate script
            script = "(cd pytorch_cosim_unittest/" + test_name + "; pycosim " + current_path + "/pytest_runner.py " + test_name + ".py > out.std 2>&1)"
            with open("pytorch_cosim_unittest/" + test_name + "/run.sh", 'w') as outfile:
                outfile.write(script)
            # start the job
            cosim_run = subprocess.Popen(["sh", "pytorch_cosim_unittest/" + test_name + "/run.sh"], env=os.environ)
            jobs.append(cosim_run)
    # wait for jobs to finish
    for job in jobs:
        job.wait()
    batch += batch_size
