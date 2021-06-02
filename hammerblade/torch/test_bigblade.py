import glob
import os
import subprocess

qsub_template = '(cd {path}; qsub -N {job_name} -l walltime=144:00:00 -l nodes=1:ppn=12 -l mem="50gb" -d {path} -o {path}/qsub.out -e {path}/qsub.err -V run.sh)\n'

def get_test_files(path):
    files = glob.glob(path + "/*.py")
    src_files = []
    for f in files:
        if f.split("/")[-1].startswith("test_"):
            src_files.append(f)
    return src_files

def run_test_file_on_cluster(test_file):
    name = "bigblade_" + test_file.split("/")[-1].split(".")[0]
    print(name)
    sh_cmd = "mkdir -p " + name
    os.system(sh_cmd)

    script = "(module load synopsys-2020/synopsys-vcs-R-2020.12; source /work/global/lc873/work/sdh/venv_cosim/bin/activate; source setup-gcc8.sh; pycosim -m pytest -vs -k-hypothesis " + test_file + " > out.std 2>&1)\n"
    with open(name + "/run.sh", 'w') as outfile:
        outfile.write(script)

    # current path
    path = str(os.path.abspath(os.getcwd())) + "/" + name
    print(path)

    qsub_starter = qsub_template.format(job_name=name, path=path)
    print(qsub_starter)
    with open(name + "/qsub.sh", 'w') as outfile:
        outfile.write(qsub_starter)

    print("starting cosim job ...")
    cosim_run = subprocess.Popen(["sh", name + "/qsub.sh"], env=os.environ)


# test

files = get_test_files("/work/global/lc873/work/sdh/brg-hb-pytorch/hammerblade/torch/tests/")

for f in files:
    run_test_file_on_cluster(f)
