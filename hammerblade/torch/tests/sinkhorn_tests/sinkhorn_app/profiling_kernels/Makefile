PYTORCH_DIR := $(BRG_BSG_BLADERUNNER_DIR)/hb-pytorch
MCPY_DIR := $(BRG_BSG_BLADERUNNER_DIR)/bsg_manycore/software/py
APP_SCRIPT := test_sinkhorn_simple.py

# There are four kernels to offload individually.
KERNELS := 0 1 2 3

ALL_LOGS := $(KERNELS:%=run_%/manycore_stats.log) cpu_run/log.txt
results.csv: $(ALL_LOGS) collect.py
	python collect.py > $@

run_%/vanilla_stats.csv: $(APP_SCRIPT)
	mkdir -p run_$*
	cd run_$* ; PYTHONPATH=$(PYTORCH_DIR) pycosim ../$(APP_SCRIPT) --hb $* > log.txt

run_%/manycore_stats.log: run_%/vanilla_stats.csv
	cd run_$* ; PYTHONPATH=$(MCPY_DIR) python -m vanilla_parser
	cd run_$* ; mv stats/manycore_stats.log .

cpu_run/log.txt: $(APP_SCRIPT)
	mkdir -p cpu_run
	cd cpu_run ; PYTHONPATH=$(PYTORCH_DIR) python ../$(APP_SCRIPT) > log.txt
