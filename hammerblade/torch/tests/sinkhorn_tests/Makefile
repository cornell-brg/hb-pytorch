PYTORCH_DIR := $(BRG_BSG_BLADERUNNER_DIR)/hb-pytorch
MCPY_DIR := $(BRG_BSG_BLADERUNNER_DIR)/bsg_manycore/software/py
APP_SCRIPT := test_sinkhorn.py

# There are four kernels to offload individually.
KERNELS := 0 1 2 3 4 5

ALL_LOGS := $(KERNELS:%=run_%/manycore_stats.log) cpu_run/log.txt
ALL_SAIF := $(KERNELS:%=saif_%/run.saif.gz)
results.csv: $(ALL_LOGS) collect.py $(ALL_SAIF)
	python collect.py > $@

# run_4/vanilla_stats.csv: APP_SCRIPT=test_cdist
run_%/vanilla_stats.csv: $(APP_SCRIPT)
	mkdir -p run_$*
	cd run_$* ; PYTHONPATH=$(PYTORCH_DIR) pycosim.profile ../$(APP_SCRIPT) --hb $* > log.txt

# saif_4/run.saif: APP_SCRIPT=test_cdist
saif_%/run.saif: $(APP_SCRIPT)
	mkdir -p saif_$*
	cd saif_$* ; PYTHONPATH=$(PYTORCH_DIR) pycosim.saif ../$(APP_SCRIPT) --hb $* > log.txt

saif_%/run.saif.gz: saif_%/run.saif
	cd saif_$* ; gzip run.saif

run_%/manycore_stats.log: run_%/vanilla_stats.csv
	cd run_$* ; PYTHONPATH=$(MCPY_DIR) python -m vanilla_parser
	cd run_$* ; mv stats/manycore_stats.log .

cpu_run/log.txt: $(APP_SCRIPT)
	mkdir -p cpu_run
	cd cpu_run ; PYTHONPATH=$(PYTORCH_DIR) python ../$(APP_SCRIPT) > log.txt

cdist: run_4/manycore_stats.log saif_4/run.saif
	cd saif_4 ; gzip run.saif
	
run_5/manycore_stats.log: test_cdist.py
	mkdir -p run_4
	cd run_4 ; PYTHONPATH=$(PYTORCH_DIR) pycosim.profile ../test_cdist.py > log.txt
	cd run_4 ; PYTHONPATH=$(MCPY_DIR) python -m vanilla_parser
	cd run_4 ; mv stats/manycore_stats.log .

saif_5/run.saif: test_cdist.py
	mkdir -p saif_4
	cd saif_4 ; PYTHONPATH=$(PYTORCH_DIR) pycosim.saif ../test_cdist.py --cosim_scale 10 > log.txt
