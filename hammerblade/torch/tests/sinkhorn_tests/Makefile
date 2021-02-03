PYTORCH_DIR := $(BRG_BSG_BLADERUNNER_DIR)/hb-pytorch
MCPY_DIR := $(BRG_BSG_BLADERUNNER_DIR)/bsg_manycore/software/py
APP_SCRIPT := test_sinkhorn.py

test_run/vanilla_stats.csv: $(APP_SCRIPT)
	mkdir -p test_run
	cd test_run ; PYTHONPATH=$(PYTORCH_DIR) pycosim ../$(APP_SCRIPT) --hb > log.txt

test_run/manycore_stats.log: test_run/vanilla_stats.csv
	cd test_run ; PYTHONPATH=$(MCPY_DIR) python -m vanilla_parser
	cd test_run ; mv stats/manycore_stats.log .