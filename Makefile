.PHONY: all notebook clean profile

ts=$(shell date "+%Y%m%d_%H%M%S")
stats_path="$(shell pwd)/profiling/$(ts)/output.pstats"
png_path="$(shell pwd)/profiling/$(ts)/output.png"
profile_timeout=5 # 1 minute

# default target, when make executed without arguments
all: notebook

notebook:
	./venv/bin/jupyter notebook

profile:
	echo "Storing results in $(shell pwd)/profiling/$(ts)"
	mkdir -p "./profiling/$(ts)"
	./venv/bin/python -m cProfile -o $(stats_path) ./profiling/main.py 
	sleep 1 && ./venv/bin/gprof2dot -f pstats $(stats_path) | dot -Tpng -o $(png_path)
