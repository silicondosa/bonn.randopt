
.PHONY: all simple benchmarks

all: simple

simple:
	python examples/simple.py

benchmarks: 
	python examples/benchmarks.py
