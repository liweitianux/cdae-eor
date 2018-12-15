all: deps cwtcore

deps:
	(cd deps && make)

cwtcore:
	python3 setup.py build_ext --inplace
