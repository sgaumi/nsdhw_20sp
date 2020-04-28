FLAGS ?=

UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
MKLROOT ?= /opt/intel/mkl
MKLEXT ?= a
CXXFLAGS :=
endif

ifeq ($(UNAME_S),Linux)
MKLROOT ?= ${HOME}/opt/conda
MKLEXT ?= so
CXXFLAGS := -Wl,--no-as-needed
endif

MKLLINKLINE := \
	${MKLROOT}/lib/libmkl_intel_lp64.${MKLEXT} \
	${MKLROOT}/lib/libmkl_sequential.${MKLEXT} \
	${MKLROOT}/lib/libmkl_core.${MKLEXT} \
	-lpthread -lm -ldl

CXX = g++
CXXFLAGS := ${CXXFLAGS} \
	-std=c++11 -shared -O3 -g -m64 \
	-I${MKLROOT}/include \
	${MKLLINKLINE} \
	${FLAGS}

PYBIND := -fPIC `python3 -m pybind11 --includes`

.PHONY: default
default: Matrix

matrix: Matrix.cpp
	${CXX} ${CXXFLAGS} ${PYBIND} $< -o $@`python3-config --extension-suffix`

.PHONY: test
test:
	${CXX} ${CXXFLAGS} ${PYBIND} Matrix.cpp -o Matrix`python3-config --extension-suffix`
	pytest

.PHONY: clean
clean:
	rm -frd *.so __pycache__ .pytest_cache performance.txt
	touch performance.txt
