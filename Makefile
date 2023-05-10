ROOT_DIR := .
STAN_ROOT_DIR ?= /Users/marioberaha/dev/bayesmix_origin/lib/math
PG_DIR := lib/polyagamma 
GSL_HOME ?= /opt/homebrew/Cellar/gsl/2.7/

SRC_DIR := $(ROOT_DIR)/src
SPIKES_DIR := $(SRC_DIR)/spikes
PROTO_DIR := $(ROOT_DIR)/protos

CXX = /opt/homebrew/opt/llvm/bin/clang
CFLAGS = \
	-std=c++1y \
	-MMD \
	-I$(STAN_ROOT_DIR) \
	-I$(STAN_ROOT_DIR)/lib/eigen_*/ \
	-I$(STAN_ROOT_DIR)/lib/boost_*/  \
	-I$(STAN_ROOT_DIR)/lib/tbb*/include  \
	-I$(PG_DIR) -I$(PG_DIR)/include -I$(PROTO_DIR) \
	-I/opt/homebrew/opt/llvm/include \
	-I/opt/homebrew/include/ \
	-$(shell pkg-config --cfalgs protobuf) \
	-D_REENTRANT -fPIC \
	-$(shell python3 -m pybind11 --includes) \
	-Wno-deprecated-declarations \
	-O3 -funroll-loops -ftree-vectorize -fopenmp 

LDLIBS = \
 	$(shell pkg-config --libs protobuf) \
	$(shell python3-config --ldflags) -lc++ \
	-lgsl -L$(GSL_HOME)/lib -lgslcblas -lpthread -L/opt/homebrew/opt/llvm/lib \

LDFLAGS = -D_REENTRANT -O3 -fopenmp -L/opt/homebrew/opt/llvm/lib \
	-L$(GSL_HOME)/lib \
	-L$(shell python3-config --ldflags)

PROTO_SRCS = $(wildcard $(PROTO_DIR)/cpp/*.cpp)
PG_SRCS = $(wildcard $(PG_DIR)/*.cpp) $(wildcard $(PG_DIR)/include/*.cpp)
SPIKES_SRCS = $(wildcard $(SPIKES_DIR)/*.cpp)
OUR_SRCS = $(wildcard $(SRC_DIR)/*.cpp)

OUR_SRCS := $(filter-out ./src/python_exports.cpp, $(OUR_SRCS))

SRCS = $(PROTO_SRCS) $(PG_SRCS) $(OUR_SRCS)
OBJS = $(subst .cpp,.o, $(SRCS))
DEPENDS := $(patsubst %.cpp,%.d,$(SRCS))

SPIKES_EXECS = $(subst .cpp,.out, $(SPIKES_SRCS))
SPIKES_OBJS =  $(subst .cpp,.o, $(SPIKES_SRCS))

info:
	@echo " Info..."
	@echo " ROOT_DIR  = $(ROOT_DIR)"
	@echo " PROTO_DIR = $(PROTO_DIR)"
	@echo " SRC_DIR = $(SRC_DIR)"
	@echo " SPIKES_DIR = $(SPIKES_DIR)"
	@echo " SOURCES = $(SRCS)"
	@echo " OBJECTS = $(OBJS)"
	@echo " EXECS = $(SPIKES_EXECS)"
	@echo " STAN_ROOT_DIR = $(STAN_ROOT_DIR)"
	@echo " DEPENDS = $(DEPENDS)"

all: generate_pybind $(SPIKES_EXECS)

generate_pybind: $(OBJS)
	$(CXX) -shared $(CFLAGS)
		src/python_exports.cpp -o \
		pp_mix_cpp.cpython-39-darwin.so \
		$(OBJS) $(LDLIBS) $(LDFLAGS) -fopenmp

$(SPIKES_EXECS): %.out: %.o $(OBJS)
	$(CXX) $(CFLAGS) $(LDFLAGS) -o $@ $(OBJS) $< $(LDLIBS)

-include $(DEPENDS)

%.o : %.cpp
	$(CXX) $(CFLAGS) -MMD -MP -c $< -o $@

clean:
	rm $(OBJS) $(SPIKES_OBJS) run_from_file.o $(DEPENDS)

distclean: clean

compile_protos:
	@ mkdir -p $(PROTO_DIR)/cpp;
	@ mkdir -p $(PROTO_DIR)/py;
	@ for filename in $(PROTO_DIR)/*.proto; do \
		protoc --proto_path=$(PROTO_DIR) --python_out=$(PROTO_DIR)/py/ $$filename; \
		protoc --proto_path=$(PROTO_DIR) --cpp_out=$(PROTO_DIR)/cpp/ $$filename; \
	done
	@ for filename in $(PROTO_DIR)/cpp/*.cc; do \
	    mv -- "$$filename" "$${filename%.cc}.cpp"; \
	done

	touch $(PROTO_DIR)/__init__.py
	touch $(PROTO_DIR)/py/__init__.py

	2to3 --output-dir=$(PROTO_DIR)/py/ -W -n $(PROTO_DIR)/py/
