OBJECTS 			:=simulator.o batch_learning.o deep_r.o
GENERATED_CODE_DIR		:=pattern_recognition_1_1_CODE
GENN_USERPROJECT_INCLUDE	:=$(abspath $(dir $(shell which genn-buildmodel.sh))../userproject/include)

CUDA_PATH			?=/usr/local/cuda
NVCC				:= $(CUDA_PATH)/bin/nvcc
NVCCFLAGS			:= -x cu -arch sm_50 -Xcudafe "--diag_suppress=2937" -std=c++11 -I$(GENN_USERPROJECT_INCLUDE)
LINKFLAGS			:=-L$(GENERATED_CODE_DIR) -L$(CUDA_PATH)/lib64 -lcuda -lcudart -lrunner -arch sm_50 --linker-options "-rpath $(GENERATED_CODE_DIR)"

.PHONY: all clean generated_code

all: pattern_recognition_1_1

pattern_recognition_1_1: $(OBJECTS) generated_code
	$(NVCC) -o $@ $(OBJECTS) $(LINKFLAGS)

%.o: %.cc
	@$(NVCC) $(NVCCFLAGS) -dc $<

generated_code:
	$(MAKE) -C $(GENERATED_CODE_DIR)

clean:
	$(MAKE) -C $(GENERATED_CODE_DIR) clean
	rm -f *.o
	rm pattern_recognition_1_1
