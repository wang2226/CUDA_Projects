CPP = g++
NVCC =  /usr/local/cuda-11.1/bin/nvcc
CCFLAGS  = -O3 -fopenmp -lm -std=c++11

CUDA_INSTALL_PATH ?= /usr/local/cuda-11.1
TOOLKIT_INSTALL_PATH ?= /usr/local/cuda-11.1/samples
CUDA_LDFLAGS = -L$(CUDA_INSTALL_PATH)/lib64 -lcudart -lcublas -lcurand -lcusolver
NVCCFLAGS = -g -gencode arch=compute_70,code=sm_70
NVCCINC += -I$(CUDA_INSTALL_PATH)/include
NVCCINC += -I$(TOOLKIT_INSTALL_PATH)/common/inc

SRC  = $(notdir $(wildcard *.cpp)) $(notdir $(wildcard *.cu))

OBJS = $(addsuffix .o, $(basename $(SRC)))
EXEC = gomoku_serial gomoku_omp \
			 gomoku_cuda_naive gomoku_cuda_seq \
			 gomoku_cuda_pvs gomoku_cuda_dynamic \
			 # gomoku_cpu

all: ${EXEC}

gomoku_serial: gomoku_serial.o common.o
	${CPP} ${CCFLAGS} ${LDFLAGS} $^ -o $@ ${LIBS}

gomoku_omp: gomoku_omp.o common.o
	${CPP} ${CCFLAGS} ${LDFLAGS} $^ -o $@ ${LIBS}

#gomoku_cpu: gomoku_cpu.o
	#${CPP} ${CFLAGS} ${LDFLAGS} $^ -o $@ ${LIBS}

gomoku_cuda_naive: gomoku_cuda_naive.o common.o
	${CPP} ${CFLAGS} ${LDFLAGS} $^ -o $@ ${LIBS} ${CUDA_LDFLAGS}

gomoku_cuda_seq: gomoku_cuda_seq.o common.o
	${CPP} ${CFLAGS} ${LDFLAGS} $^ -o $@ ${LIBS} ${CUDA_LDFLAGS}

gomoku_cuda_pvs: gomoku_cuda_pvs.o
	${CPP} ${CFLAGS} ${LDFLAGS} $^ -o $@ ${LIBS} ${CUDA_LDFLAGS}

gomoku_cuda_dynamic: gomoku_cuda_dynamic.o
	$(NVCC) -lcudadevrt  ${NVCCCFLAGS} $^ -o $@ ${CUDA_LDFLAGS}

%.o : %.cpp
	$(CPP) ${CCFLAGS} ${INC} -c $< -o $@ ${LIBS} ${CUDA_LD_FLAGS} ${NVCCINC}

%.o : %.cu
	$(NVCC) ${NVCCCFLAGS} ${INC} -c $< -o $@ ${CUDA_LD_FLAGS} ${NVCCFLAGS} ${NVCCINC}

gomoku_cuda_dynamic.o: gomoku_cuda_dynamic.cu
	$(NVCC) -Iinclude -c $< -o $@  ${NVCCCFLAGS} -G  -rdc=true  $(NVCCINC)


clean:
	rm -f *.o ${EXEC}
