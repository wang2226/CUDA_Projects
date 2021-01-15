CC =  /usr/bin/gcc
CFLAGS  = -Wall -g -std=c99
CFLAGS += -O3 -fopenmp 
CFLAGS += -D_POSIX_C_SOURCE=200112L
LIBS =

SRC  = $(notdir $(wildcard *.c)) 
OBJS = $(addsuffix .o, $(basename $(SRC)))
EXEC = prefix 

all: ${EXEC}

${EXEC}: ${OBJS}
	${CC} ${CFLAGS} ${LDFLAGS} $^ -o $@ ${LIBS} ${CUDA_LDFLAGS} -lm

%.o : %.c
	$(CC) ${CFLAGS} ${INC} ${NVCCINC} -c $< -o $@ ${LIBS} -lm

clean:
	rm -f *.o *.d *~ *.a *.so *.s ${EXEC}

