cc = gcc

sources = main.c mmio.c common.c
objects = $(addsuffix .o, $(basename $(sources)))

flags = -g -std=c11 -fopenmp
target = cg

all: $(target)

$(target) : $(objects)
	$(cc) $(flags) -o $(target) $(objects) -lm

%.o : %.c
	$(cc) -c $(flags) $< -o $@ -lm

clean:
	rm -rf $(target) $(objects)
