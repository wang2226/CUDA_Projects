#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <inttypes.h>
#include "common.h"

#define THREADS 16

void usage(int argc, char** argv);
void verify(int* sol, int* ans, int n);
void prefix_sum(int* src, int* prefix, int n);
void prefix_sum_p1(int* src, int* prefix, int n);
void prefix_sum_p2(int* src, int* prefix, int n);


int main(int argc, char** argv)
{
    // get inputs
    uint32_t n = 1048576;
    unsigned int seed = time(NULL);
    if(argc > 2) {
        n = atoi(argv[1]);
        seed = atoi(argv[2]);
    } else {
        usage(argc, argv);
        printf("using %"PRIu32" elements and time as seed\n", n);
    }


    // set up data
    int* prefix_array = (int*) AlignedMalloc(sizeof(int) * n);
    int* input_array = (int*) AlignedMalloc(sizeof(int) * n);
    srand(seed);
    for(int i = 0; i < n; i++) {
        input_array[i] = rand() % 100;
    }


    // set up timers
    uint64_t start_t;
    uint64_t end_t;
    InitTSC();


    // execute serial prefix sum and use it as ground truth
    start_t = ReadTSC();
    prefix_sum(input_array, prefix_array, n);
    end_t = ReadTSC();
    printf("Time to do O(N-1) prefix sum on a %"PRIu32" elements: %g (s)\n",
           n, ElapsedTime(end_t - start_t));


    // execute parallel prefix sum which uses a NlogN algorithm
    int* input_array1 = (int*) AlignedMalloc(sizeof(int) * n);
    int* prefix_array1 = (int*) AlignedMalloc(sizeof(int) * n);
    memcpy(input_array1, input_array, sizeof(int) * n);
    start_t = ReadTSC();
    prefix_sum_p1(input_array1, prefix_array1, n);
    end_t = ReadTSC();
    printf("Time to do O(NlogN) //prefix sum on a %"PRIu32" elements: %g (s)\n",
           n, ElapsedTime(end_t - start_t));
    verify(prefix_array, prefix_array1, n);


    // execute parallel prefix sum which uses a 2(N-1) algorithm
    memcpy(input_array1, input_array, sizeof(int) * n);
    memset(prefix_array1, 0, sizeof(int) * n);
    start_t = ReadTSC();
    prefix_sum_p2(input_array1, prefix_array1, n);
    end_t = ReadTSC();
    printf("Time to do 2(N-1) //prefix sum on a %"PRIu32" elements: %g (s)\n",
           n, ElapsedTime(end_t - start_t));
    verify(prefix_array, prefix_array1, n);


    // free memory
    AlignedFree(prefix_array);
    AlignedFree(input_array);
    AlignedFree(input_array1);
    AlignedFree(prefix_array1);


    return 0;
}

void usage(int argc, char** argv)
{
    fprintf(stderr, "usage: %s <# elements> <rand seed>\n", argv[0]);
}


void verify(int* sol, int* ans, int n)
{
    int err = 0;
    for(int i = 0; i < n; i++) {
        if(sol[i] != ans[i]) {
            err++;
        }
    }
    if(err != 0) {
        fprintf(stderr, "There was an error: %d\n", err);
    } else {
        fprintf(stdout, "Pass\n");
    }
}

void prefix_sum(int* src, int* prefix, int n)
{
    prefix[0] = src[0];
    for(int i = 1; i < n; i++) {
        prefix[i] = src[i] + prefix[i - 1];
    }
}

void prefix_sum_p1(int* src, int* prefix, int n)
{
    int* copy_out = (int*) AlignedMalloc(sizeof(int) * n);
    memset(copy_out, 0, sizeof(int) * n);
    memcpy(prefix, src, sizeof(int) * n);

    int i = 0;
    for (int j = 0; j < log2(n); j++) {

        #pragma omp parallel private(i) num_threads(THREADS)
        {
            #pragma omp for
            for (i = 1<<j; i < n; i++)
                copy_out[i] = prefix[i - (1<<j)] + prefix[i];

            #pragma omp for
            for (i = 1<<j; i < n; i++)
                prefix[i] = copy_out[i];
        }
    }

    AlignedFree(copy_out);
}

void prefix_sum_p2(int* src, int* prefix, int n)
{
    int last = src[n - 1];
    //Traverse the tree up (from leaves to root) and build partial sum at internal nodes of the tree
    for (int j = 0; j < log2(n); j++) {
        int stride =  1 << (j + 1);

        #pragma omp parallel num_threads(THREADS)
        {
            #pragma omp for
            for (int i = 0; i < n; i += stride) {
                src[i + (1 << (j+1)) - 1] = src[i + (1 << j) - 1] + src[i + (1 << (j+1)) - 1];
            }
        }
    }

    //Traverse back down to calculate the prefix sum from the partial sums
    src[n - 1] = 0;

    for (int j = log2(n) - 1; j >= 0; j--) {
        int stride =  1 << (j + 1);
        int temp = 0;

        #pragma omp parallel num_threads(THREADS)
        {
            #pragma omp for
            for (int i = 0; i < n; i += stride) {
                temp = src[i + (1 << j) - 1];
                //left child
                src[i + (1 << j) - 1] = src [i + (1 << (j + 1)) - 1];
                //right child
                src[i + (1 << (j + 1)) - 1] = temp + src[i + (1 << (j + 1)) - 1];
            }
        }
    }
    memcpy(prefix, src + 1, sizeof(int) * (n - 1));
    prefix[n - 1] = last + prefix[n - 2];
}
