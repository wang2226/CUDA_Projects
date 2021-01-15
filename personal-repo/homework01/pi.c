#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "common.h"
#include <inttypes.h>
#include <time.h>


#define PI 3.1415926535
#define THREADS 32

void usage(int argc, char** argv);
double calcPi_Serial(int num_steps);
double calcPi_P1(int num_steps);
double calcPi_P2(int num_steps);
double calcPi_MC(long iterations);
double calcPi_MCP(long iterations);


int main(int argc, char** argv)
{
    // get input values
    uint32_t num_steps = 100000;
    if(argc > 1) {
        num_steps = atoi(argv[1]);
    } else {
        usage(argc, argv);
        printf("using %"PRIu32"\n", num_steps);
    }
    fprintf(stdout, "The first 10 digits of Pi are %0.10f\n", PI);


    // set up timer
    uint64_t start_t;
    uint64_t end_t;
    InitTSC();


    // calculate in serial
    start_t = ReadTSC();
    double Pi0 = calcPi_Serial(num_steps);
    end_t = ReadTSC();
    printf("Time to calculate Pi serially with %"PRIu32" steps is: %g\n",
           num_steps, ElapsedTime(end_t - start_t));
    printf("Pi is %0.10f\n", Pi0);

    // calculate in parallel with reduce
    start_t = ReadTSC();
    double Pi1 = calcPi_P1(num_steps);
    end_t = ReadTSC();
    printf("Time to calculate Pi in // with %"PRIu32" steps is: %g\n",
           num_steps, ElapsedTime(end_t - start_t));
    printf("Pi is %0.10f\n", Pi1);


    // calculate in parallel with atomic add
    start_t = ReadTSC();
    double Pi2 = calcPi_P2(num_steps);
    end_t = ReadTSC();
    printf("Time to calculate Pi in // + atomic with %"PRIu32" steps is: %g\n",
           num_steps, ElapsedTime(end_t - start_t));
    printf("Pi is %0.10f\n", Pi2);

    // calculate using Monte Carlo in serial
    start_t = ReadTSC();
    double Pi3 = calcPi_MC(num_steps);
    end_t = ReadTSC();
    printf("Time to calculate Pi serially using Monte Carlo with %"PRIu32" steps is: %g\n",
           num_steps, ElapsedTime(end_t - start_t));
    printf("Pi is %0.10f\n", Pi3);

    // calculate using Monte Carlo in parallel with reduce
    start_t = ReadTSC();
    double Pi4 = calcPi_MCP(num_steps);
    end_t = ReadTSC();
    printf("Time to calculate Pi in // using Monte Carlo with %"PRIu32" steps is: %g\n",
           num_steps, ElapsedTime(end_t - start_t));
    printf("Pi is %0.10f\n", Pi4);

    return 0;
}


void usage(int argc, char** argv)
{
    fprintf(stdout, "usage: %s <# steps>\n", argv[0]);
}

double calcPi_Serial(int num_steps)
{
    double pi = 0.0;
    double x = 0.0;
    double y = 0.0;
    double sum = 0.0;
    double step;

    step = 1.0 / (double) num_steps;

    for(int i = 0; i < num_steps; i++) {
        x = i * step;
        y = sqrt(1 - x * x);
        sum += y * step;
    }
    pi = 4 * sum;

    return pi;
}

double calcPi_P1(int num_steps)
{
    double pi = 0.0;
    double x = 0.0;
    double y = 0.0;
    double sum = 0.0;
    double step;
    //int id = 0;
    //uint64_t start_t;
    //uint64_t end_t;

    step = 1.0 / (double) num_steps;

    // # pragma omp parallel firstprivate(x, y, id) num_threads(THREADS)
    # pragma omp parallel firstprivate(x, y) num_threads(THREADS)
    {
        //id = omp_get_thread_num();
        //start_t = ReadTSC();
        # pragma omp for reduction(+:sum) schedule(static)
        for(int i = 0; i < num_steps; i++) {
            x = i * step;
            y = sqrt(1 - x * x);
            sum += y * step;
        }
        //end_t = ReadTSC();
        //printf("%d  %g\n", id, ElapsedTime(end_t - start_t));
    }
    pi = 4 * sum;

    return pi;
}

double calcPi_P2(int num_steps)
{
    double pi = 0.0;
    double x = 0.0;
    double y = 0.0;
    double sum = 0.0;
    double sum_thread = 0.0;
    double step;

    step = 1.0 / (double) num_steps;

    # pragma omp parallel firstprivate(x, y, sum_thread) num_threads(THREADS)
    {
        # pragma omp for schedule(static)
        for(int i = 0; i < num_steps; i++) {
            x = i * step;
            y = sqrt(1 - x * x);
            sum_thread += y * step;
        }

        #pragma omp atomic
        sum += sum_thread;
    }

    pi = 4 * sum;

    return pi;
}

double calcPi_MC(long iterations) {
    double pi = 0.0;
    double x = 0.0;
    double y = 0.0;
    long hit = 0.0;
    unsigned int seed = (unsigned) time(NULL);

    for(long i = 0; i < iterations; i++) {
        x = (double)rand_r(&seed) / RAND_MAX;
        y = (double)rand_r(&seed) / RAND_MAX;

        if(x * x + y * y <= 1.0) {
            hit++;
        }
    }
    pi = 4.0 * (double) hit / (double) iterations;

    return pi;
}

double calcPi_MCP(long iterations){
    double pi = 0.0;
    double x = 0.0;
    double y = 0.0;
    long hit = 0.0;
    unsigned int seed = (unsigned) time(NULL);

    # pragma omp parallel firstprivate(x, y, seed) shared(hit) num_threads(THREADS)
    {
        # pragma omp for reduction(+:hit) schedule(static)
        for(long i = 0; i < iterations; i++) {
            x = (double)rand_r(&seed) / RAND_MAX;
            y = (double)rand_r(&seed) / RAND_MAX;

            if(x * x + y * y <= 1.0) {
                hit++;
            }
        }
    }

    pi = 4.0 * (double) hit / (double) iterations;

    return pi;
}
