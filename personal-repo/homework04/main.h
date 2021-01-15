#ifndef MAIN_H
#define MAIN_H
#include "mmio.h"
#include "common.h"

void usage(int argc, char** argv);
void print_matrix_info(char* fileName, MM_typecode matcode, 
                       int m, int n, int nnz);
void check_mm_ret(int ret);
void read_info(char* fileName, int* is_sym);
void convert_coo_to_csr(int* row_ind, int* col_ind, double* val, 
                        int m, int n, int nnz,
                        unsigned int** csr_row_ptr, unsigned int** csr_col_ind,
                        double** csr_vals);
void read_vector(char* fileName, double** vector, int* vecSize);
void spmv(unsigned int* csr_row_ptr, unsigned int* csr_col_ind, 
          double* csr_vals, int m, int n, int nnz, 
          double* vector_x, double *res);
void store_result(char *fileName, double* res, int m);
void print_time(double timer[]);
void expand_symmetry(int m, int n, int* nnz_, int** row_ind, int** col_ind, 
                     double** val);
void init_vector(double* a, int m, unsigned int seed);
int cg(unsigned int* csr_row_ptr, unsigned int* csr_col_ind,
        double* csr_vals, int m, int n, int nnz,
        double* x, double* b, int max_iter, double tol);
double ddot(const int n, double* x, const int incx, double* y, const int incy);
double dnrm2(const int n, double* x, const int incx);
void vec_add(const int n, const double a, const double* x, const int incx, 
             double* y, const int incy, double* z);
void verify_cg(int* row_ptr, int* col_ind, double* vals, int m, int n, int nnz,
               double* x, double* b);
#endif

