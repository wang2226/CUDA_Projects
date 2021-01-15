#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <omp.h>
#include <math.h>
#include "main.h"

#define MAX_FILENAME 256
#define MAX_NUM_LENGTH 100

#define NUM_TIMERS       4
#define LOAD_TIME        0
#define CONVERT_TIME     1
#define CG_TIME          2
#define STORE_TIME       3

int main(int argc, char** argv)
{
    // program info
    usage(argc, argv);


    // Initialize timess
    double timer[NUM_TIMERS];
    uint64_t t0;
    for(unsigned int i = 0; i < NUM_TIMERS; i++) {
        timer[i] = 0.0;
    }
    InitTSC();


    // get CG parameters
    int max_iter = atoi(argv[4]);
    double tol = atof(argv[5]);
    unsigned int seed = atoi(argv[6]);

    // Read the sparse matrix file and get its info first
    char matrixName[MAX_FILENAME];
    strcpy(matrixName, argv[1]);
    int is_symmetric = 0;
    read_info(matrixName, &is_symmetric);


    // Read the sparse matrix and store it in row_ind, col_ind, and val,
    // also known as co-ordinate format (COO).
    int ret;
    MM_typecode matcode;
    int m;
    int n;
    int nnz;
    int *row_ind;
    int *col_ind;
    double *val;
    fprintf(stdout, "Matrix file name: %s ... ", matrixName);
    t0 = ReadTSC();
    ret = mm_read_mtx_crd(matrixName, &m, &n, &nnz, &row_ind, &col_ind, &val,
                          &matcode);
    check_mm_ret(ret);
    // expand sparse matrix if symmetric
    if(is_symmetric) {
        expand_symmetry(m, n, &nnz, &row_ind, &col_ind, &val);
    }
    timer[LOAD_TIME] += ElapsedTime(ReadTSC() - t0);


    // Convert co-ordinate format to CSR format
    fprintf(stdout, "Converting COO to CSR...");
    unsigned int* csr_row_ptr = NULL;
    unsigned int* csr_col_ind = NULL;
    double* csr_vals = NULL;
    t0 = ReadTSC();
    convert_coo_to_csr(row_ind, col_ind, val, m, n, nnz,
                       &csr_row_ptr, &csr_col_ind, &csr_vals);
    timer[CONVERT_TIME] += ElapsedTime(ReadTSC() - t0);
    fprintf(stdout, "done\n");


    // Load the input vector b (Ax = b)
    char vectorName[MAX_FILENAME];
    strcpy(vectorName, argv[2]);
    fprintf(stdout, "Vector file name: %s ... ", vectorName);
    double* b;
    unsigned int vector_size;
    t0 = ReadTSC();
    read_vector(vectorName, &b, &vector_size);
    timer[LOAD_TIME] += ElapsedTime(ReadTSC() - t0);
    assert(m == vector_size);
    fprintf(stdout, "file loaded\n");


    // Execute CG
    fprintf(stdout, "Executing Conjugate Gradient ... ");
    double* x = (double*) malloc(sizeof(double) * n);;
    assert(x);
    init_vector(x, n, seed);
    t0 = ReadTSC();
    cg(csr_row_ptr, csr_col_ind, csr_vals, m, n, nnz, x, b, max_iter, tol);
    timer[CG_TIME] += ElapsedTime(ReadTSC() - t0);
    fprintf(stdout, "done\n");


    // check answer
    verify_cg(csr_row_ptr, csr_col_ind, csr_vals, m, n, nnz, x, b);

    // Store the calculated answer in a file, one element per line.
    char resName[MAX_FILENAME];
    strcpy(resName, argv[3]);
    fprintf(stdout, "Result file name: %s ... ", resName);
    t0 = ReadTSC();
    store_result(resName, x, m);
    timer[STORE_TIME] += ElapsedTime(ReadTSC() - t0);
    fprintf(stdout, "file saved\n");


    // print timer
    print_time(timer);


    // Free memory
    free(csr_row_ptr);
    free(csr_col_ind);
    free(csr_vals);
    free(b);
    free(x);
    free(row_ind);
    free(col_ind);
    free(val);

    return 0;
}


/* This function checks the number of input parameters to the program to make
   sure it is correct. If the number of input parameters is incorrect, it
   prints out a message on how to properly use the program.
   input parameters:
       int    argc
       char** argv
   return parameters:
       none
 */
void usage(int argc, char** argv)
{
    if(argc < 7) {
        fprintf(stderr, "usage: %s <mat> <vec> <res> <max iter> <tol> <seed>\n",
                argv[0]);
        exit(EXIT_FAILURE);
    }
}

/* This function prints out information about a sparse matrix
   input parameters:
       char*       fileName    name of the sparse matrix file
       MM_typecode matcode     matrix information
       int         m           # of rows
       int         n           # of columns
       int         nnz         # of non-zeros
   return paramters:
       none
 */
void print_matrix_info(char* fileName, MM_typecode matcode,
                       int m, int n, int nnz)
{
    fprintf(stdout, "-----------------------------------------------------\n");
    fprintf(stdout, "Matrix name:     %s\n", fileName);
    fprintf(stdout, "Matrix size:     %d x %d => %d\n", m, n, nnz);
    fprintf(stdout, "-----------------------------------------------------\n");
    fprintf(stdout, "Is matrix:       %d\n", mm_is_matrix(matcode));
    fprintf(stdout, "Is sparse:       %d\n", mm_is_sparse(matcode));
    fprintf(stdout, "-----------------------------------------------------\n");
    fprintf(stdout, "Is complex:      %d\n", mm_is_complex(matcode));
    fprintf(stdout, "Is real:         %d\n", mm_is_real(matcode));
    fprintf(stdout, "Is integer:      %d\n", mm_is_integer(matcode));
    fprintf(stdout, "Is pattern only: %d\n", mm_is_pattern(matcode));
    fprintf(stdout, "-----------------------------------------------------\n");
    fprintf(stdout, "Is general:      %d\n", mm_is_general(matcode));
    fprintf(stdout, "Is symmetric:    %d\n", mm_is_symmetric(matcode));
    fprintf(stdout, "Is skewed:       %d\n", mm_is_skew(matcode));
    fprintf(stdout, "Is hermitian:    %d\n", mm_is_hermitian(matcode));
    fprintf(stdout, "-----------------------------------------------------\n");

}


/* This function checks the return value from the matrix read function,
   mm_read_mtx_crd(), and provides descriptive information.
   input parameters:
       int ret    return value from the mm_read_mtx_crd() function
   return paramters:
       none
 */
void check_mm_ret(int ret)
{
    switch(ret)
    {
        case MM_COULD_NOT_READ_FILE:
            fprintf(stderr, "Error reading file.\n");
            exit(EXIT_FAILURE);
            break;
        case MM_PREMATURE_EOF:
            fprintf(stderr, "Premature EOF (not enough values in a line).\n");
            exit(EXIT_FAILURE);
            break;
        case MM_NOT_MTX:
            fprintf(stderr, "Not Matrix Market format.\n");
            exit(EXIT_FAILURE);
            break;
        case MM_NO_HEADER:
            fprintf(stderr, "No header information.\n");
            exit(EXIT_FAILURE);
            break;
        case MM_UNSUPPORTED_TYPE:
            fprintf(stderr, "Unsupported type (not a matrix).\n");
            exit(EXIT_FAILURE);
            break;
        case MM_LINE_TOO_LONG:
            fprintf(stderr, "Too many values in a line.\n");
            exit(EXIT_FAILURE);
            break;
        case MM_COULD_NOT_WRITE_FILE:
            fprintf(stderr, "Error writing to a file.\n");
            exit(EXIT_FAILURE);
            break;
        case 0:
            fprintf(stdout, "file loaded.\n");
            break;
        default:
            fprintf(stdout, "Error - should not be here.\n");
            exit(EXIT_FAILURE);
            break;

    }
}

/* This function reads information about a sparse matrix using the
   mm_read_banner() function and printsout information using the
   print_matrix_info() function.
   input parameters:
       char*       fileName    name of the sparse matrix file
   return paramters:
       none
 */
void read_info(char* fileName, int* is_sym)
{
    FILE* fp;
    MM_typecode matcode;
    int m;
    int n;
    int nnz;

    if((fp = fopen(fileName, "r")) == NULL) {
        fprintf(stderr, "Error opening file: %s\n", fileName);
        exit(EXIT_FAILURE);
    }

    if(mm_read_banner(fp, &matcode) != 0)
    {
        fprintf(stderr, "Error processing Matrix Market banner.\n");
        exit(EXIT_FAILURE);
    }

    if(mm_read_mtx_crd_size(fp, &m, &n, &nnz) != 0) {
        fprintf(stderr, "Error reading size.\n");
        exit(EXIT_FAILURE);
    }

    print_matrix_info(fileName, matcode, m, n, nnz);
    *is_sym = mm_is_symmetric(matcode);

    fclose(fp);
}

/* This function converts a sparse matrix stored in COO format to CSR format.
   input parameters:
       int*	row_ind		list or row indices (per non-zero)
       int*	col_ind		list or col indices (per non-zero)
       double*	val		list or values  (per non-zero)
       int	m		# of rows
       int	n		# of columns
       int	n		# of non-zeros
   output parameters:
       unsigned int** 	csr_row_ptr	pointer to row pointers (per row)
       unsigned int** 	csr_col_ind	pointer to column indices (per non-zero)
       double** 	csr_vals	pointer to values (per non-zero)
   return paramters:
       none
 */
void convert_coo_to_csr(int* row_ind, int* col_ind, double* val,
                        int m, int n, int nnz,
                        unsigned int** csr_row_ptr, unsigned int** csr_col_ind,
                        double** csr_vals)

{
    // Temporary pointers
    unsigned int* row_ptr_;
    unsigned int* col_ind_;
    double* vals_;

    // We now how large the data structures should be
    // csr_row_ptr -> m + 1
    // csr_col_ind -> nnz
    // csr_vals    -> nnz
    row_ptr_ = (unsigned int*) malloc(sizeof(unsigned int) * (m + 1));
    assert(row_ptr_);
    col_ind_ = (unsigned int*) malloc(sizeof(unsigned int) * nnz);
    assert(col_ind_);
    vals_ = (double*) malloc(sizeof(double) * nnz);
    assert(vals_);

    // Now determine how many non-zero elements are in each row
    // Use a histogram to do this
    unsigned int* buckets = malloc(sizeof(unsigned int) * m);
    assert(buckets);
    memset(buckets, 0, sizeof(unsigned int) * m);

    for(unsigned int i = 0; i < nnz; i++) {
        // row_ind[i] - 1 because index in mtx format starts from 1 (not 0)
        buckets[row_ind[i] - 1]++;
    }

    // Now use a cumulative sum to determine the starting position of each
    // row in csr_col_ind and csr_vals - this information is also what is
    // stored in csr_row_ptr
    for(unsigned int i = 1; i < m; i++) {
        buckets[i] = buckets[i] + buckets[i - 1];
    }
    // Copy this to csr_row_ptr
    row_ptr_[0] = 0;
    for(unsigned int i = 0; i < m; i++) {
        row_ptr_[i + 1] = buckets[i];
    }

    // We can use row_ptr_ to copy the column indices and vals to the
    // correct positions in csr_col_ind and csr_vals
    unsigned int* tmp_row_ptr = (unsigned int*) malloc(sizeof(unsigned int) *
                                                       m);
    assert(tmp_row_ptr);
    memcpy(tmp_row_ptr, row_ptr_, sizeof(unsigned int) * m);

    // Now go through each non-zero and copy it to its appropriate position
    for(unsigned int i = 0; i < nnz; i++)  {
        col_ind_[tmp_row_ptr[row_ind[i] - 1]] = col_ind[i] - 1;
        vals_[tmp_row_ptr[row_ind[i] - 1]] = val[i];
        tmp_row_ptr[row_ind[i] - 1]++;
    }

    // Copy the memory address to the input parameters
    *csr_row_ptr = row_ptr_;
    *csr_col_ind = col_ind_;
    *csr_vals = vals_;

    // Free memory
    free(tmp_row_ptr);
    free(buckets);
}

/* Reads in a vector from file.
   input parameters:
       char*	fileName	name of the file containing the vector
   output parameters:
       double**	vector		pointer to the vector
       int*	vecSize 	pointer to # elements in the vector
   return parameters:
       none
 */
void read_vector(char* fileName, double** vector, int* vecSize)
{
    FILE* fp = fopen(fileName, "r");
    assert(fp);
    char line[MAX_NUM_LENGTH];
    fgets(line, MAX_NUM_LENGTH, fp);
    fclose(fp);

    unsigned int vector_size = atoi(line);
    double* vector_ = (double*) malloc(sizeof(double) * vector_size);

    fp = fopen(fileName, "r");
    assert(fp);
    // first read the first line to get the # elements
    fgets(line, MAX_NUM_LENGTH, fp);

    unsigned int index = 0;
    while(fgets(line, MAX_NUM_LENGTH, fp) != NULL) {
        vector_[index] = atof(line);
        index++;
    }

    fclose(fp);
    assert(index == vector_size);

    *vector = vector_;
    *vecSize = vector_size;
}


/* SpMV function for CSR stored sparse matrix
 */
void spmv(unsigned int* csr_row_ptr, unsigned int* csr_col_ind,
          double* csr_vals, int m, int n, int nnz,
          double* vector_x, double *res)
{
    int i, j;

    #pragma omp parallel for private(i, j) schedule(static)
    for(i = 0; i < m; i++) {
        res[i] = 0.0;

        for(j = csr_row_ptr[i]; j < csr_row_ptr[i+1]; j++) {
            res[i] += csr_vals[j] * vector_x[csr_col_ind[j]];
        }
    }
}


/* Save result vector in a file
 */
void store_result(char *fileName, double* res, int m)
{
    FILE* fp = fopen(fileName, "w");
    assert(fp);

    fprintf(fp, "%d\n", m);
    for(int i = 0; i < m; i++) {
        fprintf(fp, "%0.20f\n", res[i]);
    }

    fclose(fp);
}

/* Print timing information
 */
void print_time(double timer[])
{
    fprintf(stdout, "Module\t\tTime\n");
    fprintf(stdout, "Load\t\t");
    fprintf(stdout, "%f\n", timer[LOAD_TIME]);
    fprintf(stdout, "Convert\t\t");
    fprintf(stdout, "%f\n", timer[CONVERT_TIME]);
    fprintf(stdout, "CG\t\t");
    fprintf(stdout, "%f\n", timer[CG_TIME]);
    fprintf(stdout, "Store\t\t");
    fprintf(stdout, "%f\n", timer[STORE_TIME]);
}

void expand_symmetry(int m, int n, int* nnz_, int** row_ind, int** col_ind,
                     double** val)
{
    fprintf(stdout, "Expanding symmetric matrix ... ");
    int nnz = *nnz_;

    // first, count off-diagonal non-zeros
    int not_diag = 0;
    for(int i = 0; i < nnz; i++) {
        if((*row_ind)[i] != (*col_ind)[i]) {
            not_diag++;
        }
    }

    int* _row_ind = (int*) malloc(sizeof(int) * (nnz + not_diag));
    assert(_row_ind);
    int* _col_ind = (int*) malloc(sizeof(int) * (nnz + not_diag));
    assert(_col_ind);
    double* _val = (double*) malloc(sizeof(double) * (nnz + not_diag));
    assert(_val);

    memcpy(_row_ind, *row_ind, sizeof(int) * nnz);
    memcpy(_col_ind, *col_ind, sizeof(int) * nnz);
    memcpy(_val, *val, sizeof(double) * nnz);
    int index = nnz;
    for(int i = 0; i < nnz; i++) {
        if((*row_ind)[i] != (*col_ind)[i]) {
            _row_ind[index] = (*col_ind)[i];
            _col_ind[index] = (*row_ind)[i];
            _val[index] = (*val)[i];
            index++;
        }
    }
    assert(index == (nnz + not_diag));

    free(*row_ind);
    free(*col_ind);
    free(*val);

    *row_ind = _row_ind;
    *col_ind = _col_ind;
    *val = _val;
    *nnz_ = nnz + not_diag;

    fprintf(stdout, "done\n");
    fprintf(stdout, "  Total # of non-zeros is %d\n", nnz + not_diag);
}

void init_locks(omp_lock_t** locks, int m)
{
    omp_lock_t* _locks = (omp_lock_t*) malloc(sizeof(omp_lock_t) * m);
    assert(_locks);
    for(int i = 0; i < m; i++) {
        omp_init_lock(&(_locks[i]));
    }
    *locks = _locks;
}

void destroy_locks(omp_lock_t* locks, int m)
{
    assert(locks);
    for(int i = 0; i < m; i++) {
        omp_destroy_lock(&(locks[i]));
    }
    free(locks);
}

void init_vector(double* a, int m, unsigned int seed)
{
    assert(a);
    if(seed == 0) {
        memset(a, 0, sizeof(double) * m);
    } else {
        srand(seed);
        for(int i = 0; i < m; i++) {
            a[i] = (1.0 * rand()) / RAND_MAX;
        }
    }
}

// dot product between two vectors x and y
// the result is returned by the function
//
// n is the total number of elements in the vectors
// incx and incy indicates strided access into x and y respectively.
// i.e., if incx is 2, then the values are stored at location 0, 2, 4, etc.
// that is, d += x[i * incx] * y[i * incy]
double ddot(const int n, double* x, const int incx, double* y, const int incy)
{
    // COMPLETE THIS FUNCTION
    double sum = 0.0;

    int i = 0;

    #pragma omp parallel for private(i) reduction(+:sum)
    for (i = 0; i < n; i++) {
        sum += x[i * incx] * y[i * incy];
    }

    return sum;
}


// Calculate the L2-norm for a vector
// L2 norm is calculated by sqrt(x^T * x), where x^T indicates the
//   transpose of the vector x
double dnrm2(const int n, double* x, const int incx)
{
    // COMPLETE THIS FUNCTION
    double nrm = 0.0;

    nrm = sqrt(ddot(n, x, incx, x, incx));

    return nrm;
}


// add two vectors
// z = a * x + y
//
// n is the total number of elements in the vectors
// incx and incy indicates strided access into x and y respectively.
// i.e., if incx is 2, then the values are stored at location 0, 2, 4, etc.
// that is, z[i * incy] = a * x[i * incx] + y[i * incy]
void vec_add(const int n, const double a, const double* x, const int incx,
             double* y, const int incy, double* z)
{
    // COMPLETE THIS FUNCTION

    #pragma omp parallel for
    for (int i = 0; i < n; i++)
        z[i * incy] = a * x[i * incx] + y[i * incy];
}


int cg(unsigned int* csr_row_ptr, unsigned int* csr_col_ind,
       double* csr_vals, int m, int n, int nnz,
       double* x, double* b, int max_iter, double tol)
{
    // COMPLETE THIS FUNCTION

    // set up the workspace
    double* rk = (double*) malloc(sizeof(double) * m);
    assert(rk);
    double* pk = (double*) malloc(sizeof(double) * m);
    assert(pk);
    double* ap = (double*) malloc(sizeof(double) * m);
    assert(ap);
    double residual = 0.0;


    // r0 = b - Ax
    // some code needs to go here

    spmv(csr_row_ptr, csr_col_ind, csr_vals, m, n, nnz, x, ap);
    vec_add(m, -1.0, ap, 1, b, 1, rk);

    // if r0 is sufficiently small, return x0 as the result
    // some code needs to go here

    residual = dnrm2(m, rk, 1);

    if(residual < tol) {
        fprintf(stdout, "\tInput is the solution\n");
        return 0;
    } else {
        fprintf(stdout, "\n\tInitial residual is %f\n", residual);
    }

    // p0 = r0
    memcpy(pk, rk, sizeof(double) * m);

    int k = 0;
    double residual_new = 0.0;
    // repeat until convergence of max iterations has been reached
    for(int i = 0; i < max_iter; i++) {
      	double rr_old = ddot(m, rk, 1, rk, 1);

        // A * p
        // some code needs to go here

        spmv(csr_row_ptr, csr_col_ind, csr_vals, m, n, nnz, pk, ap);

        // d = p^T * A * p
        // some code needs to go here

        double d = ddot(m, pk, 1, ap, 1);

        // alpha = (r^t * r) / d;
        // some code needs to go here

        double alpha = ddot(m, rk, 1, rk, 1) / d;


        // xk = xk + alpha * pk
        // some code needs to go here

        vec_add(m, alpha, pk, 1, x, 1, x);

        // rk = rk - alpha * A*p
        // some code needs to go here

        vec_add(m, -alpha, ap, 1, rk, 1, rk);


/*
        #pragma omp parallel
        #pragma omp single
        {
            #pragma omp task
            {
                // xk = xk + alpha * pk
                // some code needs to go here

                vec_add(m, alpha, pk, 1, x, 1, x);
            }

            #pragma omp task
            {
                // rk = rk - alpha * A*p
                // some code needs to go here

                vec_add(m, -alpha, ap, 1, rk, 1, rk);
            }

            #pragma omp taskwait
        }
*/

        // r^t * r
        // some code needs to go here

        residual_new = dnrm2(m, rk, 1);

        if(residual_new < tol) {
            fprintf(stdout, "\tSolution calculated. Final residual: %f\n",
                    residual_new);
            break;
        } else {
            //fprintf(stdout, "\tIt: %d\tresidual: %f\n", k, residual_new);
        }

        // beta = (r^t * r) / residual
        // some code needs to go here

        // p = r + beta * p
        // some code needs to go here

        double beta = ddot(m, rk, 1, rk, 1) / rr_old;

        vec_add(m, beta, pk, 1, rk, 1, pk);

        residual = residual_new;
        k++;
    }

    free(rk);
    free(pk);
    free(ap);
}

void verify_cg(int* row_ptr, int* col_ind, double* vals, int m, int n, int nnz,
               double* x, double* b)
{
    fprintf(stdout, "Verifying answer by calculating A * x ... \n");
    double* ap = (double*) malloc(sizeof(double) * m);
    assert(ap);

    // calculate A * x
    spmv(row_ptr, col_ind, vals, m, n, nnz, x, ap);

    // calculate b - A * x
    vec_add(m, -1.0, ap, 1, b, 1, ap);

    double norm = dnrm2(m, ap, 1);
    fprintf(stdout, "\tnorm of (b - A * x): %e\n", norm);
}
