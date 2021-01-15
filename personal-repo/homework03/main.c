#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <omp.h>
#include "main.h"

#define MAX_FILENAME 256
#define MAX_NUM_LENGTH 100
#define MAX_ITER 100

#define NUM_TIMERS       6
#define LOAD_TIME        0
#define CONVERT_TIME     1
#define LOCK_INIT_TIME   2
#define SPMV_COO_TIME    3
#define SPMV_CSR_TIME    4
#define STORE_TIME       5

#define THREADS 16

// store each [row, column, value] as a tuple
typedef struct COO {
	int row;
	int column;
	double value;
} COO;

//helper
int compare(const void *a,const void *b){
	COO c = *(COO*)a;
	COO d = *(COO*)b;
	return c.row - d.row;
}

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


	// Read the sparse matrix file name
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


	// load and expand sparse matrix from file (if symmetric)
	fprintf(stdout, "Matrix file name: %s ... ", matrixName);
	t0 = ReadTSC();
	ret = mm_read_mtx_crd(matrixName, &m, &n, &nnz, &row_ind, &col_ind, &val,
	                      &matcode);
	check_mm_ret(ret);
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
	// IMPLEMENT THIS FUNCTION - MAKE SURE IT's **NOT** O(N^2)
	convert_coo_to_csr(row_ind, col_ind, val, m, n, nnz,
	                   &csr_row_ptr, &csr_col_ind, &csr_vals);
	timer[CONVERT_TIME] += ElapsedTime(ReadTSC() - t0);
	fprintf(stdout, "done\n");


	// Load the input vector file
	char vectorName[MAX_FILENAME];
	strcpy(vectorName, argv[2]);
	fprintf(stdout, "Vector file name: %s ... ", vectorName);
	double* vector_x;
	unsigned int vector_size;
	t0 = ReadTSC();
	read_vector(vectorName, &vector_x, &vector_size);
	timer[LOAD_TIME] += ElapsedTime(ReadTSC() - t0);
	assert(n == vector_size);
	fprintf(stdout, "file loaded\n");


	// Calculate COO SpMV
	// first set up some locks
	t0 = ReadTSC();
	omp_lock_t* writelock;
	init_locks(&writelock, m);
	timer[LOCK_INIT_TIME] += ElapsedTime(ReadTSC() - t0);

	double *res_coo = (double*) malloc(sizeof(double) * m);;
	assert(res_coo);
	fprintf(stdout, "Calculating COO SpMV ... ");
	t0 = ReadTSC();
	for(unsigned int i = 0; i < MAX_ITER; i++) {
		// IMPLEMENT THIS FUNCTION - MAKE SURE IT'S PARALLELIZED
		spmv_coo(row_ind, col_ind, val, m, n, nnz, vector_x, res_coo,
		         writelock);
	}
	timer[SPMV_COO_TIME] += ElapsedTime(ReadTSC() - t0);
	fprintf(stdout, "done\n");



	// Calculate CSR SpMV
	double *res_csr = (double*) malloc(sizeof(double) * m);;
	assert(res_csr);
	fprintf(stdout, "Calculating CSR SpMV ... ");
	t0 = ReadTSC();
	for(unsigned int i = 0; i < MAX_ITER; i++) {
		// IMPLEMENT THIS FUNCTION - MAKE SURE IT'S PARALLELIZED
		spmv(csr_row_ptr, csr_col_ind, csr_vals, m, n, nnz, vector_x, res_csr);
	}
	timer[SPMV_CSR_TIME] += ElapsedTime(ReadTSC() - t0);
	fprintf(stdout, "done\n");


	// Store the calculated vector in a file, one element per line.
	char resName[MAX_FILENAME];
	strcpy(resName, argv[3]);
	fprintf(stdout, "Result file name: %s ... ", resName);
	t0 = ReadTSC();
store_result(resName, res_csr, m);
//store_result(resName, res_coo, m);
	timer[STORE_TIME] += ElapsedTime(ReadTSC() - t0);
	fprintf(stdout, "file saved\n");


	// print timer
	print_time(timer);


	// Free memory
	free(csr_row_ptr);
	free(csr_col_ind);
	free(csr_vals);
	free(vector_x);
	free(res_coo);
	free(res_csr);
	free(row_ind);
	free(col_ind);
	free(val);
	destroy_locks(writelock, m);

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
	if(argc < 4) {
		fprintf(stderr, "usage: %s <matrix> <vector> <result>\n", argv[0]);
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
	//declare for sorting non-zero values by rows
	struct COO* coo_triple = (struct COO*) malloc(sizeof(struct COO) * nnz);

	for(int i = 0; i < nnz; i++)
	{
		coo_triple[i].row = row_ind[i];
		coo_triple[i].column = col_ind[i];
		coo_triple[i].value = val[i];
	}

	//sort by rows
	qsort(coo_triple, nnz, sizeof(coo_triple[0]), compare);

	*csr_row_ptr = (unsigned int*) malloc(sizeof(int) * (m + 1));
	*csr_col_ind = (unsigned int*) malloc(sizeof(int) * nnz);
	*csr_vals =  (double*) malloc(sizeof(double) * nnz);

	//put back after sorted for both COO and CSR
	#pragma omp parallel for num_threads(THREADS) schedule(guided)
	for(int i = 0; i < nnz; i++)
	{
		//COO with pre-sort by rows
		row_ind[i] = coo_triple[i].row;
		col_ind[i] = coo_triple[i].column;
		val[i] = coo_triple[i].value;

		//CSR needs to be pre-sorted in order to calculate row pointers
		(*csr_col_ind)[i] = coo_triple[i].column;
		(*csr_vals)[i] = coo_triple[i].value;
	}

	//initialize row pointers
	for(int i = 0; i < m + 1; i++)
		(*csr_row_ptr)[i] = 0;

	//count how many non-zeros in each row for row pointers
	for(int i = 0; i < nnz; i++)
	{
		int index = row_ind[i] - 1;
		(*csr_row_ptr)[index] += 1;
	}

	//add in pre-fix sum fashion for row pointers
	int curr_sum = 0;
	for(int i = 0; i < m; i++)
	{
		int temp = (*csr_row_ptr)[i];
		(*csr_row_ptr)[i] = curr_sum;
		curr_sum += temp;
	}

	//last element in row pointer is the number of none-zeros
	(*csr_row_ptr)[m] = nnz;

	free(coo_triple);
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

/* SpMV function for COO stored sparse matrix
 */
void spmv_coo(unsigned int* row_ind, unsigned int* col_ind, double* vals,
              int m, int n, int nnz, double* vector_x, double *res,
              omp_lock_t* writelock)
{
	int i;

	#pragma omp parallel num_threads(THREADS)
	{
		#pragma omp for private(i) schedule(static)
		for(i = 0; i < m; i++) {
			res[i] = 0.0;
		}

		//need locks if COO is not pre-sorted by rows
		#pragma omp for private(i) schedule(static)
		for(i = 0; i < nnz; i++) {
			//omp_set_lock(&(writelock[row_ind[i]-1]));
			res[row_ind[i] - 1] += vals[i] * vector_x[col_ind[i] - 1];
			//omp_unset_lock(&(writelock[row_ind[i]-1]));
		}
	}
}



/* SpMV function for CSR stored sparse matrix
 */
void spmv(unsigned int* csr_row_ptr, unsigned int* csr_col_ind,
          double* csr_vals, int m, int n, int nnz,
          double* vector_x, double *res)
{
	int i, j;
	#pragma omp parallel num_threads(THREADS)
	{
		#pragma omp for private(i, j) schedule(static)
		for(i = 0; i < m; i++) {
			res[i] = 0.0;

			for(j = csr_row_ptr[i]; j < csr_row_ptr[i+1]; j++) {
				res[i] += csr_vals[j] * vector_x[csr_col_ind[j] - 1];
			}
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
		fprintf(fp, "%0.10f\n", res[i]);
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
	fprintf(stdout, "Lock Init\t");
	fprintf(stdout, "%f\n", timer[LOCK_INIT_TIME]);
	fprintf(stdout, "COO SpMV\t");
	fprintf(stdout, "%f\n", timer[SPMV_COO_TIME]);
	fprintf(stdout, "CSR SpMV\t");
	fprintf(stdout, "%f\n", timer[SPMV_CSR_TIME]);
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
