#include "matrix.h"

/**
 * Create a new matrix with the given number of rows and columns
 *
 * @param rows The number of rows in the matrix
 * @param cols The number of columns in the matrix
 * @return A pointer to the new matrix
 */
Matrix *matrix_create(int rows, int cols){

}
void matrix_free(Matrix *M);
void matrix_print(Matrix *M);

void matrix_multiply(Matrix *A, Matrix *B, Matrix *C);
void matrix_add(Matrix *A, Matrix *B, Matrix *C);
void matrix_subtract(Matrix *A, Matrix *B, Matrix *C);
void matrix_scalar_multiply(Matrix *A, double x, Matrix *B);
void matrix_scalar_add(Matrix *A, double x, Matrix *B);

void matrix_elemwise_action(double (*action)(double), Matrix *A, Matrix *B);
