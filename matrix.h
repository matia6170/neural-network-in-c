// matrix.h

#ifndef MATRIX_H
#define MATRIX_H
#include "util.h"

typedef struct {
    int rows;
    int cols;
    double *data;
} Matrix;

/**
 * Create a new matrix with the given number of rows and columns
 *
 * @param rows The number of rows in the matrix
 * @param cols The number of columns in the matrix
 * @return A pointer to the new matrix
 */
Matrix *matrix_create(int rows, int cols);
void matrix_free(Matrix *M);
void matrix_print(Matrix *M);

void matrix_multiply(Matrix *A, Matrix *B, Matrix *C);
void matrix_add(Matrix *A, Matrix *B, Matrix *C);
void matrix_subtract(Matrix *A, Matrix *B, Matrix *C);
void matrix_scalar_multiply(Matrix *A, double x, Matrix *B);
void matrix_scalar_add(Matrix *A, double x, Matrix *B);


void matrix_elemwise_action(double (*action)(double), Matrix *A, Matrix *B);

void inline static matrix_sigmoid(Matrix *A, Matrix *B) { matrix_elemwise_action(sigmoid, A, B); }

#endif