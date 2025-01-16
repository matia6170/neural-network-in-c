#ifndef NETWORK_H
#define NETWORK_H

#include "matrix.h"
typedef struct {
    int num_layers;
    int *sizes;
    Matrix **biases;
    Matrix **weights;
    // int num_biases;
    // int num_weights;
} Network;

Network *network_create(int *sizes, int num_layers);


void network_backpropagate(Network *net, Matrix *input, Matrix *output, Matrix *nabla_b, Matrix *nabla_w);


#endif