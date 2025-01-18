#include "network.h"

#include <stdlib.h>

#include "matrix.h"
#include "util.h"

Network *network_create(int *sizes, int num_layers) {
    Network *net = malloc(sizeof(Network));

    net->num_layers = num_layers;
    net->sizes = sizes;

    return net;
}

void cost_derivative(Matrix *output_activations, Matrix *y, Matrix *result) {
    matrix_subtract(output_activations, y, result);
}

void network_backpropagate(Network *net, Matrix *input, Matrix *output, Matrix ***nabla_b, Matrix ***nabla_w) {
    // set nabla_b and nabla_w to zero

    // nabla_b is a list of 1d matrices
    // each matrix is the change in bias for given index of the layer
    *nabla_b = malloc(sizeof(Matrix) * net->num_layers);
    (*nabla_b)[0] = NULL;
    for (int i = 1; i < net->num_layers; i++) {
        (*nabla_b)[i] = matrix_create(net->sizes[i], 1);
    }

    // nabla_w is a list of 2d matrices
    // each matrix is the change in weight for given index of the layer
    *nabla_w = malloc(sizeof(Matrix) * net->num_layers);
    (*nabla_w)[0] = NULL;
    for (int i = 1; i < net->num_layers; i++) {
        (*nabla_w)[i] = matrix_create(net->sizes[i], net->sizes[i - 1]);
    }

    // feedforward step to get activations

    // activations is a list of 1d matrices
    Matrix **activations = malloc(sizeof(Matrix *) * net->num_layers);
    // Init each activations[i] to an empty matrix with correct size
    for (int i = 0; i < net->num_layers; i++) {
        activations[i] = matrix_create(net->sizes[i], 1);
    }
    activations[0] = input;

    // zs is a list of 1d matrices
    Matrix **zs = malloc(sizeof(Matrix *) * net->num_layers);
    zs[0] = NULL;
    // Init each zs[i] to an empty matrix with correct size
    for (int i = 1; i < net->num_layers; i++) {
        zs[i] = matrix_create(net->sizes[i], 1);
    }

    // calculate activations
    for (int i = 1; i < net->num_layers; i++) {
        // z = w * a (weights[i] is a 2d matrix, activations[i] is a 1d matrix)
        matrix_multiply(net->weights[i], activations[i - 1], zs[i]);
        // z = z + b
        matrix_add(&zs[i], net->biases[i], &zs[i]);

        // a = sigmoid(z)
        matrix_sigmoid(&zs[i], activations[i]);  // activation function
    }

    // calculate output error (delta)
    // backpropagate to get nabla_b and nabla_w

    /*
     * calcualte last layer error
     * delta = (a - y) * sigmoid_prime(z)
     */

    // delta = (a - y)
    Matrix *delta = matrix_create(net->sizes[net->num_layers - 1], 1);
    cost_derivative(activations[net->num_layers - 1], output, delta);
    // temp = sigmoid_prime(zs[last layer])
    Matrix *temp = matrix_create(net->sizes[net->num_layers - 1], 1);
    matrix_elemwise_action(sigmoid_prime, zs[net->num_layers - 1], temp);
    // delta = delta * temp
    matrix_elemwise_mult(delta, temp, delta);

    // nabla_b[last layer] = delta
    matrix_copy(delta, (*nabla_b)[net->num_layers - 1]);

    // nabla_w[last layer] = delta @ activations[last layer - 1].transpose
    matrix_multiply(delta, activations[net->num_layers - 2], (*nabla_w)[net->num_layers - 1]);

    matrix_free(delta);
    matrix_free(temp);

    /*
     * backpropagate the error to previous layers
     */
    for (int i = net->num_layers - 1; i > 0; i++) {
        Matrix *z = zs[i];
        Matrix *delta = matrix_create(net->sizes[i], 1);
        cost_derivative(activations[i], output, delta);

        Matrix *temp = matrix_create(net->sizes[i], 1);
        matrix_elemwise_action(sigmoid_prime, z, temp);
    }

    // update weights and biases

    free(activations);
    free(zs);
}