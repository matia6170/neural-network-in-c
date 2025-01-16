#ifndef UTIL_H
#define UTIL_H

#include <math.h>

inline static double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

inline static double sigmoid_prime(double x) {
    return sigmoid(x) * (1 - sigmoid(x));
}

#endif