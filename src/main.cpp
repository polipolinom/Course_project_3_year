#include <iostream>
#include "algorithms/householder_reflection.h"
#include "algorithms/bidiagonalization.h"

using namespace convolution_svd::details;

int main() {
    Matrix<long double> A({{1, 2, 0, 3, 4, 0}, 
                          {0, 1, 2, 0, 3, 4}, 
                          {0, 0, 1, 0, 0, 3}, 
                          {5, 6, 0, 1, 2, 0}, 
                          {0, 5, 6, 0, 1, 2},
                          {0, 0, 5, 0, 0, 1}});
    conv_block_bidiagonalization(A, 3, 3);
    std::cout << A;
    return 0;
}