#include <math.h>

#include <iostream>

#include "../course-project-second-year/algorithms/householder_reflections.h"
#include "../course-project-second-year/types/matrix.h"
#include "algorithms/band_reduction.h"
#include "algorithms/regularization.h"
#include "algorithms/svd_convolution.h"
#include "utils/conv_matrix.h"

using namespace convolution_svd;
using namespace svd_computation;

int it;

int main() {
    it = 0;

    Matrix<long double> kernel1 = {{1, 2, 3}};
    Matrix<long double> kernel2 = {{3, 4, 5}};
    Matrix<long double> kernel3 = {{5, 6, 7}};
    Matrix<long double> kernel4 = {{7, 8, 9}};
    Matrix<long double> left_basis;
    Matrix<long double> right_basis;
    auto values = svd_convolution_1d({{kernel1, kernel3, kernel2}}, 5, &left_basis, &right_basis, 1);
    Matrix<long double> A = correlation_conv({{kernel1, kernel3, kernel2}}, 1, 5);
    for (size_t i = 0; i < A.height(); ++i) {
        std::cout << "[";
        for (size_t j = 0; j < A.width(); ++j) {
            std::cout << A(i, j) << ", ";
        }
        std::cout << "],\n";
    }
    for (auto i : values) {
        std::cout << i << " ";
    }
    return 0;
}