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

    Matrix<long double> kernel1 = {{1, 2, 3}, {4, 5, 6}};
    Matrix<long double> left_basis;
    Matrix<long double> right_basis;
    Matrix<long double> A = correlation_conv({{kernel1}, {kernel1}}, 4, 4);
    for (size_t i = 0; i < A.height(); ++i) {
        std::cout << "[";
        for (size_t j = 0; j < A.width(); ++j) {
            std::cout << A(i, j) << ", ";
        }
        std::cout << "],\n";
    }
    auto values = svd_convolution_2d({{kernel1}, {kernel1}}, 4, 4, &left_basis, &right_basis, 1);
    for (auto i : values) {
        std::cout << i << " ";
    }
    return 0;
}