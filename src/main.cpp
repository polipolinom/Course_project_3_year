#include <math.h>

#include <iostream>

#include "../course-project-second-year/algorithms/householder_reflections.h"
#include "../course-project-second-year/types/matrix.h"
#include "algorithms/svd_convolution.h"

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
    auto ans = svd_convolution_1d({{kernel1}, {kernel2}}, 5, &left_basis, &right_basis, true);
    for (auto i : ans) {
        std::cout << i << " ";
    }
    return 0;
}