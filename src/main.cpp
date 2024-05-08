#include <math.h>

#include <iostream>

#include "../course-project-second-year/algorithms/householder_reflections.h"
#include "../course-project-second-year/types/matrix.h"
#include "algorithms/regularization.h"
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
    auto kernels = clip_singular_1d({{kernel1, kernel3}, {kernel2, kernel4}}, 5, 0, 100);
    for (auto i : kernels) {
        for (auto j : i) {
            std::cout << j << std::endl;
        }
    }
    return 0;
}