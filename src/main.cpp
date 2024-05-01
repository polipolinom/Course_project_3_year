#include <math.h>

#include <iostream>

#include "../course-project-second-year/algorithms/householder_reflections.h"
#include "../course-project-second-year/types/matrix.h"
#include "algorithms/band_reduction.h"
#include "algorithms/householder_reflection.h"
#include "algorithms/svd_banded.h"
#include "utils/conv_matrix.h"

using namespace convolution_svd;
using namespace svd_computation;

int main() {
    Matrix<long double> kernel1 = {{1, 2, 3}};
    size_t image_height = 1;
    size_t image_width = 10;
    auto A = correlation_conv({{kernel1}}, image_height, image_width);
    size_t band_size = A.width() - A.height() + 1;
    auto ans = apply_banded_qr(A, band_size, 0, 0, A.height() - 1, A.width() - 1);
    for (auto i : ans) {
        std::cout << i << " ";
    }
    std::cout << std::endl;
    /*std::cout << "[";
    for (size_t i = 0; i < A.height(); ++i) {
        std::cout << "[";
        for (size_t j = 0; j < A.width(); ++j) {
            std::cout << A(i, j) << ", ";
        }
        std::cout << "],\n";
    }
    std::cout << "]";*/
    return 0;
}