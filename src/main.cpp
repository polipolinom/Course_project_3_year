#include <math.h>

#include <iostream>

#include "../course-project-second-year/algorithms/householder_reflections.h"
#include "../course-project-second-year/types/matrix.h"
#include "algorithms/band_reduction.h"
#include "algorithms/constants.h"
#include "algorithms/householder_reflection.h"
#include "algorithms/svd_banded.h"
#include "utils/conv_matrix.h"

using namespace convolution_svd;
using namespace svd_computation;

int it;

int main() {
    it = 0;
    Matrix<long double> kernel1 = {{-0.00620603, -0.227034, -0.409059, -0.302032, -0.635999}};
    size_t image_height = 1;
    size_t image_width = 20;
    auto A = correlation_conv({{kernel1}}, image_height, image_width, true);
    size_t band_size = 5;
    auto ans = svd_banded_reduction(A, band_size, 1e-16);
    for (auto i : ans) {
        std::cout << i << " ";
    }
    std::cout << std::endl;
    std::cout << it << std::endl;

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