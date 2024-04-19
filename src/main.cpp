#include <iostream>
#include "algorithms/svd_banded.h"
#include "../course-project-second-year/types/matrix.h"
#include "../course-project-second-year/algorithms/householder_reflections.h"
#include "utils/conv_matrix.h"
#include <math.h>
#include "algorithms/householder_reflection.h"

using namespace convolution_svd;
using namespace svd_computation;
 
int main() { 
    Matrix<long double> kernel = {{2, 6}, {8, 9}, {6, 3}};
    size_t image_height = 5;
    size_t image_width = 5;
    auto A = correlation_conv(kernel, image_height, image_width);
    size_t band_size = A.width() - A.height() + 1;
    std::cout << band_size * band_size * A.height() << std::endl;
    // std::cout << band_size << std::endl;

    std::cout << A << std::endl;
    auto shift = convolution_svd::details::wilkinson_shift(A, band_size);
    auto first = Matrix<long double>(A.row(0) * A(0, 0));
    first(0, 0) -= shift;
    auto v = left_segment_reflection(first, 0, band_size - 1, 0, false);
    convolution_svd::details::mult_right_reflection_banded(A, band_size, v, 0, 0, band_size - 1);
    for (size_t ind = 0; ind < 3; ++ind) {
        auto left_reflector = left_segment_reflection(A, ind, std::min(A.height() - 1, ind + band_size), ind, false);
        convolution_svd::details::mult_left_reflection_banded(A, band_size, left_reflector, ind, std::min(A.height() - 1, ind + band_size), ind);
        if (ind + band_size - 1 < A.width()) {
            auto right_reflector = right_segment_reflection(A, ind, ind + band_size - 1, std::min(ind + band_size - 1 + band_size, A.width() - 1), false);
            convolution_svd::details::mult_right_reflection_banded(A, band_size, right_reflector, ind, ind + band_size - 1, std::min(ind + band_size - 1 + band_size, A.width() - 1));
        }
    }
    convolution_svd::details::set_low_values_zero(A);
    // for (size_t i = 0; i < A.height(); ++i) {
    //     for (size_t j = 0; j < i; ++j) {
    //         assert(A(i, j) == 0 && 1);
    //     }
    //     for (size_t j = i + band_size; j < A.width(); ++j) {
    //         assert(A(i, j) == 0 && 2);
    //     }
    // }
    std::cout << A;
    return 0;
}