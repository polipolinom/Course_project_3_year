#pragma once

#include <cmath>

#include "../course-project-second-year/types/matrix.h"
#include "../course-project-second-year/types/vector.h"
#include "svd_convolution.h"

namespace convolution_svd {
using namespace svd_computation;
std::vector<std::vector<Matrix<long double>>> clip_singular_1d(std::vector<std::vector<Matrix<long double>>> kernels,
                                                               const size_t signal_size, const size_t stride = 1,
                                                               const long double lower_bound = 0,
                                                               const long double upper_bound = 1,
                                                               const long double eps = constants::DEFAULT_EPSILON) {
    assert(signal_size > 0);
    size_t C_in = kernels.size();
    assert(C_in > 0);
    size_t C_out = kernels[0].size();
    assert(C_out > 0);
    size_t kernel_size = kernels[0][0].width();

    Matrix<long double> left_basis;
    Matrix<long double> right_basis;
    auto values = svd_convolution_1d(kernels, signal_size, &left_basis, &right_basis, stride, false, eps);
    for (auto& value : values) {
        value = std::min(value, upper_bound);
        value = std::max(value, lower_bound);
    }

    auto matrix_kernels = left_basis *
                          Matrix<long double>::diagonal(values, C_out * ((signal_size - kernel_size + stride) / stride),
                                                        C_in * signal_size) *
                          right_basis;
    std::vector<std::vector<Matrix<long double>>> new_kernels(C_in, std::vector<Matrix<long double>>(C_out));

    for (size_t i = 0; i < C_in; ++i) {
        for (size_t j = 0; j < C_out; ++j) {
            Matrix<long double> kernel(1, kernel_size);
            for (size_t k = 0; k < kernel_size; ++k) {
                kernel(0, k) = matrix_kernels(j, C_in * k + i);
            }
            new_kernels[i][j] = kernel;
        }
    }

    return new_kernels;
}

std::vector<std::vector<Matrix<long double>>> clip_singular_2d(std::vector<std::vector<Matrix<long double>>> kernels,
                                                               const size_t image_height, const size_t image_width, 
                                                               const size_t stride = 1,
                                                               const long double lower_bound = 0,
                                                               const long double upper_bound = 1,
                                                               const long double eps = constants::DEFAULT_EPSILON) {
    assert(image_height > 0);
    assert(image_width > 0);
    size_t C_in = kernels.size();
    assert(C_in > 0);
    size_t C_out = kernels[0].size();
    assert(C_out > 0);
    size_t kernel_height  = kernels[0][0].height();
    size_t kernel_width =  kernels[0][0].width();

    Matrix<long double> left_basis;
    Matrix<long double> right_basis;
    auto values = svd_convolution_2d(kernels, image_height, image_width, &left_basis, &right_basis, stride, false, eps);
    for (auto& value : values) {
        value = std::min(value, upper_bound);
        value = std::max(value, lower_bound);
    }

    auto matrix_kernels = left_basis *
                          Matrix<long double>::diagonal(values, C_out * ((image_height - kernel_height + stride) / stride) * ((image_width - kernel_width + stride) / stride),
                                                        C_in * image_height * image_width) *
                          right_basis;

    std::vector<std::vector<Matrix<long double>>> new_kernels(C_in, std::vector<Matrix<long double>>(C_out));
    for (size_t i = 0; i < C_in; ++i) {
        for (size_t j = 0; j < C_out; ++j) {
            Matrix<long double> kernel(kernel_height, kernel_width);
            for (size_t q = 0; q < kernel_width; ++q) {
                for (size_t p = 0; p < kernel_height; ++p) {
                    kernel(p, q) = matrix_kernels(j, q * C_in * image_height + p * C_in + i);
                }
            }
            new_kernels[i][j] = kernel;
        }
    }
    return new_kernels;
}
}  // namespace convolution_svd