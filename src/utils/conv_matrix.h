#pragma once

#include "../course-project-second-year/types/matrix.h"

namespace convolution_svd {

using namespace::svd_computation;

namespace details {
template<typename Type>
void add_kernel_correlation(Matrix<Type>& conv, const Matrix<Type> kernel, 
                            size_t kernel_ind_in, size_t kernel_ind_out,
                            size_t C_in, size_t C_out, 
                            size_t image_height, size_t image_width) {
    assert(kernel_ind_in >= 0 && kernel_ind_in < C_in);
    assert(kernel_ind_out >= 0 && kernel_ind_out < C_out);
    assert(image_height > 0);
    assert(image_width > 0);
    assert(conv.height() == (image_height - kernel.height() + 1) * (image_width - kernel.width() + 1) * C_out);
    assert(conv.width() == image_height * image_width * C_in);

    for (size_t j = 0; j < kernel.height(); ++j) {
        for (size_t i = 0; i < kernel.width(); ++i) {
            for (size_t k = 0; k < image_width - kernel.width() + 1; ++k) {
                for (size_t p = 0; p < image_height - kernel.height() + 1; ++p) {
                    size_t block_row = k * (image_height - kernel.height() + 1) * C_out + p * C_out;
                    size_t block_column = (k + i) * (image_height * C_in) + (p + j) * C_in;

                    conv(block_row + kernel_ind_out, block_column + kernel_ind_in) = kernel(j, i);
                }
            }
        }
    }
}
} 


template<typename Type>
Matrix<Type> correlation_conv(const Matrix<Type>& kernel, size_t image_height, size_t image_width) {
    assert(image_height > 0);
    assert(image_width > 0);

    Matrix<Type> conv((image_height - kernel.height() + 1) * (image_width - kernel.width() + 1), image_height * image_width);
    for (size_t i = 0; i < kernel.width(); ++i) {
        for (size_t j = 0; j < image_width - kernel.width() + 1; ++j) {
            size_t row = j * (image_height - kernel.height() + 1);
            size_t column = (i + j) * image_height;
            for (size_t p = 0; p < image_height - kernel.height() + 1; ++p) {
                for (size_t q = 0; q < kernel.height(); ++q) {
                    conv(row + p, column + p + q) = kernel(q, i);
                }
            } 

        }
    }
    return conv;
}

template<typename Type>
Matrix<Type> correlation_conv(const std::vector<std::vector<Matrix<Type>>>& kernel, size_t image_height, size_t image_width) {
    assert(image_height > 0);
    assert(image_width > 0);

    size_t C_in = kernel.size();
    assert(C_in > 0);
    size_t C_out = kernel[0].size();
    assert(C_out > 0);

    size_t kernel_height = kernel[0][0].height();
    assert(kernel_height > 0);
    size_t kernel_width = kernel[0][0].width();
    assert(kernel_width > 0);

    Matrix<Type> conv((image_height - kernel_height + 1) * (image_width - kernel_width + 1) * C_out, image_height * image_width * C_in);

    for (size_t i = 0; i < C_in; ++i) {
        assert(kernel[i].size() == C_out);
        for (size_t j = 0; j < C_out; ++j) {
            assert(kernel[i][j].height() == kernel_height);
            assert(kernel[i][j].width() == kernel_width);
            details::add_kernel_correlation(conv, kernel[i][j], i, j, C_in, C_out, image_height, image_width);
        }
    }

    return conv;
}

template<typename Type>
Matrix<Type> correlation_conv(const std::initializer_list<std::initializer_list<Matrix<Type>>>& kernel, size_t image_height, size_t image_width) {
    assert(image_height > 0);
    assert(image_width > 0);

    size_t C_in = kernel.size();
    assert(C_in > 0);
    size_t C_out = kernel.begin()[0].size();
    assert(C_out > 0);

    size_t kernel_height = kernel.begin()[0].begin()[0].height();
    assert(kernel_height > 0);
    size_t kernel_width = kernel.begin()[0].begin()[0].width();
    assert(kernel_width > 0);

    Matrix<Type> conv((image_height - kernel_height + 1) * (image_width - kernel_width + 1) * C_out, image_height * image_width * C_in);

    for (size_t i = 0; i < C_in; ++i) {
        assert(kernel.begin()[i].size() == C_out);
        for (size_t j = 0; j < C_out; ++j) {
            assert(kernel.begin()[i].begin()[j].height() == kernel_height);
            assert(kernel.begin()[i].begin()[j].width() == kernel_width);
            details::add_kernel_correlation(conv, kernel.begin()[i].begin()[j], i, j, C_in, C_out, image_height, image_width);
        }
    }

    return conv;
}
}