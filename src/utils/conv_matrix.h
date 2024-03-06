#pragma once

#include "../course-project-second-year/types/matrix.h"

namespace convolution_svd {

using namespace::svd_computation;

template<typename Type>
Matrix<Type> correlation_conv(Matrix<Type> kernel, size_t image_height, size_t image_width) {
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
}