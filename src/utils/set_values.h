#pragma once

#include "../algorithms/constants.h"
#include "../course-project-second-year/types/matrix.h"

namespace convolution_svd {
namespace details {
using namespace svd_computation;

template <typename Type>
void set_low_values_zero(Matrix<Type>& A, const long double eps = constants::DEFAULT_EPSILON) {
    for (size_t i = 0; i < A.height(); ++i) {
        for (size_t j = 0; j < A.width(); ++j) {
            if (abs(A(i, j)) <= eps) {  // if A(i, j) == NaN, it remains nan
                A(i, j) = Type(0.0);
            }
        }
    }
}
}  // namespace details
}  // namespace colvolution_svd
