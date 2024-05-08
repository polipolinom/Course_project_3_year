#pragma once

#include "../algorithms/constants.h"
#include "../course-project-second-year/types/matrix.h"

namespace convolution_svd {
namespace details {
using namespace svd_computation;

template <typename Type>
bool is_diagonal(const Matrix<Type>& A, const long double eps = constants::DEFAULT_EPSILON) {
    if (A.height() != A.width()) {
        return false;
    }
    for (size_t i = 0; i < A.height(); ++i) {
        for (size_t j = 0; j < A.width(); ++j) {
            if (i != j && abs(A(i, j)) > eps) {
                return false;
            }
        }
    }
    return true;
}

template <typename Type>
bool is_diagonal_banded(const Matrix<Type>& A, const size_t band_size, size_t row_start, size_t column_start,
                        size_t row_end, size_t column_end, const long double eps = constants::DEFAULT_EPSILON) {
    for (size_t i = row_start; i <= row_end; ++i) {
        for (size_t j = i + 1; j < std::min(column_end + 1, i + band_size); ++j) {
            if (std::abs(A(i, j)) > eps) {
                return false;
            }
        }
    }
    return true;
}

template <typename Type>
bool is_zero(const Matrix<Type>& A, const long double eps = constants::DEFAULT_EPSILON) {
    for (int i = 0; i < A.height(); ++i) {
        for (int j = 0; j < A.width(); ++j) {
            if (abs(A(i, j)) > eps) {
                return false;
            }
        }
    }
    return true;
}

template <typename Type>
bool check_zeros(const std::vector<Type>& v, size_t first, size_t last,
                 const long double eps = constants::DEFAULT_EPSILON) {
    for (size_t ind = first; ind < last; ++ind) {
        if (std::abs(v[ind]) > eps) {
            return false;
        }
    }
    return true;
}

template <typename Type>
bool is_unitary(const Matrix<Type>& A, const long double eps = constants::DEFAULT_EPSILON) {
    if (A.height() != A.width()) {
        return false;
    }

    Matrix<Type> B = A * conjugate(A) - Matrix<Type>::identity(A.height());
    if (!is_zero(B)) {
        return false;
    }

    return true;
}
}  // namespace details
}  // namespace convolution_svd
