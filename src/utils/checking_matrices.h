#pragma once

#include "../algorithms/constants.h"
#include "../course-project-second-year/types/matrix.h"

namespace convolution_svd{
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
bool is_diagonal_banded(const Matrix<Type>& A, const size_t band_size, const long double eps = constants::DEFAULT_EPSILON) {
    if (A.height() != A.width()) {
        return false;
    }
    for (size_t i = 0; i < A.height(); ++i) {
        for (size_t j = 0; j < std::min(A.width(), i + band_size); ++j) {
            if (i != j && abs(A(i, j)) > eps) {
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
}  // namespace svd_computation
