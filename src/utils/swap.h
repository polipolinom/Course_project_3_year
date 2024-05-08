#pragma once

#include "../course-project-second-year/types/matrix.h"

namespace convolution_svd {
namespace details {

using namespace ::svd_computation;

template <typename Type>
void swap_rows_basis(Matrix<Type>& A, size_t ind1, size_t ind2, Matrix<Type>* left_basis = nullptr) {
    assert(0 <= ind1 < A.height());
    assert(0 <= ind2 < A.height());

    for (size_t k = 0; k < A.width(); ++k) {
        std::swap(A(ind1, k), A(ind2, k));
    }

    if (left_basis != nullptr) {
        (*left_basis) = Matrix<Type>::identity(A.height());
        (*left_basis)(ind1, ind1) = 0;
        (*left_basis)(ind2, ind2) = 0;
        (*left_basis)(ind1, ind2) = 1;
        (*left_basis)(ind2, ind1) = 1;
    }

    return;
}

template <typename Type>
void swap_columns_basis(Matrix<Type>& A, size_t ind1, size_t ind2, Matrix<Type>* right_basis = nullptr) {
    assert(0 <= ind1 < A.width());
    assert(0 <= ind2 < A.width());

    for (size_t k = 0; k < A.height(); ++k) {
        std::swap(A(k, ind1), A(k, ind2));
    }

    if (right_basis != nullptr) {
        (*right_basis) = Matrix<Type>::identity(A.width());
        (*right_basis)(ind1, ind1) = 0;
        (*right_basis)(ind2, ind2) = 0;
        (*right_basis)(ind1, ind2) = 1;
        (*right_basis)(ind2, ind1) = 1;
    }

    return;
}

inline void swap_columns(Matrix<long double>& A, int ind1, int ind2) {
    assert(ind1 >= 0 && ind1 < A.width());
    assert(ind2 >= 0 && ind2 < A.width());
    for (int i = 0; i < A.height(); ++i) {
        std::swap(A(i, ind1), A(i, ind2));
    }
}

inline void swap_rows(Matrix<long double>& A, int ind1, int ind2) {
    assert(ind1 >= 0 && ind1 < A.height());
    assert(ind2 >= 0 && ind2 < A.height());
    for (int i = 0; i < A.width(); ++i) {
        std::swap(A(ind1, i), A(ind2, i));
    }
}
}  // namespace details
}  // namespace convolution_svd