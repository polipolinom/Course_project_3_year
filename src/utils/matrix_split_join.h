#pragma once

#include <cassert>

#include "../course-project-second-year/types/matrix.h"

namespace convolution_svd {
namespace details {
using namespace svd_computation;

template <typename Type>
std::pair<Matrix<Type>, Matrix<Type>> split_matrix(const Matrix<Type>& A, const int row, const int column) {
    assert(row >= 0 && column >= 0 && row < A.height() && column < A.width());
    Matrix<Type> result1(row + 1, column + 1);
    Matrix<Type> result2(A.height() - row - 1, A.width() - column - 1);
    for (size_t i = 0; i <= row; ++i) {
        for (size_t j = 0; j <= column; ++j) {
            result1(i, j) = A(i, j);
        }
    }
    for (size_t i = row + 1; i < A.height(); ++i) {
        for (size_t j = column + 1; j < A.width(); ++j) {
            result2(i - row - 1, j - column - 1) = A(i, j);
        }
    }
    return {result1, result2};
}

template <typename Type>
Matrix<Type> join_matrix(const Matrix<Type>& up, const Matrix<Type>& down) {
    Matrix<Type> result(up.height() + down.height(), up.width() + down.width());
    for (size_t i = 0; i < up.height(); ++i) {
        for (size_t j = 0; j < up.width(); ++j) {
            result(i, j) = up(i, j);
        }
    }
    for (size_t i = 0; i < down.height(); ++i) {
        for (size_t j = 0; j < down.width(); ++j) {
            result(i + up.height(), j + up.width()) = down(i, j);
        }
    }
    return result;
}
}  // namespace details
}  // namespace convolution_svd
