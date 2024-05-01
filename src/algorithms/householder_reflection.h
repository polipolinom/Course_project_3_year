#pragma once

#include <cmath>
#include <utility>

#include "../course-project-second-year/types/matrix.h"
#include "../course-project-second-year/types/vector.h"
#include "constants.h"

namespace convolution_svd {
using namespace ::svd_computation;

namespace details {

template <typename Type>
long double column_abs_under(const Matrix<Type>& A, const size_t row, const size_t column, size_t stride = 1) {
    long double s = 0.0;
    for (size_t k = row; k < A.height(); k += stride) {
        s += abs(A(k, column)) * abs(A(k, column));
    }
    s = sqrtl(s);
    return s;
}

template <typename Type>
long double row_abs_under(const Matrix<Type>& A, const size_t row, const size_t column, size_t stride = 1) {
    long double s = 0.0;
    for (size_t k = column; k < A.width(); k += stride) {
        s += abs(A(row, k)) * abs(A(row, k));
    }
    s = sqrtl(s);
    return s;
}

template <typename Type>
long double row_abs_segment(const Matrix<Type>& A, size_t row, size_t row_end, size_t column, size_t stride = 1) {
    long double s = 0.0;
    if (row_end < row) {
        std::swap(row, row_end);
    }
    for (size_t k = row; k <= row_end; k += stride) {
        s += abs(A(k, column)) * abs(A(k, column));
    }
    s = sqrtl(s);
    return s;
}

template <typename Type>
long double column_abs_segment(const Matrix<Type>& A, size_t row, size_t column, size_t column_end, size_t stride = 1) {
    long double s = 0.0;
    if (column_end < column) {
        std::swap(column, column_end);
    }
    for (size_t k = column; k <= column_end; k += stride) {
        s += abs(A(row, k)) * abs(A(row, k));
    }
    s = sqrtl(s);
    return s;
}

void mult_left_reflection_banded(Matrix<long double>& A, size_t band_size, Matrix<long double>& reflector,
                                 const size_t row, const size_t row_end, const size_t column) {
    assert(row >= 0 && row < A.height());
    assert(row_end >= row && row_end < A.height());
    assert(column >= 0 && column <= A.width());

    Vector<long double> tmp(4 * band_size);
    for (size_t ind = 0; ind < tmp.size(); ++ind) {
        int i = (int)(ind + column) - 2 * (int)band_size;
        if (i < 0 || i >= A.width()) {
            continue;
        }
        for (size_t j = row; j <= row_end; ++j) {
            tmp[ind] += A(j, i) * reflector(0, j - row);
        }
    }

    for (size_t k = 0; k < tmp.size(); ++k) {
        int i = (int)(k + column) - 2 * (int)band_size;
        if (i < 0 || i >= A.width()) {
            continue;
        }
        for (size_t ind = row; ind <= row_end; ++ind) {
            A(ind, i) -= 2.0 * tmp[k] * reflector(0, ind - row);
        }
    }
}

void mult_right_reflection_banded(Matrix<long double>& A, size_t band_size, Matrix<long double>& reflector,
                                  const size_t row, const size_t column, const size_t column_end) {
    assert(row >= 0 && row < A.height());
    assert(column_end >= column && column_end < A.width());
    assert(column >= 0 && column <= A.width());

    Vector<long double> tmp(4 * band_size);
    for (size_t ind = 0; ind < tmp.size(); ++ind) {
        int i = (int)(ind + row) - 2 * (int)band_size;
        if (i < 0 || i >= A.height()) {
            continue;
        }
        for (size_t j = column; j <= column_end; ++j) {
            tmp[ind] += A(i, j) * reflector(0, j - column);
        }
    }

    for (size_t k = 0; k < tmp.size(); ++k) {
        int i = (int)(k + row) - 2 * (int)band_size;
        if (i < 0 || i >= A.height()) {
            continue;
        }
        for (size_t ind = column; ind <= column_end; ++ind) {
            A(i, ind) -= 2.0 * tmp[k] * reflector(0, ind - column);
        }
    }
}
}  // namespace details

template <typename Type>
Matrix<Type> right_segment_reflection(Matrix<Type>& A, int row, int column, int column_end, bool change_matrix = true,
                                      const long double eps = convolution_svd::constants::DEFAULT_EPSILON) {
    assert(row >= 0 && row < A.height());
    assert(column >= 0 && column < A.width());
    assert((column_end >= column && column_end < A.width()) || (column_end <= column && column_end >= 0));

    Matrix<Type> ans(1, std::abs(column_end - column) + 1);

    long double s = 0;
    s = details::column_abs_segment(A, row, column, column_end);

    Type alpha = Type(s);
    if (abs(A(row, column)) > eps) {
        alpha *= A(row, column) / abs(A(row, column));
    }

    if (s <= eps) {
        return ans;
    }

    long double coef = 0;

    if (column <= column_end) {
        ans(0, 0) = A(row, column) - alpha;
        for (size_t k = column + 1; k <= column_end; ++k) {
            ans(0, k - column) = A(row, k);
        }
    } else {
        ans(0, column - column_end) = A(row, column) - alpha;
        for (size_t k = column_end; k < column; ++k) {
            ans(0, k - column_end) = A(row, k);
        }
    }
    coef = details::row_abs_under(ans, 0, 0);

    if (coef <= eps) {
        return ans;
    }
    ans /= coef;

    if (column_end < column) {
        std::swap(column, column_end);
    }
    if (change_matrix) {
        Vector<Type> tmp(A.height());
        for (size_t ind = 0; ind < tmp.size(); ++ind) {
            for (size_t j = column; j <= column_end; ++j) {
                tmp[ind] += A(ind, j) * ans(0, j - column);
            }
        }

        for (size_t k = 0; k < A.height(); ++k) {
            for (size_t ind = column; ind <= column_end; ++ind) {
                A(k, ind) -= Type(2.0) * tmp[k] * ans(0, ind - column);
            }
        }
    }
    return ans;
}

template <typename Type>
Matrix<Type> left_segment_reflection(Matrix<Type>& A, int row, int row_end, int column, bool change_matrix = true,
                                     const long double eps = convolution_svd::constants::DEFAULT_EPSILON) {
    assert(row >= 0 && row < A.height());
    assert(column >= 0 && column < A.width());
    assert((row_end >= row && row_end < A.height()) || (row_end <= row && row_end >= 0));

    Matrix<Type> ans(1, std::abs(row_end - row) + 1);

    long double s = 0;
    s = details::row_abs_segment(A, row, row_end, column);

    if (s <= eps) {
        return ans;
    }

    Type alpha = Type(s);
    if (abs(A(row, column)) > eps) {
        alpha *= A(row, column) / abs(A(row, column));
    }

    long double coef = 0;

    if (row <= row_end) {
        ans(0, 0) = A(row, column) - alpha;
        for (size_t k = row + 1; k <= row_end; ++k) {
            ans(0, k - row) = A(k, column);
        }
    } else {
        ans(0, row - row_end) = A(row, column) - alpha;
        for (size_t k = row_end; k < row; ++k) {
            ans(0, k - row_end) = A(k, column);
        }
    }
    coef = details::row_abs_under(ans, 0, 0);

    if (coef <= eps) {
        return ans;
    }
    ans /= coef;

    if (row_end < row) {
        std::swap(row, row_end);
    }
    if (change_matrix) {
        Vector<Type> tmp(A.width());
        for (size_t ind = 0; ind < tmp.size(); ++ind) {
            for (size_t j = row; j <= row_end; ++j) {
                tmp[ind] += A(j, ind) * ans(0, j - row);
            }
        }

        for (size_t k = 0; k < A.width(); ++k) {
            for (size_t ind = row; ind <= row_end; ++ind) {
                A(ind, k) -= Type(2.0) * tmp[k] * ans(0, ind - row);
            }
        }
    }
    return ans;
}
}  // namespace convolution_svd