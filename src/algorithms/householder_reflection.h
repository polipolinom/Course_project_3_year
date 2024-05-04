#pragma once

#include <cmath>
#include <utility>

#include "../course-project-second-year/types/matrix.h"
#include "../course-project-second-year/types/vector.h"
#include "constants.h"

namespace convolution_svd {
using namespace ::svd_computation;

namespace details {

inline long double sign(long double x, long double eps) {
    if (x >= eps) {
        return 1;
    }
    if (x < -eps) {
        return -1;
    }
    return 0;
}

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

template <typename Type>
long double all_zeros_column(const Matrix<Type>& A, size_t row, size_t column, size_t column_end, long double eps) {
    if (column_end < column) {
        std::swap(column, column_end);
    }
    for (size_t k = column + 1; k <= column_end; k += 1) {
        if (std::abs(A(row, k)) > eps) {
            return false;
        }
    }
    return true;
}

template <typename Type>
long double all_zeros_row(const Matrix<Type>& A, size_t row, size_t row_end, size_t column, long double eps) {
    if (row_end < row) {
        std::swap(row, row_end);
    }
    for (size_t k = row + 1; k <= row_end; k += 1) {
        if (std::abs(A(k, column)) > eps) {
            return false;
        }
    }
    return true;
}

inline void mult_left_reflection_banded(Matrix<long double>& A, size_t band_size, Matrix<long double>& reflector,
                                        const size_t row, const size_t row_end, const size_t column, bool set_zero,
                                        const long double eps = convolution_svd::constants::DEFAULT_EPSILON) {
    assert(row >= 0 && row < A.height());
    assert(row_end >= row && row_end < A.height());
    assert(column >= 0 && column <= A.width());

    for (size_t ind = 0; ind < reflector.width(); ++ind) {
        if (std::abs(reflector(0, ind)) < eps) {
            reflector(0, ind) = 0;
        }
    }
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
    if (set_zero) {
        for (size_t i = row + 1; i <= row_end; ++i) {
            A(i, column) = 0.0;
        }
    }
}

inline void mult_right_reflection_banded(Matrix<long double>& A, size_t band_size, Matrix<long double>& reflector,
                                         const size_t row, const size_t column, const size_t column_end, bool set_zero,
                                         const long double eps = convolution_svd::constants::DEFAULT_EPSILON) {
    assert(row >= 0 && row < A.height());
    assert(column_end >= column && column_end < A.width());
    assert(column >= 0 && column <= A.width());

    for (size_t ind = 0; ind < reflector.width(); ++ind) {
        if (std::abs(reflector(0, ind)) < eps) {
            reflector(0, ind) = 0;
        }
    }
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
    if (set_zero) {
        for (size_t i = column + 1; i <= column_end; ++i) {
            A(row, i) = 0.0;
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

    if (std::abs(s) < eps) {
        if (column < column_end) {
            ans(0, 0) = 1;
        } else {
            ans(0, column_end - column) = 1;
        }
        return ans;
    }

    long double coef = 0;
    if (column <= column_end) {
        long double d = -alpha * details::sign(A(row, column), 0);
        ans(0, 0) = A(row, column) - d;
        for (size_t k = column + 1; k <= column_end; ++k) {
            ans(0, k - column) = A(row, k);
        }
        assert(-2 * ans(0, 0) * d >= 0);
        coef = std::sqrt(-2 * ans(0, 0) * d);
    } else {
        long double d = -alpha * details::sign(A(row, column), 0);
        ans(0, column - column_end) = A(row, column) - d;
        for (size_t k = column_end; k < column; ++k) {
            ans(0, k - column_end) = A(row, k);
        }
        coef = std::sqrt(-2 * ans(0, column - column_end) * d);
    }
    if (coef < eps) {
        ans = Matrix<Type>(1, std::abs(column_end - column) + 1);
        if (column < column_end) {
            ans(0, 0) = 1;
        } else {
            ans(0, column_end - column) = 1;
        }
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

    Type alpha = Type(s);

    if (std::abs(s) < eps) {
        if (row < row_end) {
            ans(0, 0) = 1;
        } else {
            ans(0, row_end - row) = 1;
        }
        return ans;
    }

    long double coef = 0;

    if (row <= row_end) {
        long double d = -alpha * details::sign(A(row, column), eps);
        ans(0, 0) = A(row, column) - d;
        for (size_t k = row + 1; k <= row_end; ++k) {
            ans(0, k - row) = A(k, column);
        }
        assert(-2 * ans(0, 0) * d >= 0);
        coef = std::sqrt(-2 * ans(0, 0) * d);
    } else {
        long double d = -alpha * details::sign(A(row, column), eps);
        ans(0, row - row_end) = A(row, column) - d;
        for (size_t k = row_end; k < row; ++k) {
            ans(0, k - row_end) = A(k, column);
        }
        coef = std::sqrt(-2 * ans(0, row - row_end) * d);
    }
    if (coef < eps) {
        ans = Matrix<Type>(1, std::abs(row_end - row) + 1);
        if (row < row_end) {
            ans(0, 0) = 1;
        } else {
            ans(0, row_end - row) = 1;
        }
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