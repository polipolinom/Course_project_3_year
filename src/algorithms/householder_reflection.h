 #pragma once

#include "../course-project-second-year/types/matrix.h"
#include "../course-project-second-year/types/vector.h"
#include "constants.h"

namespace convolution_svd{
using namespace ::svd_computation;

namespace details{

template <typename Type>
long double column_abs_under(const Matrix<Type>& A, const size_t row, const size_t column,
                             size_t stride = 1) {
    long double s = 0.0;
    for (size_t k = row; k < A.height(); k += stride) {
        s += abs(A(k, column)) * abs(A(k, column));
    }
    s = sqrtl(s);
    return s;
}

template <typename Type>
long double row_abs_under(const Matrix<Type>& A, const size_t row, const size_t column,
                          size_t stride = 1) {
    long double s = 0.0;
    for (size_t k = column; k < A.width(); k += stride) {
        s += abs(A(row, k)) * abs(A(row, k));
    }
    s = sqrtl(s);
    return s;
}

template <typename Type>
long double row_abs_segment(const Matrix<Type>& A, 
                            const size_t row, const size_t row_end,
                            const size_t column,
                            size_t stride = 1) {
    long double s = 0.0;
    for (size_t k = row; k <= row_end; k += stride) {
        s += abs(A(k, column)) * abs(A(k, column));
    }
    s = sqrtl(s);
    return s;
}


template <typename Type>
long double column_abs_segment(const Matrix<Type>& A, const size_t row, 
                            const size_t column, const size_t column_end,
                            size_t stride = 1) {
    long double s = 0.0;
    for (size_t k = column; k <= column_end; k += stride) {
        s += abs(A(row, k)) * abs(A(row, k));
    }
    s = sqrtl(s);
    return s;
}
}// namespace details

template <typename Type>
Matrix<Type> right_segment_reflection(Matrix<Type>& A, 
                                      size_t row, size_t column,
                                      size_t column_end, bool change_matrix = true, 
                                      const long double eps = convolution_svd::constants::DEFAULT_EPSILON) {
    
    assert(row >= 0 && row < A.height());
    assert(column >= 0 && column < A.width());
    assert(column_end >= column && column_end < A.width());

    Matrix<Type> ans(1, column_end - column + 1);
    
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

    ans(0, 0) = A(row, column) - alpha;
    for (size_t k = column + 1; k <= column_end; ++k) {
        ans(0, k - column) = A(row, k);
    }
    coef = details::row_abs_under(ans, 0, 0);


    if (coef <= eps) {
        return ans;
    }
    ans /= coef;

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
Matrix<Type> left_segment_reflection(Matrix<Type>& A, 
                                     size_t row, size_t row_end,
                                     size_t column, bool change_matrix = true, 
                                     const long double eps = convolution_svd::constants::DEFAULT_EPSILON) {
    
    assert(row >= 0 && row < A.height());
    assert(column >= 0 && column < A.width());
    assert(row_end >= row && row_end < A.height());

    Matrix<Type> ans(1, row_end - row + 1);
    
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

    ans(0, 0) = A(row, column) - alpha;
    for (size_t k = row + 1; k <= row_end; ++k) {
        ans(0, k - row) = A(k, column);
    }
    coef = details::row_abs_under(ans, 0, 0);


    if (coef <= eps) {
        return ans;
    }
    ans /= coef;

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
} // namespace convolution_svd