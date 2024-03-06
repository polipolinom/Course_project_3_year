 #pragma once

#include "../course-project-second-year/types/matrix.h"
#include "../course-project-second-year/types/vector.h"
#include "constants.h"

namespace convolution_svd{
namespace details{

using namespace ::svd_computation;

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
Matrix<Type> get_left_reflector(const Matrix<Type>& A, size_t row, size_t column,
                           size_t stride = 1, size_t full = true,
                           const long double eps = convolution_svd::constants::DEFAULT_EPSILON) {
    assert(row >= 0 && row < A.height());
    assert(column >= 0 && column < A.width());
    assert(stride >= 1);

    Matrix<Type> ans;
    
    long double s = 0;
     if (full) {
        ans = Matrix<Type>(A.height(), 1);
    } else {
        ans = Matrix<Type>((A.height() - row + stride - 1) / stride, 1);
    }
    s = column_abs_under(A, row, column, stride);

    if (s <= eps) {
        return ans;
    }

    Type alpha = Type(s);
    if (abs(A(row, column)) > eps) {
        alpha *= A(row, column) / abs(A(row, column));
    }

    long double coef = 0;

    if (full) {
        ans(row, 0) = A(row, column) - alpha;
    } else {
        ans(0, 0) = A(row, column) - alpha;
    }
    for (size_t k = row + stride; k < A.height(); k += stride) {
        if (full) {
            ans(k, 0) = A(k, column);
        } else {
            ans((k - row) / stride, 0) = A(k, column);
        }
    }
    coef = column_abs_under(ans, 0, 0);

    if (coef <= eps) {
        return ans;
    }
    ans /= coef;

    return ans;
}

template <typename Type>
Matrix<Type> get_right_reflector(const Matrix<Type>& A, size_t row, size_t column,
                           size_t stride = 1, size_t full = true,
                           const long double eps = convolution_svd::constants::DEFAULT_EPSILON) {
    assert(row >= 0 && row < A.height());
    assert(column >= 0 && column < A.width());
    assert(stride >= 1);

    Matrix<Type> ans;
    
    long double s = 0;
    if (full) {
        ans = Matrix<Type>(1, A.width());
    } else {
        ans = Matrix<Type>(1, (A.width() - column + stride - 1) / stride);
    }
    s = row_abs_under(A, row, column, stride);

    if (s <= eps) {
        return ans;
    }

    Type alpha = Type(s);
    if (abs(A(row, column)) > eps) {
        alpha *= A(row, column) / abs(A(row, column));
    }

    long double coef = 0;

    if (full) {
        ans(0, column) = A(row, column) - alpha;
    } else {
        ans(0, 0) = A(row, column) - alpha;
    }
    for (size_t k = column + stride; k < A.width(); k += stride) {
        if (full) {
            ans(0, k) = A(row, k);
        } else {
            ans(0, (k - column) / stride) = A(row, k);
        }
    }
    coef = row_abs_under(ans, 0, 0);


    if (coef <= eps) {
        return ans;
    }
    ans /= coef;

    return ans;
}


template <typename Type>
Matrix<Type> get_reflector(const Matrix<Type>& A, size_t row, size_t column,
                           size_t stride = 1, size_t full = true,
                           typename Vector<Type>::Orientation orientation = Vector<Type>::Orientation::Vertical,
                           const long double eps = convolution_svd::constants::DEFAULT_EPSILON) {
    
    assert(row >= 0 && row < A.height());
    assert(column >= 0 && column < A.width());
    assert(stride >= 1);

    if (orientation == Vector<Type>::Orientation::Vertical) {
       return get_left_reflector(A, row, column, stride, full, eps);
    } else {
        return get_right_reflector(A, row, column, stride, full, eps);
    }
}
} // namespace details
} // namespace convolution_svd