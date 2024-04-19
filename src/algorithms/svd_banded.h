#pragma once

#include "../utils/set_values.h"
#include "../utils/checking_matrices.h"
#include "../utils/matrix_split_join.h"
#include "../course-project-second-year/types/matrix.h"
#include "../course-project-second-year/types/vector.h"
#include "../course-project-second-year/algorithms/svd_computation.h"
#include "constants.h"
#include "householder_reflection.h"

#include <math.h>

namespace convolution_svd {
using namespace svd_computation;

Matrix<long double> apply_banded_qr(const Matrix<long double>&, size_t, const long double);

namespace details {

long double sign(long double x) {
    if (x >= 0) {
        return 1;
    }
    return -1;
}

long double wilkinson_shift(const Matrix<long double>& A, size_t k) {
    assert(A.height() >= 2);
    long double a1 = 0, a2 = 0, b = 0;
    a1 = A(A.height() - 2, A.width() - 2) * A(A.height() - 2, A.width() - 2) + A(A.height() - 1, A.width() - 2) * A(A.height() - 1, A.width() - 2);
    a2 = A(A.height() - 1, A.width() - 1) *  A(A.height() - 1, A.width() - 1);
    b = A(A.height() - 1, A.width() - 2) * A(A.height() - 1, A.width() - 1);
    // std::cout << a1 << " " << a2 << " " << b << std::endl;
    long double delta = (a1 - a2) / 2; 
    if (delta == 0 && b == 0) {
        return 0;
    }
    return a2 - sign(delta) * b * b / (std::abs(delta) + std::sqrt(delta * delta + b * b));
}

Matrix<long double> do_square(const Matrix<long double>& A) {
    Matrix<long double> res(std::max(A.height(), A.width()), std::max(A.height(), A.width()));
    for (size_t i = 0; i < A.height(); ++i) {
        for (size_t j = 0; j < A.width(); ++j) {
            res(i, j) = A(i, j);
        }
    }
    return res;
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

long double sum_square(const Matrix<long double>& A, 
                       size_t row, size_t column, size_t len) {
    assert(row < A.height());
    assert(column < A.width());
    long double sum = 0;
    for (size_t i = std::max(0, (int)row - (int)len + 1); i <= row; i++) {
        for (size_t j = column; j < std::min(column + len, A.width()); j++) {
            sum += abs(A(i, j));
        }
    }
    return sum;
}

inline bool split(Matrix<long double>& A, size_t band_size, 
                  const long double eps = constants::DEFAULT_EPSILON) {
    using Matrix = Matrix<long double>;

    for (size_t ind = 0; ind + 1 < A.height(); ++ind) {
        if (sum_square(A, ind, ind + 1, band_size) < eps) {
            auto [result1, result2] = split_matrix(A, ind, ind);
            result1 = apply_banded_qr(result1, band_size, eps);
            result2 = apply_banded_qr(result2, band_size, eps);
            A = join_matrix(result1, result2);
            return true;
        }
    }
    return false;
}

inline bool erase_small_diagonal(Matrix<long double>& A, size_t band_size, 
                                 const long double eps = constants::DEFAULT_EPSILON) {
    using Matrix = Matrix<long double>;

    for (size_t k = 0; k < A.height(); ++k) {
        if (abs(A(k, k)) < eps) {
            for (size_t ind = k + 1; ind < A.height(); ++ind) {
                auto left_reflector = left_segment_reflection(A, ind, std::min(A.height() - 1, ind + band_size), ind, false);
                details::mult_left_reflection_banded(A, band_size, left_reflector, ind, std::min(A.height() - 1, ind + band_size), ind);
            }
            std::cout << k << std::endl;
            std::cout << A << std::endl;
            break;
        }
    }
    return 0;
}
} // namespace details

Matrix<long double> apply_banded_qr(const Matrix<long double>& banded_matrix, size_t band_size,
                                    const long double eps = constants::DEFAULT_EPSILON) {
    assert(band_size > 0);

    auto A = banded_matrix;

    if (A.width() == 1) {
        if (abs(A(0, 0)) <= eps) {
            return {{0.0}};
        }
        return {{A(0, 0)}};
    }

    if (band_size >= banded_matrix.width() || band_size >= banded_matrix.height()) {
        auto diag = compute_svd<long double>(banded_matrix, nullptr, nullptr, eps);
        return Matrix<long double>::diagonal(diag, banded_matrix.height(), banded_matrix.width());
    }

    size_t operations = 0;
    while (operations < constants::MAX_OPERATIONS * A.height()) {
         operations++;

        if (details::split(A, band_size, eps)) {
            return A;
        }

        /*if (details::erase_small_diagonal(A, band_size, eps)) {
            return A;
        }*/

         if (details::is_diagonal_banded(A, band_size)) {
            break;
        }

        auto shift = details::wilkinson_shift(A, band_size);
        auto first = Matrix<long double>(A.row(0) * A(0, 0));
        first(0, 0) -= shift;
        auto v = left_segment_reflection(first, 0, band_size - 1, 0, false);
        details::mult_right_reflection_banded(A, band_size, v, 0, 0, band_size - 1);
        for (size_t ind = 0; ind < A.height(); ++ind) {
            auto left_reflector = left_segment_reflection(A, ind, std::min(A.height() - 1, ind + band_size), ind, false);
            details::mult_left_reflection_banded(A, band_size, left_reflector, ind, std::min(A.height() - 1, ind + band_size), ind);
            if (ind + band_size - 1 < A.width()) {
                auto right_reflector = right_segment_reflection(A, ind, ind + band_size - 1, std::min(ind + band_size - 1 + band_size, A.width() - 1), false);
                details::mult_right_reflection_banded(A, band_size, right_reflector, ind, ind + band_size - 1, std::min(ind + band_size - 1 + band_size, A.width() - 1));
            }
        }
    }
    details::set_low_values_zero(A);
    return A;
}
} // namespace convolution_svd