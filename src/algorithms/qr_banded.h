#pragma once

#include <math.h>

#include <vector>

#include "../course-project-second-year/algorithms/QR_algorithm.h"
#include "../course-project-second-year/algorithms/svd_computation.h"
#include "../course-project-second-year/types/matrix.h"
#include "../course-project-second-year/types/vector.h"
#include "../utils/checking_matrices.h"
#include "../utils/matrix_split_join.h"
#include "../utils/set_values.h"
#include "band_reduction.h"
#include "constants.h"
#include "householder_reflection.h"

namespace convolution_svd {
using namespace svd_computation;

std::vector<long double> apply_banded_qr(Matrix<long double>&, size_t, size_t, size_t, size_t, size_t,
                                         const long double);

namespace details {

long double sign(long double x) {
    if (x >= 0) {
        return 1;
    }
    return -1;
}

long double wilkinson_shift(const Matrix<long double>& A, size_t k, size_t row, size_t column,
                            long double eps = constants::DEFAULT_EPSILON) {
    assert(row > 0 && column > 0);
    assert(row < A.height() && column < A.width());
    long double a1 = 0, a2 = 0, b = 0;
    for (size_t i = std::max(0, (int)column - (int)k - 2); i <= row; ++i) {
        a1 += A(i, column - 1) * A(i, column - 1);
    }
    for (size_t i = std::max(0, (int)column - (int)k - 1); i <= row; ++i) {
        a2 += A(i, column) * A(i, column);
    }
    for (size_t i = std::max(0, (int)column - (int)k - 1); i <= row; ++i) {
        b += A(i, column) * A(i, column - 1);
    }
    // std::cout << a1 << " " << a2 << " " << b << std::endl;
    long double delta = (a1 - a2) / 2;
    if (delta < eps) {
        return a2 - std::abs(b);
    }
    return a2 - sign(delta) * b * b / (std::abs(delta) + std::sqrt(delta * delta + b * b));
}

Matrix<long double> shift_riley2(const Matrix<long double>& A, size_t k, size_t row_start, size_t column_start,
                                 size_t row_end, size_t column_end) {
    assert(row_end > row_start && column_end > column_start);
    assert(row_end < A.height() && column_end < A.width());
    long double a1 = 0, a2 = 0, b = 0;
    for (size_t i = std::max(0, (int)column_end - (int)k - 2); i <= row_end; ++i) {
        a1 += A(i, column_end - 1) * A(i, column_end - 1);
    }
    for (size_t i = std::max(0, (int)column_end - (int)k - 1); i <= row_end; ++i) {
        a2 += A(i, column_end) * A(i, column_end);
    }
    for (size_t i = std::max(0, (int)column_end - (int)k - 1); i <= row_end; ++i) {
        b += A(i, column_end) * A(i, column_end - 1);
    }

    Matrix<long double> first(column_end - column_start + 1, 1);
    for (size_t i = column_start; i <= std::min(column_end, column_start + k - 1); ++i) {
        first(i - column_start, 0) = A(row_start, i) * A(row_start, column_start);
    }
    Matrix<long double> mid(row_end - row_start + 1, 1);
    for (size_t i = row_start; i < std::min(row_end + 1, row_start + k); ++i) {
        for (size_t j = 0; j < std::min(k, column_end - column_start + 1); ++j) {
            mid(i - row_start, 0) += A(i, j + column_start) * first(j, 0);
        }
    }
    Matrix<long double> second(column_end - column_start + 1, 1);
    for (size_t i = column_start; i < std::min(column_end + 1, column_start + 2 * k); ++i) {
        for (size_t j = 0; j < std::min(k, row_end - row_start + 1); ++j) {
            second(i - column_start, 0) += A(j + row_start, i) * mid(j, 0);
        }
    }

    auto shift = second - (a1 + a2) * first +
                 (a1 * a2 - b * b) * Vector<long double>::standart_basis(0, column_end - column_start + 1);
    return shift;
}

long double sum_square(const Matrix<long double>& A, size_t row, size_t column, size_t len) {
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

inline std::vector<long double> split(Matrix<long double>& A, size_t band_size, size_t row_start, size_t column_start,
                                      size_t row_end, size_t column_end,
                                      const long double eps = constants::DEFAULT_EPSILON) {
    using Matrix = Matrix<long double>;

    for (size_t ind = row_start; ind < row_end; ++ind) {
        if (sum_square(A, ind, ind + 1, band_size) < eps) {
            auto val1 = apply_banded_qr(A, band_size, row_start, column_start, ind, ind, eps);
            auto val2 = apply_banded_qr(A, band_size, ind + 1, ind + 1, row_end, column_end, eps);
            return join_vector(val1, val2);
        }
    }
    return {};
}

size_t decrease_band(const Matrix<long double>& A, size_t& band_size, size_t row_start, size_t column_start,
                     size_t row_end, size_t column_end, const long double eps = constants::DEFAULT_EPSILON) {
    while (band_size > 2) {
        bool flag = true;
        for (size_t i = row_start; i <= row_end; ++i) {
            if (i + band_size - 1 > column_end) {
                break;
            }
            if (std::abs(A(i, i + band_size - 1)) > eps) {
                flag = false;
                break;
            }
        }
        if (flag == false) {
            break;
        }
        --band_size;
    }
    return band_size;
}
}  // namespace details

std::vector<long double> apply_banded_qr(Matrix<long double>& A, size_t band_size, size_t row_start,
                                         size_t column_start, size_t row_end, size_t column_end,
                                         const long double eps = constants::DEFAULT_EPSILON) {
    assert(band_size > 0);
    assert(column_start >= 0 && row_start >= 0);
    assert(column_end >= column_start && column_end < A.width());
    assert(row_end >= row_start && row_end < A.height());

    if (column_end - column_start == 0) {
        if (std::abs(A(row_start, column_start)) < eps) {
            return {0.0};
        }
        return {std::abs(A(row_start, column_start))};
    }

    if (band_size >= column_end - column_start || band_size >= row_end - row_start) {
        Matrix<long double> small(row_end - row_start + 1, column_end - column_start + 1);
        for (size_t i = row_start; i <= row_end; ++i) {
            for (size_t j = column_start; j <= column_end; ++j) {
                small(i - row_start, j - column_start) = A(i, j);
            }
        }
        return compute_svd<long double>(small, nullptr, nullptr, eps);
    }

    size_t operations = 0;
    while (operations < constants::MAX_OPERATIONS * (row_end - row_start + 1)) {
        operations++;
        details::decrease_band(A, band_size, row_start, column_start, row_end, column_end, eps);

        if (details::is_diagonal_banded(A, band_size, row_start, column_start, row_end, column_end, eps)) {
            break;
        }

        auto values = details::split(A, band_size, row_start, column_start, row_end, column_end, eps);
        if (values.size() != 0) {
            return values;
        }

        auto shift = details::wilkinson_shift(A, band_size, row_end, column_end, eps);
        Matrix<long double> first(column_end - column_start + 1, 1);
        for (size_t i = column_start; i <= std::min(column_end, column_start + band_size - 1); ++i) {
            first(i - column_start, 0) = A(row_start, i) * A(row_start, column_start);
        }
        first(0, 0) -= shift;
        auto v = left_segment_reflection(first, 0, band_size - 1, 0, false);
        details::mult_right_reflection_banded(A, band_size, v, row_start, column_start, column_start + band_size - 1);
        for (size_t ind = row_start; ind <= row_end; ++ind) {
            auto left_reflector = left_segment_reflection(A, ind, std::min(row_end, ind + band_size), ind, false);
            details::mult_left_reflection_banded(A, band_size, left_reflector, ind, std::min(row_end, ind + band_size),
                                                 ind);
            if (ind + band_size - 1 <= column_end) {
                auto right_reflector = right_segment_reflection(
                    A, ind, ind + band_size - 1, std::min(ind + band_size - 1 + band_size, column_end), false);
                details::mult_right_reflection_banded(A, band_size, right_reflector, ind, ind + band_size - 1,
                                                      std::min(ind + band_size - 1 + band_size, column_end));
            }
        }
    }

    std::vector<long double> ans(std::min(row_end - row_start, column_end - column_start) + 1);
    for (size_t i = row_start; i <= std::min(row_end, column_end); ++i) {
        if (std::abs(A(i, i)) < eps) {
            ans[i] = 0.0;
        } else {
            ans[i] = std::abs(A(i, i));
        }
    }
    return ans;
}
}  // namespace convolution_svd