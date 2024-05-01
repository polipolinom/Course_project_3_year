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

std::vector<long double> apply_banded_qr(const Matrix<long double>&, size_t, const long double);

namespace details {

long double sign(long double x) {
    if (x >= 0) {
        return 1;
    }
    return -1;
}

long double wilkinson_shift(const Matrix<long double>& A, size_t k, long double eps = constants::DEFAULT_EPSILON) {
    assert(A.height() >= 2);
    long double a1 = 0, a2 = 0, b = 0;
    a1 = A(A.height() - 2, A.width() - 2) * A(A.height() - 2, A.width() - 2) +
         A(A.height() - 1, A.width() - 2) * A(A.height() - 1, A.width() - 2);
    a2 = A(A.height() - 1, A.width() - 1) * A(A.height() - 1, A.width() - 1);
    b = A(A.height() - 1, A.width() - 2) * A(A.height() - 1, A.width() - 1);
    // std::cout << a1 << " " << a2 << " " << b << std::endl;
    long double delta = (a1 - a2) / 2;
    if (delta < eps) {
        return a2 - std::abs(b);
    }
    return a2 - sign(delta) * b * b / (std::abs(delta) + std::sqrt(delta * delta + b * b));
}

Matrix<long double> shift_riley2(const Matrix<long double>& A, size_t k) {
    assert(A.height() >= 2);
    long double a1 = 0, a2 = 0, b = 0;
    a1 = A(A.height() - 2, A.width() - 2) * A(A.height() - 2, A.width() - 2) +
         A(A.height() - 1, A.width() - 2) * A(A.height() - 1, A.width() - 2);
    a2 = A(A.height() - 1, A.width() - 1) * A(A.height() - 1, A.width() - 1);
    b = A(A.height() - 1, A.width() - 2) * A(A.height() - 1, A.width() - 1);

    auto ATA = mult_band(transpose(A), k, 1, A, 1, k);
    auto ATAATA = mult_band(ATA, k, k, ATA, k, k);
    auto first = Matrix<long double>(ATAATA.column(0) - ATA.column(0) * (a1 + a2) +
                                     (a1 * a2 - b * b) * Vector<long double>::standart_basis(0, ATA.height()));
    return first;
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

std::vector<long double> join_vector(const std::vector<long double>& v1, const std::vector<long double>& v2) {
    std::vector<long double> res(v1.size() + v2.size());
    for (size_t i = 0; i < v1.size(); ++i) {
        res[i] = v1[i];
    }
    for (size_t i = 0; i < v2.size(); ++i) {
        res[i + v1.size()] = v2[i];
    }
    return res;
}

inline std::vector<long double> split(Matrix<long double>& A, size_t band_size,
                                      const long double eps = constants::DEFAULT_EPSILON) {
    using Matrix = Matrix<long double>;

    for (size_t ind = 0; ind + 1 < A.height(); ++ind) {
        if (sum_square(A, ind, ind + 1, band_size) < eps) {
            auto [result1, result2] = split_matrix(A, ind, ind);
            auto val1 = apply_banded_qr(result1, band_size, eps);
            auto val2 = apply_banded_qr(result2, band_size, eps);
            return join_vector(val1, val2);
        }
    }
    return {};
}

size_t decrease_band(const Matrix<long double>& A, size_t& band_size,
                     const long double eps = constants::DEFAULT_EPSILON) {
    while (band_size > 2) {
        bool flag = true;
        for (size_t i = 0; i < A.height(); ++i) {
            if (i + band_size - 1 >= A.width()) {
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

std::vector<long double> apply_banded_qr(const Matrix<long double>& banded_matrix, size_t band_size,
                                         const long double eps = constants::DEFAULT_EPSILON) {
    assert(band_size > 0);

    auto A = banded_matrix;

    if (A.width() == 1) {
        if (std::abs(A(0, 0)) < eps) {
            return {0.0};
        }
        return {std::abs(A(0, 0))};
    }

    if (band_size >= banded_matrix.width() || band_size >= banded_matrix.height()) {
        return compute_svd<long double>(banded_matrix, nullptr, nullptr, eps);
    }

    size_t operations = 0;
    while (operations < constants::MAX_OPERATIONS * A.height()) {
        operations++;
        details::decrease_band(A, band_size, eps);

        if (details::is_diagonal_banded(A, band_size, eps)) {
            break;
        }

        auto values = details::split(A, band_size, eps);
        if (values.size() != 0) {
            return values;
        }

        auto shift = details::wilkinson_shift(A, band_size, eps);
        auto first = Matrix<long double>(A.row(0) * A(0, 0));
        first(0, 0) -= shift;
        auto v = left_segment_reflection(first, 0, band_size - 1, 0, false);
        details::mult_right_reflection_banded(A, band_size, v, 0, 0, band_size - 1);
        for (size_t ind = 0; ind < A.height(); ++ind) {
            auto left_reflector =
                left_segment_reflection(A, ind, std::min(A.height() - 1, ind + band_size), ind, false);
            details::mult_left_reflection_banded(A, band_size, left_reflector, ind,
                                                 std::min(A.height() - 1, ind + band_size), ind);
            if (ind + band_size - 1 < A.width()) {
                auto right_reflector = right_segment_reflection(
                    A, ind, ind + band_size - 1, std::min(ind + band_size - 1 + band_size, A.width() - 1), false);
                details::mult_right_reflection_banded(A, band_size, right_reflector, ind, ind + band_size - 1,
                                                      std::min(ind + band_size - 1 + band_size, A.width() - 1));
            }
        }
    }

    std::vector<long double> ans(A.height());
    for (size_t i = 0; i < A.height(); ++i) {
        if (std::abs(A(i, i)) < eps) {
            ans[i] = 0.0;
        } else {
            ans[i] = std::abs(A(i, i));
        }
    }
    return ans;
}
}  // namespace convolution_svd