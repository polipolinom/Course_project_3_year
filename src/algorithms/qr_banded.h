#pragma once

#include <math.h>

#include <vector>

#include "../course-project-second-year/algorithms/QR_algorithm.h"
#include "../course-project-second-year/algorithms/givens_rotation.h"
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

inline void apply_banded_qr(Matrix<long double>&, size_t, std::vector<long double>&, size_t, size_t, size_t, size_t,
                            const long double);

namespace details {

inline long double sign(long double x) {
    if (x >= 0) {
        return 1;
    }
    return -1;
}

inline long double wilkinson_shift(const Matrix<long double>& A, size_t k, size_t row, size_t column,
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

inline Matrix<long double> shift_riley2(const Matrix<long double>& A, size_t k, size_t row_start, size_t column_start,
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

inline bool check_square(Matrix<long double>& A, size_t row, size_t column, size_t len, size_t row_start,
                         size_t column_end, long double eps = constants::DEFAULT_EPSILON) {
    assert(row < A.height());
    assert(column < A.width());
    long double sum = 0;
    for (size_t i = std::max((int)row_start, (int)row - (int)len + 1); i <= row; i++) {
        for (size_t j = column; j <= std::min(column + len, column_end); j++) {
            if (std::abs(A(i, j)) > eps) {
                return false;
            }
        }
    }
    return true;
}

inline bool split(Matrix<long double>& A, size_t band_size, std::vector<long double>& ans, size_t row_start,
                  size_t column_start, size_t row_end, size_t column_end, long double eps_cmp,
                  const long double eps = constants::DEFAULT_EPSILON) {
    for (size_t ind = row_start; ind < row_end; ++ind) {
        if (check_square(A, ind, ind + 1, band_size, row_start, column_end, eps_cmp)) {
            apply_banded_qr(A, band_size, ans, row_start, column_start, ind, ind, eps);
            apply_banded_qr(A, band_size, ans, ind + 1, ind + 1, row_end, column_end, eps);
            return true;
        }
    }
    return false;
}

inline bool delete_small_diags(Matrix<long double>& A, size_t band_size, std::vector<long double>& ans,
                               size_t row_start, size_t column_start, size_t row_end, size_t column_end,
                               long double eps_cmp, const long double eps = constants::DEFAULT_EPSILON) {
    for (size_t ind = row_start; ind <= std::min(row_end, column_end - 1); ++ind) {
        if (std::abs(A(ind, ind)) < eps_cmp) {
            for (size_t k = ind + 1; k <= row_end; ++k) {
                auto [cos, sin] = get_givens_rotation(A(k, k), A(ind, k), 1e-20);
                for (size_t i = k; i <= std::min(column_end, k + band_size); ++i) {
                    long double x = A(k, i);
                    long double y = A(ind, i);

                    A(k, i) = cos * x - sin * y;
                    A(ind, i) = sin * x + cos * y;
                }
            }
            apply_banded_qr(A, band_size, ans, row_start, column_start, ind, ind, eps);
            apply_banded_qr(A, band_size, ans, ind + 1, ind + 1, row_end, column_end, eps);
            return true;
        }
    }
    return false;
}

inline size_t decrease_band(Matrix<long double>& A, size_t& band_size, size_t row_start, size_t column_start,
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

inline void reduce_first(Matrix<long double>& A, size_t band_size, std::vector<long double>& ans, size_t row_start,
                         size_t column_start, size_t row_end, size_t column_end, long double eps_cmp,
                         const long double eps = constants::DEFAULT_EPSILON) {
    for (size_t ind = column_start + 1; ind <= column_end; ++ind) {
        auto [cos, sin] = get_givens_rotation(A(ind, ind), A(row_start, ind), 1e-20);
        for (size_t k = ind; k <= std::min(column_end, ind + band_size); ++k) {
            long double x = A(ind, k);
            long double y = A(row_start, k);

            A(ind, k) = cos * x - sin * y;
            A(row_start, k) = sin * x + cos * y;
        }
    }
    if (std::abs(A(row_start, column_start)) < eps) {
        ans.emplace_back(0);
    } else {
        ans.emplace_back(std::abs(A(row_start, column_start)));
    }
    apply_banded_qr(A, band_size, ans, row_start + 1, column_start + 1, row_end, column_end, eps);
}
}  // namespace details

inline void apply_banded_qr(Matrix<long double>& A, size_t band_size, std::vector<long double>& ans, size_t row_start,
                            size_t column_start, size_t row_end, size_t column_end,
                            const long double eps = constants::DEFAULT_EPSILON) {
    assert(band_size > 0);
    assert(column_start >= 0 && row_start >= 0);
    assert(column_end >= column_start && column_end < A.width());
    assert(row_end >= row_start && row_end < A.height());

    if (column_end - column_start == 0) {
        if (std::abs(A(row_start, column_start)) < eps) {
            ans.emplace_back(0.0);
            return;
        }
        ans.emplace_back(std::abs(A(row_start, column_start)));
        return;
    }

    if (band_size > 2 && band_size >= column_end - column_start || band_size >= row_end - row_start) {
        Matrix<long double> small(row_end - row_start + 1, column_end - column_start + 1);
        for (size_t i = row_start; i <= row_end; ++i) {
            for (size_t j = column_start; j <= column_end; ++j) {
                small(i - row_start, j - column_start) = A(i, j);
            }
        }
        auto values = compute_svd<long double>(small, nullptr, nullptr, eps);
        for (auto x : values) {
            ans.emplace_back(x);
        }
        return;
    }

    size_t operations = 0;
    while (operations < constants::MAX_OPERATIONS * A.height()) {
        long double new_eps = 1;
        for (size_t ind = row_start; ind <= row_end; ++ind) {
            long double sum = 0;
            for (size_t j = ind; j <= std::min(ind + band_size, column_end); ++j) {
                sum += std::abs(A(ind, j));
            }
            new_eps = std::max(new_eps, sum);
        }
        new_eps *= eps;
        operations++;
        it++;
        details::decrease_band(A, band_size, row_start, column_start, row_end, column_end, new_eps);

        if (details::is_diagonal_banded(A, band_size, row_start, column_start, row_end, column_end, new_eps)) {
            break;
        }

        if (band_size == 2 && details::delete_small_diags(A, band_size, ans, row_start, column_start, row_end,
                                                          column_end, new_eps, eps)) {
            return;
        }

        if (details::split(A, band_size, ans, row_start, column_start, row_end, column_end, new_eps, eps)) {
            return;
        }

        if (std::abs(A(row_start, column_start)) < new_eps) {
            details::reduce_first(A, band_size, ans, row_start, column_start, row_end, column_end, new_eps, eps);
            return;
        }

        auto first = details::shift_riley2(A, band_size, row_start, column_start, row_end, column_end);
        auto v = left_segment_reflection(first, 0, std::min(2 * band_size, first.height() - 1), 0, false, 1e-20);
        details::mult_right_reflection_banded(A, 2 * band_size, v, row_start, column_start,
                                              std::min(column_start + 2 * band_size, column_end), false, 1e-20);
        for (size_t ind = row_start; ind <= row_end; ++ind) {
            auto left_reflector =
                left_segment_reflection(A, ind, std::min(row_end, ind + 2 * band_size), ind, false, 1e-20);
            details::mult_left_reflection_banded(A, 2 * band_size, left_reflector, ind,
                                                 std::min(row_end, ind + 2 * band_size), ind, false, 1e-20);

            if (ind + band_size - 1 <= column_end) {
                auto right_reflector =
                    right_segment_reflection(A, ind, ind + band_size - 1,
                                             std::min(ind + band_size - 1 + 2 * band_size, column_end), false, 1e-20);
                details::mult_right_reflection_banded(A, 2 * band_size, right_reflector, ind, ind + band_size - 1,
                                                      std::min(ind + band_size - 1 + 2 * band_size, column_end), false,
                                                      1e-20);
            }
        }
    }
    for (size_t i = row_start; i <= std::min(row_end, column_end); ++i) {
        if (std::abs(A(i, i)) < eps) {
            ans.emplace_back(0);
        } else {
            ans.emplace_back(std::abs(A(i, i)));
        }
    }
    return;
}
}  // namespace convolution_svd