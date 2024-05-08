#pragma once

#include <cmath>
#include <utility>

#include "../course-project-second-year/algorithms/givens_rotation.h"
#include "../course-project-second-year/types/matrix.h"
#include "../course-project-second-year/types/vector.h"
#include "constants.h"
#include "householder_reflection.h"

namespace convolution_svd {
using namespace ::svd_computation;

namespace details {
inline void diag_reduct_element(Matrix<long double>& A, size_t i, size_t b, Matrix<long double>* left_basis = nullptr,
                                Matrix<long double>* right_basis = nullptr,
                                const long double eps = convolution_svd::constants::DEFAULT_EPSILON) {
    size_t ind_row = i;
    if (std::abs(A(i, b)) < eps) {
        return;
    }
    long double value = 0;
    while (ind_row < A.height()) {
        if (ind_row == i) {
            auto [cos, sin] = get_givens_rotation(A(ind_row, b - 1), A(ind_row, b), 1e-20);
            if (right_basis) {
                multiply_left_givens(*right_basis, cos, sin, ind_row + b - 1, ind_row + b);
            }
            for (size_t k = ind_row; k < std::min(ind_row + b, A.height()); ++k) {
                long double x = A(k, b - 1 - (k - ind_row));
                long double y = A(k, b - (k - ind_row));

                A(k, b - 1 - (k - ind_row)) = cos * x - sin * y;
                A(k, b - (k - ind_row)) = sin * x + cos * y;
            }
            if (ind_row + b < A.height()) {
                value = -A(ind_row + b, 0) * sin;
                A(ind_row + b, 0) *= cos;
            } else {
                break;
            }
        } else if (std::abs(value) >= eps) {
            auto [cos, sin] = get_givens_rotation(A(ind_row, b), value, 1e-20);
            if (right_basis) {
                multiply_left_givens(*right_basis, cos, sin, ind_row + b, ind_row + b + 1);
            }
            A(ind_row, b) = A(ind_row, b) * cos - sin * value;
            for (size_t k = ind_row + 1; k < std::min(A.height(), ind_row + b + 1); ++k) {
                long double x = A(k, b - (k - ind_row));
                long double y = A(k, b + 1 - (k - ind_row));

                A(k, b - (k - ind_row)) = cos * x - sin * y;
                A(k, b + 1 - (k - ind_row)) = sin * x + cos * y;
            }
            if (ind_row + b + 1 < A.height()) {
                value = -A(ind_row + b + 1, 0) * sin;
                A(ind_row + b + 1, 0) *= cos;
            } else {
                break;
            }
        }

        if (ind_row == i) {
            ind_row += b - 1;
        } else {
            ind_row += b;
        }

        if (ind_row >= A.height()) {
            break;
        }
        if (std::abs(value) < eps) {
            break;
        }

        auto [cos, sin] = get_givens_rotation(A(ind_row, 0), value, 1e-20);
        if (left_basis) {
            multiply_right_givens(*left_basis, cos, sin, ind_row, ind_row + 1);
        }
        A(ind_row, 0) = A(ind_row, 0) * cos - sin * value;

        // multiply_left_givens(A, cos, sin, ind, ind + 1);
        for (size_t k = 1; k <= b; ++k) {
            long double x = A(ind_row, k);
            long double y = A(ind_row + 1, k - 1);

            A(ind_row, k) = cos * x - sin * y;
            A(ind_row + 1, k - 1) = sin * x + cos * y;
        }
        value = -A(ind_row + 1, b) * sin;
        A(ind_row + 1, b) *= cos;
    }
}

inline void down_diag_reduct_element(Matrix<long double>& A, const size_t i, const size_t b, const size_t up_band_size,
                                     Matrix<long double>* left_basis, Matrix<long double>* right_basis,
                                     const long double eps = convolution_svd::constants::DEFAULT_EPSILON) {
    if (std::abs(A(i, b)) < eps) {
        return;
    }
    size_t ind_row = i;
    long double value = 0;
    while (ind_row < A.height()) {
        if (ind_row == i) {
            auto [cos, sin] = get_givens_rotation(A(i - 1, b + 1), A(i, b), 1e-20);
            if (left_basis) {
                multiply_right_givens(*left_basis, cos, sin, i - 1, i);
            }
            for (size_t j = b + 1; j < A.width(); ++j) {
                long double x = A(i - 1, j);
                long double y = A(i, j - 1);

                A(i - 1, j) = cos * x - sin * y;
                A(i, j - 1) = sin * x + cos * y;
            }
            value = -A(i, A.width() - 1) * sin;
            A(i, A.width() - 1) *= cos;
        } else {
            if (std::abs(value) < eps) {
                break;
            }
            auto [cos, sin] = get_givens_rotation(A(ind_row - 1, b), value, 1e-20);
            if (left_basis) {
                multiply_right_givens(*left_basis, cos, sin, ind_row - 1, ind_row);
            }
            A(ind_row - 1, b) = cos * A(ind_row - 1, b) - sin * value;
            for (size_t j = b + 1; j < A.width(); ++j) {
                long double x = A(ind_row - 1, j);
                long double y = A(ind_row, j - 1);

                A(ind_row - 1, j) = cos * x - sin * y;
                A(ind_row, j - 1) = sin * x + cos * y;
            }
            value = -A(ind_row, A.width() - 1) * sin;
            A(ind_row, A.width() - 1) *= cos;
        }
        if (std::abs(value) < eps) {
            break;
        }
        auto [cos, sin] = get_givens_rotation(A(ind_row - 1, A.width() - 1), value);
        if (right_basis) {
            multiply_left_givens(*right_basis, cos, sin, ind_row - 1 + up_band_size - 1, ind_row - 1 + up_band_size);
        }
        A(ind_row - 1, A.width() - 1) = cos * A(ind_row - 1, A.width() - 1) - sin * value;
        for (size_t j = A.width() - 1; j > b; --j) {
            if (ind_row + A.width() - j - 1 >= A.height()) {
                break;
            }
            long double x = A(ind_row + (A.width() - j) - 1, j - 1);
            long double y = A(ind_row + (A.width() - j) - 1, j);

            A(ind_row + (A.width() - j) - 1, j - 1) = cos * x - sin * y;
            A(ind_row + (A.width() - j) - 1, j) = sin * x + cos * y;
        }
        if (ind_row + A.width() - b - 1 >= A.height()) {
            break;
        }
        value = -A(ind_row + (A.width() - b) - 1, b) * sin;
        A(ind_row + (A.width() - b) - 1, b) *= cos;

        ind_row += A.width() - b - 1;
    }
}
}  // namespace details

inline void band_diag_reduction(Matrix<long double>& A, Matrix<long double>* left_basis = nullptr,
                                Matrix<long double>* right_basis = nullptr,
                                const long double eps = convolution_svd::constants::DEFAULT_EPSILON) {
    for (size_t b = A.width(); b > 2; --b) {
        for (size_t i = 0; i < A.height(); ++i) {
            details::diag_reduct_element(A, i, b - 1, left_basis, right_basis, eps);
        }
    }
}

inline void band_down_diag_reduction(Matrix<long double>& A, const size_t down_band_size, const size_t up_band_size,
                                     Matrix<long double>* left_basis = nullptr,
                                     Matrix<long double>* right_basis = nullptr,
                                     const long double eps = convolution_svd::constants::DEFAULT_EPSILON) {
    assert(A.width() == up_band_size + down_band_size);
    assert(up_band_size >= 2);

    for (size_t b = 0; b < down_band_size; ++b) {
        for (size_t i = down_band_size - b; i < A.height(); ++i) {
            details::down_diag_reduct_element(A, i, b, up_band_size, left_basis, right_basis, eps);
        }
    }
}

}  // namespace convolution_svd