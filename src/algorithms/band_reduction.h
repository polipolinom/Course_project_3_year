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
            auto [cos, sin] = get_givens_rotation(A(ind_row, b - 1), A(ind_row, b), eps);
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
            auto [cos, sin] = get_givens_rotation(A(ind_row, b), value, eps);
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

        auto [cos, sin] = get_givens_rotation(A(ind_row, 0), value, eps);
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

// inline void block_band_upper(Matrix<long double>& A, size_t band_width, size_t block_height, size_t block_width,
//                              const long double eps = convolution_svd::constants::DEFAULT_EPSILON) {
//     // band_width is number blocks in band
//     assert(band_width >= 0 && band_width * block_width <= A.width());
//     assert(A.height() % block_height == 0 && A.width() % block_width == 0);
//     if (block_height <= block_width) {
//         if (block_height == 1) {
//             return band_diag_reduction(A, A.width() - A.height() + 1);
//         }
//         for (size_t i = 0; i < A.height() / block_height; ++i) {
//             size_t row = block_height * i;
//             size_t column = block_width * i;
//             for (size_t ind = 0; ind < block_height; ++ind) {
//                 auto v = left_segment_reflection(A, row + ind, row + block_height - 1, column + ind, false, eps);
//                 details::mult_left_reflection_banded(A, band_width * block_width, v, row + ind, row + block_height -
//                 1,
//                                                      column + ind, false, eps);
//             }
//         }
//     } else if (block_height >= block_width * band_width) {
//         for (size_t i = 0; i < A.height() / block_height; ++i) {
//             size_t row = block_height * i;
//             size_t column = block_width * i;
//             for (size_t ind = 0; ind < block_width * band_width; ++ind) {
//                 auto v = left_segment_reflection(A, row + ind, row, column + ind, false, eps);
//                 details::mult_left_reflection_banded(A, band_width * block_width, v, row, row + ind, column + ind,
//                                                      false, eps);
//             }
//         }
//         A.transpose();
//     } else {
//     }
// }

}  // namespace convolution_svd