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
inline void reduct_element(Matrix<long double>& A, size_t i, size_t b,
                           const long double eps = convolution_svd::constants::DEFAULT_EPSILON) {
    size_t ind = i;
    while (ind < A.height() && ind + b - 1 < A.width()) {
        if (ind == i) {
            if (std::abs(A(ind, ind + b - 1)) < eps) {
                break;
            }
            auto [cos, sin] = get_givens_rotation(A(ind, ind + b - 2), A(ind, ind + b - 1), eps);
            for (size_t k = ind; k < std::min(ind + b + 1, A.height()); ++k) {
                long double x = A(k, ind + b - 2);
                long double y = A(k, ind + b - 1);

                A(k, ind + b - 2) = cos * x - sin * y;
                A(k, ind + b - 1) = sin * x + cos * y;
            }
        } else if (ind + b < A.width() && std::abs(A(ind, ind + b)) >= eps) {
            auto [cos, sin] = get_givens_rotation(A(ind, ind + b - 1), A(ind, ind + b), eps);
            for (size_t k = ind; k < std::min(ind + b + 1, A.height()); ++k) {
                long double x = A(k, ind + b - 1);
                long double y = A(k, ind + b);

                A(k, ind + b - 1) = cos * x - sin * y;
                A(k, ind + b) = sin * x + cos * y;
            }
        }

        ind += b - 2;

        if (ind + 1 >= A.height() || ind >= A.width()) {
            continue;
        }
        if (std::abs(A(ind + 1, ind)) < eps) {
            continue;
        }
        auto [cos, sin] = get_givens_rotation(A(ind, ind), A(ind + 1, ind), eps);
        for (size_t k = ind; k < std::min(A.width(), ind + b + 1); ++k) {
            long double x = A(ind, k);
            long double y = A(ind + 1, k);

            A(ind, k) = cos * x - sin * y;
            A(ind + 1, k) = sin * x + cos * y;
        }
    }
}
}  // namespace details

inline void band_reduction(Matrix<long double>& A, size_t band_width,
                           const long double eps = convolution_svd::constants::DEFAULT_EPSILON) {
    assert(band_width >= 0 && band_width <= A.width());

    for (size_t b = band_width; b > 2; --b) {
        for (size_t i = 0; i < A.height(); ++i) {
            if (i + b - 1 >= A.width()) {
                break;
            }
            details::reduct_element(A, i, b, eps);
        }
    }
}

inline void block_band_upper(Matrix<long double>& A, size_t band_width, size_t block_height, size_t block_width,
                             const long double eps = convolution_svd::constants::DEFAULT_EPSILON) {
    // band_width is number blocks in band
    assert(band_width >= 0 && band_width * block_width <= A.width());
    assert(A.height() % block_height == 0 && A.width() % block_width == 0);
    if (block_height <= block_width) {
        if (block_height == 1) {
            return band_reduction(A, A.width() - A.height() + 1);
        }
        for (size_t i = 0; i < A.height() / block_height; ++i) {
            size_t row = block_height * i;
            size_t column = block_width * i;
            for (size_t ind = 0; ind < block_height; ++ind) {
                auto v = left_segment_reflection(A, row + ind, row + block_height - 1, column + ind, false, eps);
                details::mult_left_reflection_banded(A, band_width * block_width, v, row + ind, row + block_height - 1,
                                                     column + ind, false, eps);
            }
        }
    } else if (block_height >= block_width * band_width) {
        for (size_t i = 0; i < A.height() / block_height; ++i) {
            size_t row = block_height * i;
            size_t column = block_width * i;
            for (size_t ind = 0; ind < block_width * band_width; ++ind) {
                auto v = left_segment_reflection(A, row + ind, row, column + ind, false, eps);
                details::mult_left_reflection_banded(A, band_width * block_width, v, row, row + ind, column + ind,
                                                     false, eps);
            }
        }
        A.transpose();
    } else {
    }
}

}  // namespace convolution_svd