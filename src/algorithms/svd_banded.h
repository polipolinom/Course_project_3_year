#pragma once

#include <math.h>

#include <algorithm>
#include <vector>

#include "../course-project-second-year/algorithms/QR_algorithm.h"
#include "../course-project-second-year/types/matrix.h"
#include "../course-project-second-year/types/vector.h"
#include "band_reduction.h"
#include "constants.h"
#include "qr_banded.h"

namespace convolution_svd {
using namespace svd_computation;

inline std::vector<long double> svd_banded(Matrix<long double> A, size_t band_width,
                                           const long double eps = constants::DEFAULT_EPSILON) {
    std::vector<long double> ans = {};
    apply_banded_qr(A, band_width, ans, 0, 0, A.height() - 1, A.width() - 1, eps);
    std::sort(ans.begin(), ans.end(), std::greater<long double>());
    return ans;
}

inline std::vector<long double> svd_banded_reduction(Matrix<long double> A, size_t band_width,
                                                     const long double eps = constants::DEFAULT_EPSILON) {
    band_reduction(A, band_width, 1e-20);
    size_t min_size = std::min(A.height(), A.width());
    Matrix<long double> B(min_size, min_size);
    for (size_t i = 0; i < min_size; ++i) {
        B(i, i) = A(i, i);
        if (i + 1 < min_size) {
            B(i, i + 1) = A(i, i + 1);
        }
    }
    std::vector<long double> ans = {};
    apply_banded_qr(B, 2, ans, 0, 0, A.height() - 1, A.width() - 1, eps);
    std::sort(ans.begin(), ans.end(), std::greater<long double>());
    return ans;
}

}  // namespace convolution_svd