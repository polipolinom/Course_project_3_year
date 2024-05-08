#pragma once

#include <math.h>

#include <algorithm>
#include <vector>

#include "../course-project-second-year/algorithms/QR_algorithm.h"
#include "../course-project-second-year/types/matrix.h"
#include "../course-project-second-year/types/vector.h"
#include "../utils/swap.h"
#include "band_reduction.h"
#include "constants.h"
#include "qr_banded.h"

namespace convolution_svd {
using namespace svd_computation;

namespace details {
inline void sort_singular_values(std::vector<long double>& ans, Matrix<long double>* left_basis,
                                 Matrix<long double>* right_basis) {
    for (size_t i = 1; i < ans.size(); ++i) {
        size_t j = i;
        while (j > 0 && ans[j - 1] < ans[j]) {
            std::swap(ans[j - 1], ans[j]);
            if (left_basis) {
                swap_columns(*left_basis, j - 1, j);
            }
            if (right_basis) {
                swap_rows(*right_basis, j - 1, j);
            }
            j--;
        }
    }
}
}  // namespace details

inline std::vector<long double> svd_banded(Matrix<long double> A, size_t band_width,
                                           const long double eps = constants::DEFAULT_EPSILON) {
    std::vector<long double> ans = {};
    apply_banded_qr(A, band_width, ans, 0, 0, A.height() - 1, A.width() - 1, eps);
    std::sort(ans.begin(), ans.end(), std::greater<long double>());
    return ans;
}

inline std::vector<long double> svd_banded_reduction(Matrix<long double> A, Matrix<long double>* left_basis = nullptr,
                                                     Matrix<long double>* right_basis = nullptr,
                                                     const long double eps = constants::DEFAULT_EPSILON) {
    band_diag_reduction(A, left_basis, right_basis, 1e-20);
    std::vector<long double> diag;
    std::vector<long double> subdiag;
    for (size_t ind = 0; ind < A.height(); ++ind) {
        diag.emplace_back(A(ind, 0));
        subdiag.emplace_back(A(ind, 1));
    }
    diag.emplace_back(0);
    if (left_basis) {
        Matrix<long double> new_left_basis(left_basis->height(), left_basis->width() + 1);
        for (size_t i = 0; i < new_left_basis.height(); ++i) {
            for (size_t j = 0; j < new_left_basis.width() - 1; ++j) {
                new_left_basis(i, j) = (*left_basis)(i, j);
            }
        }
        std::vector<long double> ans = {};
        apply_bidiagonal_qr(diag, subdiag, ans, 0, diag.size(), &new_left_basis, right_basis, eps);
        for (size_t i = 0; i < ans.size(); ++i) {
            if (ans[i] < 0.0) {
                for (size_t j = 0; j < new_left_basis.height(); ++j) {
                    new_left_basis(j, i) *= -1;
                }
                ans[i] *= -1;
            }
        }
        details::sort_singular_values(ans, &new_left_basis, right_basis);
        ans.pop_back();
        for (size_t i = 0; i < new_left_basis.height(); ++i) {
            for (size_t j = 0; j < new_left_basis.width() - 1; ++j) {
                (*left_basis)(i, j) = new_left_basis(i, j);
            }
        }
        return ans;
    }
    std::vector<long double> ans = {};
    apply_bidiagonal_qr(diag, subdiag, ans, 0, diag.size(), nullptr, right_basis, eps);
    for (size_t i = 0; i < ans.size(); ++i) {
        if (ans[i] < 0.0) {
            ans[i] *= -1;
        }
    }
    details::sort_singular_values(ans, nullptr, right_basis);
    ans.pop_back();
    return ans;
}

}  // namespace convolution_svd