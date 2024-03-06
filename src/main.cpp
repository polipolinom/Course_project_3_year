#include <iostream>
#include "algorithms/householder_reflection.h"
#include "algorithms/bidiagonalization.h"
#include "utils/conv_matrix.h"
#include "../course-project-second-year/algorithms/QR_decomposition.h"
#include "../course-project-second-year/algorithms/givens_rotation.h"
#include "../course-project-second-year/utils/set_values.h"

#include <math.h>

using namespace convolution_svd::details;
using namespace convolution_svd;
using namespace svd_computation::details;

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
    std::cout << a1 << " " << a2 << " " << b << std::endl;
    long double delta = (a1 - a2) / 2;
    return a2 - sign(delta) * b * b / (std::abs(delta) + std::sqrt(delta * delta + b * b));
}

long double sum_diag(const Matrix<long double>& A, size_t k) {
    long double sum = 0;
    for (size_t ind = 0;  ind < A.height(); ++ind) {
        sum += A(ind, ind + k - 1) * A(ind, ind + k - 1);
    }
    return std::sqrt(sum);
}
 
int main() { 
    size_t n = 5;
    auto k = 3;
    Matrix<long double> A = Matrix<long double>::banded({2,3, 4}, 5, 5 + 2);
    std::cout << transpose(A) * A << std::endl;

    long double eps = 1e-8;
    while (sum_diag(A, k) > eps) {
        auto shift = wilkinson_shift(A, k);
        std::cout << sum_diag(A, k) << std::endl;
        std::cout <<  "shift: " << shift << std::endl;
        for (size_t start = 1; start < k; start++) {
            for (size_t ind = start; ind < A.height() + k - 1; ind += k - 1) {
                if (ind == start) {
                    auto [cos, sin] = get_givens_rotation(A(start - 1, start - 1) * A(start - 1, start - 1) - shift, 
                                                          A(start - 1, start) * A(start - 1, start - 1));
                    multiply_right_givens(A, cos, -sin, start, start - 1);
                } else {
                    auto [cos, sin] = get_givens_rotation(A(ind - k, ind - 1), A(ind - k, ind));
                    multiply_right_givens(A, cos, -sin, ind, ind - 1);
                }
                if (ind < A.height()) {
                    auto [cos, sin] = get_givens_rotation(A(ind - 1, ind - 1), A(ind, ind - 1));
                    multiply_left_givens(A, cos, -sin, ind, ind - 1);
                }
            } 
            //std::cout << A << std::endl;
            //return 0;
        }
    }
    return 0;
}