#include <iostream>
#include "algorithms/svd_banded.h"
#include "../course-project-second-year/types/matrix.h"

#include <math.h>

using namespace convolution_svd;
using namespace svd_computation;
 
int main() { 
    size_t n = 5;
    auto k = 3;
    long double eps = 1e-8;
    Matrix<long double> A = Matrix<long double>::banded({2,3, 4}, 5, 5 + 2);
    A = apply_banded_qr(A, k);
    std::cout << A;
    return 0;
}