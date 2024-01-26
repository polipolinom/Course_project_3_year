#include <iostream>
#include "algorithms/householder_reflection.h"

using namespace convolution_svd::details;

int main() {
    Matrix<long double> A({{1, 10}, {2, 10}, {2, 10}, {3, 10}, {3, 10}});
    auto ref = get_reflector(A, 0, 0, 2);
    std::cout << (Matrix<long double>::identity(5) - 2.0 * Matrix<long double>({ref}) * conjugate(Matrix<long double>({ref}))) * A;
    return 0;
}