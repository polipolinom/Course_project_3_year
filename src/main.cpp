#include <iostream>
#include "algorithms/householder_reflection.h"

using namespace convolution_svd::details;

int main() {
    auto ref = get_reflector(Vector<long double>({1, 2, 3}), 0);
    std::cout << (Matrix<long double>::identity(3) - 2.0 * Matrix<long double>({ref}) * conjugate(Matrix<long double>({ref}))) * Vector<long double>({1, 2, 3});
    return 0;
}