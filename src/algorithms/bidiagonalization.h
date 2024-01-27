 #pragma once

#include "../course-project-second-year/types/vector.h"
#include "../course-project-second-year/types/matrix.h"
#include "constants.h"
#include "householder_reflection.h"

namespace convolution_svd{
namespace details{

using namespace ::svd_computation;

template <typename Type>
void conv_block_bidiagonalization(Matrix<Type>& A, size_t kernel_height = 1, size_t kernel_width = 1,
                                  Matrix<Type>* left_basis = nullptr, Matrix<Type>* right_basis = nullptr, 
                                  const long double eps = constants::DEFAULT_EPSILON) {
    assert(kernel_height >= 1 && A.height() % kernel_height == 0);
    assert(kernel_width >= 1 && A.width() % kernel_width == 0);

    auto u0 = get_reflector(A, 0, 0, kernel_height);
    auto M0 = Matrix<Type>::identity(A.height()) - Type(2.0) * u0 * conjugate(u0);
    
    A = M0 * A;  

    auto u1 = get_reflector(A, 1, 1, kernel_height);
    auto M1 = Matrix<Type>::identity(A.height()) - Type(2.0) * u1 * conjugate(u1);

    A = M1 * A;

    auto u2 = get_reflector(A, 2, 2, kernel_height);
    auto M2 = Matrix<Type>::identity(A.height()) - Type(2.0) * u2 * conjugate(u2);

    A = M2 * A;

    std::cout << u0 * conjugate(u0) + u1 * conjugate(u1) + u2 * conjugate(u2) << "\n===\n";
}

} // namespace details
} // namespace convolution_svd