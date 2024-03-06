 #pragma once

#include "../course-project-second-year/types/vector.h"
#include "../course-project-second-year/types/matrix.h"
#include "../course-project-second-year/utils/set_values.h"
#include "constants.h"
#include "householder_reflection.h"
#include "../course-project-second-year/algorithms/givens_rotation.h"
#include "../course-project-second-year/algorithms/householder_reflections.h"
#include "../utils/swap.h"

#include <iostream>

namespace convolution_svd{
namespace details{

using namespace::svd_computation;

template <typename Type>
void corr_bidiagonalization(Matrix<Type>& A, 
                            size_t kernel_height = 1, size_t kernel_width = 1,
                            size_t block_height = 1, size_t block_width = 1, 
                            Matrix<Type>* left_basis = nullptr, Matrix<Type>* right_basis = nullptr, 
                            const long double eps = constants::DEFAULT_EPSILON) {
    //assert(block_height > 0 && A.height() % block_height == 0);
    //assert(block_width > 0 && A.width() % block_width == 0);
    assert(kernel_height > 0);
    assert(kernel_width > 0);


    //right_reflection(A, 0, 1);
    /*{
        auto [c, s] = get_givens_rotation(A(1, 1), A(2, 1));
        multiply_left_givens(A, c, -s, 2, 1);
    }*/
    //left_reflection(A, 1, 1);
    //right_reflection(A, 1, 2);
    //left_reflection(A, 2, 2);
    //right_reflection(A, 2, 3);
    //left_reflection(A, 3, 3);
    //right_reflection(A, 3, 4);


    svd_computation::details::set_low_values_zero(A);
    std::cout << "\n\n" << A << "\n"; 
}
} // namespace details
} // namespace convolution_svd