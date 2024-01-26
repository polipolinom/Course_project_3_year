#include "../course-project-second-year/types/vector.h"
#include "constants.h"

namespace convolution_svd{
namespace details{

using namespace ::svd_computation;

void conv_block_bidiagonalization(const Matrix<Type>& A, size_t kernel_height = 1, size_t kernel_width = 1,
                                  Matrix<Type>* left_basis = nullptr, Matrix<Type>* right_basis = nullptr, 
                                  const long double eps = constants::DEFAULT_EPSILON) {
    assert()
}

} // namespace details
} // namespace convolution_svd