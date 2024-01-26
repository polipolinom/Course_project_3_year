#include "../course-project-second-year/types/matrix.h"
#include "../course-project-second-year/types/vector.h"
#include "constants.h"

namespace convolution_svd{
namespace details{

using namespace ::svd_computation;

template <typename Type>
long double abs_under(const Vector<Type>& v, const size_t ind) {
    long double s = 0.0;
    for (size_t k = ind; k < v.size(); ++k) {
        s += abs(v[k]) * abs(v[k]);
    }
    s = sqrtl(s);
    return s;
}

template <typename Type>
Vector<Type> get_reflector(const Vector<Type>& v, const size_t ind, 
                           const long double eps = convolution_svd::constants::DEFAULT_EPSILON) {
    assert(ind >= 0 && ind < v.size());
    
    Vector<Type> ans(v.size());
    long double s = abs_under(v, ind);
     if (s <= eps) {
        return ans;
    }

    Type alpha = Type(s);
    if (abs(v[ind]) > eps) {
        alpha *= v[ind] / abs(v[ind]);
    }

    ans[ind] = v[ind] - alpha;
    for (size_t k = ind + 1; k < v.size(); ++k) {
        ans[k] = v[k];
    }

    long double coef = abs_under(ans, 0);
    if (coef <= eps) {
        return ans;
    }
    ans /= coef;
    return ans;
}
} // namespace details
} // namespace convolution_svd