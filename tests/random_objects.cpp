#include "random_objects.h"

#include <cassert>
#include <chrono>
#include <ctime>
#include <random>

Matrix<long double> get_random_kernel(size_t min_height, size_t max_height, size_t min_width, size_t max_width,
                                      long double max_number) {
    assert(max_height > 0 && max_height >= min_height);
    assert(max_width > 0 && max_width >= min_width);

    std::mt19937 rnd(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<long double> distribution(-max_number, max_number);

    int n = rnd() % (max_height - min_height + 1) + min_height;
    int m = rnd() % (max_width - min_width + 1) + min_width;

    Matrix<long double> A(n, m);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            A(i, j) = distribution(rnd);
        }
    }

    return A;
}
