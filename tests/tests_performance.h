#pragma once

#include <gtest/gtest.h>

#include <chrono>
#include <ctime>
#include <random>

#include "../course-project-second-year/algorithms/svd_computation.h"
#include "../course-project-second-year/types/matrix.h"
#include "../src/algorithms/svd_banded.h"
#include "../src/utils/conv_matrix.h"
#include "random_objects.h"

using namespace svd_computation;
using namespace convolution_svd;
using namespace std::chrono;

std::vector<std::pair<std::pair<int, int>, int>> tests_performance_image(const std::vector<size_t> &ns,
                                                                         const std::vector<size_t> &ms,
                                                                         size_t kernel_height, size_t kernel_width,
                                                                         long double max_number) {
    size_t iterations_count = 1;

    std::vector<std::pair<std::pair<int, int>, int>> res;
    for (size_t ind = 0; ind < ns.size(); ++ind) {
        auto n = ns[ind];
        auto m = ms[ind];
        int total_time = 0;
        int total_time_qr = 0;
        int total_time_reduction = 0;
        for (size_t i = 0; i < iterations_count; i++) {
            auto kernel = get_random_kernel(kernel_height, kernel_height, kernel_width, kernel_width, max_number);
            auto A = correlation_conv(kernel, n, m);
            {
                auto start = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
                compute_svd(A);
                auto finish = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
                total_time += finish - start;
            }

            {
                auto start = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
                svd_banded(A, kernel_width);
                auto finish = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
                total_time_qr += finish - start;
            }

            {
                auto start = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
                svd_banded_reduction(A, kernel_width);
                auto finish = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
                total_time_reduction += finish - start;
            }
        }
        res.push_back({{total_time_qr / iterations_count, total_time_reduction / iterations_count},
                       total_time / iterations_count});
    }
    return res;
}
