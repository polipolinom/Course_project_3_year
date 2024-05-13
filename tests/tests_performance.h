#pragma once

#include <gtest/gtest.h>

#include <chrono>
#include <ctime>
#include <random>
#include <vector>

#include "../course-project-second-year/algorithms/svd_computation.h"
#include "../course-project-second-year/types/matrix.h"
#include "../src/algorithms/svd_banded.h"
#include "../src/algorithms/svd_convolution.h"
#include "../src/utils/conv_matrix.h"
#include "random_objects.h"

using namespace svd_computation;
using namespace convolution_svd;
using namespace std::chrono;

std::vector<std::pair<int, int>> tests_performance_image(const std::vector<size_t> &ms, size_t kernel_width,
                                                         size_t C_in, size_t C_out, long double max_number) {
    size_t iterations_count = 2;

    std::vector<std::pair<int, int>> res;
    for (size_t ind = 0; ind < ms.size(); ++ind) {
        int m = ms[ind];
        int total_time_full = 0;
        int total_time = 0;
        for (size_t i = 0; i < iterations_count; i++) {
            std::vector<std::vector<Matrix<long double>>> kernels(C_in, std::vector<Matrix<long double>>(C_out));
            for (size_t j = 0; j < C_in; ++j) {
                for (size_t k = 0; k < C_out; ++k) {
                    kernels[j][k] = get_random_kernel(1, 1, kernel_width, kernel_width, max_number);
                }
            }

            Matrix<long double> left_basis;
            Matrix<long double> right_basis;

            {
                auto start = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
                svd_convolution_1d(kernels, m, &left_basis, &right_basis, 1, false);
                auto finish = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
                total_time += finish - start;
            }

            {
                auto start = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
                svd_convolution_1d(kernels, m, &left_basis, &right_basis, 1, true);
                auto finish = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
                total_time_full += finish - start;
            }
        }
        res.push_back({total_time_full / iterations_count, total_time / iterations_count});
        std::cout << m << std::endl;
    }
    return res;
}
