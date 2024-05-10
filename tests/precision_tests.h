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

void tests_precision_epsilons(std::vector<long double> epsilons, size_t image_height, size_t image_width,
                              size_t kernel_height, size_t kernel_width, long double max_number) {
    size_t iterations_count = 5;

    for (auto eps : epsilons) {
        auto n = image_height;
        auto m = image_width;
        long double total_qr_min = 0;
        long double total_qr_1quart = 0;
        long double total_qr_mid = 0;
        long double total_qr_max = 0;
        long double total_qr_3quart = 0;
        for (size_t i = 0; i < iterations_count; i++) {
            auto kernel = get_random_kernel(kernel_height, kernel_height, kernel_width, kernel_width, max_number);
            auto A = correlation_conv(kernel, n, m, true);
            auto true_values = compute_svd<long double>(A, nullptr, nullptr, 1e-18);
            auto band_qr_values = svd_banded(A, kernel_width, eps);
            if (true_values.size() != band_qr_values.size()) {
                std::cout << "Wrong count in band qr";
                return;
            }
            auto values = true_values;
            for (size_t i = 0; i < values.size(); ++i) {
                values[i] = std::abs(values[i] - band_qr_values[i]);
            }
            std::sort(values.begin(), values.end());
            total_qr_min += values[0];
            total_qr_max += values.back();
            total_qr_mid += values[values.size() / 2];
            total_qr_1quart += values[values.size() / 4];
            total_qr_3quart += values[3 * values.size() / 4];
        }
        std::cout << "With eps = " << eps << std::endl;
        std::cout << "Min: " << total_qr_min / iterations_count << std::endl;
        std::cout << "Q1: " << total_qr_1quart / iterations_count << std::endl;
        std::cout << "Median: " << total_qr_mid / iterations_count << std::endl;
        std::cout << "Q3: " << total_qr_3quart / iterations_count << std::endl;
        std::cout << "Max: " << total_qr_max / iterations_count << std::endl;
        std::cout << std::endl;
    }
}
