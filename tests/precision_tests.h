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

std::vector<std::pair<long double, long double>> tests_precision_epsilons(std::vector<long double> epsilons,
                                                                          size_t image_height, size_t image_width,
                                                                          size_t kernel_height, size_t kernel_width,
                                                                          long double max_number) {
    size_t iterations_count = 10;

    std::vector<std::pair<long double, long double>> res;
    for (auto eps : epsilons) {
        auto n = image_height;
        auto m = image_width;
        long double total_qr = 0;
        long double total_reduction = 0;
        for (size_t i = 0; i < iterations_count; i++) {
            auto kernel = get_random_kernel(kernel_height, kernel_height, kernel_width, kernel_width, max_number);
            auto A = correlation_conv(kernel, n, m, true);
            auto true_values = compute_svd<long double>(A, nullptr, nullptr, eps);
            auto band_qr_values = svd_banded(A, kernel_width, eps);
            auto band_reduction_values = svd_banded(A, kernel_width, eps);
            if (true_values.size() != band_qr_values.size()) {
                std::cout << "Wrong count in band qr";
                return {};
            }
            if (true_values.size() != band_reduction_values.size()) {
                std::cout << "Wrong count in band reduction";
                return {};
            }
            long double qr = 0;
            long double reduction = 0;
            for (size_t ind = 0; ind < true_values.size(); ++ind) {
                if (std::abs(true_values[ind] - band_qr_values[ind]) == 0) {
                    qr += 0;
                } else {
                    qr += std::abs(true_values[ind] - band_qr_values[ind]) / std::abs(true_values[ind]);
                }
                if (std::abs(true_values[ind] - band_reduction_values[ind]) == 0) {
                    reduction += 0;
                } else {
                    reduction += std::abs(true_values[ind] - band_reduction_values[ind]) / std::abs(true_values[ind]);
                }
            }
            qr /= true_values.size();
            reduction /= true_values.size();
            total_qr += qr;
            total_reduction += reduction;
        }
        res.emplace_back(total_qr / iterations_count, total_reduction / iterations_count);
    }
    return res;
}
