#include <gtest/gtest.h>

#include <iostream>

#include "../src/algorithms/constants.h"
#include "dumb_test.h"
#include "graphic_tests.h"
#include "precision_tests.h"

int it;

int main(int argc, char **argv) {
    /*::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
    */
    it = 0;
    freopen("input.txt", "r", stdin);
    int C_in, C_out, kernel_width, m;
    std::cin >> C_in >> C_out >> kernel_width >> m;
    std::vector<std::vector<Matrix<long double>>> kernels(C_in, std::vector<Matrix<long double>>(C_out));
    for (size_t j = 0; j < C_in; ++j) {
        for (size_t k = 0; k < C_out; ++k) {
            Matrix<long double> now(1, kernel_width);
            for (size_t i = 0; i < kernel_width; ++i) {
                std::cin >> now(0, i);
            }
            kernels[j][k] = now;
        }
    }

    auto start = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
    svd_convolution_1d(kernels, m, nullptr, nullptr, 1, false);
    auto finish = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
    std::cout << finish - start;
    // size_t kernel_width = 64;
    // size_t C_in = 1;
    // size_t C_out = 16;
    // performance_tests_image_size(kernel_width, C_in, C_out, 1.0);
    // size_t image_height = 1;
    // size_t image_width = 500;
    // long double max_number = 1;
    // std::vector<long double> epsilons = {1e-6, 1e-8, 1e-10, 1e-12, 1e-14, 1e-16, 1e-18};

    // tests_precision_epsilons(epsilons, image_height, image_width, kernel_height, kernel_width, max_number);
}