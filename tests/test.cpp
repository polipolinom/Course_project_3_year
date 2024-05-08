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
    size_t kernel_height = 1;
    size_t kernel_width = 20;
    // performance_tests_image_size(kernel_height, kernel_width, 1.0);
    size_t image_height = 1;
    size_t image_width = 50;
    long double max_number = 1;
    std::vector<long double> epsilons = {1e-6, 1e-8, 1e-10, 1e-12, 1e-14, 1e-16, 1e-18};

    auto res = tests_precision_epsilons(epsilons, image_height, image_width, kernel_height, kernel_width, max_number);
    std::cout << "Banded qr:\n";
    for (auto [x, y] : res) {
        std::cout << x << " ";
    }
    std::cout << "\nBand reduction: \n";
    for (auto [x, y] : res) {
        std::cout << y << " ";
    }
}