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
    size_t kernel_width = 5;
    size_t image_height = 1;
    size_t image_width = 10;
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

// 1 3
// Just QR ms:
// 0, 4, 54, 233, 538, 797, 968, 1486, 2002, 2404, 6112, 11160, 16531
// Reduction to bidiagonal time ms:
// 0, 0, 2, 14, 56, 147, 295, 626, 1147, 1798, 9644, 78132
// Original algo time ms:
// 0, 0, 0, 23, 88, 208, 377, 779, 1370, 12750, 88296
