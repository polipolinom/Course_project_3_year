#include <gtest/gtest.h>

#include <iostream>

#include "dumb_test.h"
#include "graphic_tests.h"

int main(int argc, char **argv) {
    /*::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
    */
    size_t kernel_height = 1;
    size_t kernel_width = 20;
    long double max_number = 1e5;

    performance_tests_image_size(kernel_height, kernel_width, max_number);
}

// 1 3
// Just QR ms:
// 0, 4, 54, 233, 538, 797, 968, 1486, 2002, 2404, 6112, 11160, 16531
// Reduction to bidiagonal time ms:
// 0, 0, 2, 14, 56, 147, 295, 626, 1147, 1798, 9644, 78132
// Original algo time ms:
// 0, 0, 0, 23, 88, 208, 377, 779, 1370, 12750, 88296
