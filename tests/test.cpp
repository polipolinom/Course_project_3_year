#include <gtest/gtest.h>

#include <iostream>

#include "../src/algorithms/constants.h"
#include "dumb_test.h"
#include "graphic_tests.h"
#include "precision_tests.h"


int main(int argc, char **argv) {
    size_t kernel_width = 10;
    size_t C_in = 4;
    size_t C_out = 4;
    performance_tests_image_size(kernel_width, C_in, C_out, 1.0);
}