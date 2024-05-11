#include "graphic_tests.h"

#include <vector>

#include "tests_performance.h"

void performance_tests_image_size(size_t kernel_width, size_t C_in, size_t C_out, long double max_number) {
    std::vector<size_t> ms = {100, 500, 1000, 2500, 5000, 7500, 10000, 12500, 15000, 16000};

    std::cout << " kernel width: " << kernel_width << " C_in: " << C_in << " C_out: " << C_out
              << " max number: " << max_number << std::endl;
    auto time_ms = tests_performance_image(ms, kernel_width, C_in, C_out, max_number);
    std::cout << "Band reduction ms:\n";
    for (size_t i = 0; i < ms.size(); i++) {
        std::cout << time_ms[i].first << ", ";
    }
    std::cout << std::endl;
    std::cout << "Original algo time ms:\n";
    for (size_t i = 0; i < ms.size(); i++) {
        std::cout << time_ms[i].second << ", ";
    }
}

// void performance_tests_kernel_size(size_t image_height, size_t image_width) {}