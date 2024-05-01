#include "graphic_tests.h"

#include <vector>

#include "tests_performance.h"

void performance_tests_image_size(size_t kernel_height, size_t kernel_width, long double max_number) {
    std::vector<size_t> ns = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    std::vector<size_t> ms = {25, 50, 75, 100, 125, 150, 175, 200, 300, 400, 500};

    assert(ms.size() == ns.size());

    std::cout << "kernel height: " << kernel_height << " kernel width: " << kernel_width
              << " max number: " << max_number << std::endl;
    auto time_ms = tests_performance_image(ns, ms, kernel_height, kernel_width, max_number);
    std::cout << "Just QR ms:\n";
    for (size_t i = 0; i < ns.size(); i++) {
        std::cout << time_ms[i].first.first << ", ";
    }
    std::cout << std::endl;
    std::cout << "Reduction to bidiagonal time ms:\n";
    for (size_t i = 0; i < ns.size(); i++) {
        std::cout << time_ms[i].first.second << ", ";
    }
    std::cout << std::endl;
    std::cout << "Original algo time ms:\n";
    for (size_t i = 0; i < ns.size(); i++) {
        std::cout << time_ms[i].second << ", ";
    }
}

// void performance_tests_kernel_size(size_t image_height, size_t image_width) {}