#pragma once

#include <vector>

#include "../course-project-second-year/types/matrix.h"
#include "../course-project-second-year/types/vector.h"
#include "../utils/conv_matrix.h"
#include "householder_reflection.h"
#include "svd_banded.h"

namespace convolution_svd {
using namespace svd_computation;
std::vector<long double> svd_convolution_1d(std::vector<std::vector<Matrix<long double>>> kernels,
                                            const size_t signal_size, Matrix<long double>* left_basis = nullptr,
                                            Matrix<long double>* right_basis = nullptr, const size_t stride = 1,
                                            bool full_basis = false,
                                            const long double eps = constants::DEFAULT_EPSILON) {
    assert(signal_size > 0);
    size_t C_in = kernels.size();
    assert(C_in > 0);
    size_t C_out = kernels[0].size();
    assert(C_out > 0);
    size_t kernel_size = kernels[0][0].width();

    if (full_basis) {
        if (left_basis) {
            (*left_basis) = Matrix<long double>::identity(C_out * (signal_size - kernel_size + 1));
        }
        if (right_basis) {
            (*right_basis) = Matrix<long double>::identity(signal_size * C_in);
        }
    } else {
        if (left_basis) {
            (*left_basis) = Matrix<long double>(C_out, C_out * (signal_size - kernel_size + 1));
            for (size_t i = 0; i < C_out; ++i) {
                (*left_basis)(i, i) = 1;
            }
        }
        if (right_basis) {
            (*right_basis) = Matrix<long double>(signal_size * C_in, kernel_size * C_in);
            for (size_t i = 0; i < kernel_size * C_in; ++i) {
                (*right_basis)(i, i) = 1;
            }
        }
    }
    if (C_in == 1 && C_out == 1) {
        auto kernel = kernels[0][0];
        Matrix<long double> A(signal_size - kernel_size + 1, kernel_size);
        for (size_t i = 0; i < A.height(); i += stride) {
            for (size_t j = 0; j < A.width(); ++j) {
                A(i, j + i) = kernel(0, j);
            }
        }
        auto values = svd_banded_reduction(A, left_basis, right_basis, eps);
        if (left_basis) {
            Matrix<long double> new_left_basis(left_basis->height(), (signal_size - kernel_size + stride) / stride);
            for (size_t i = 0; i < new_left_basis.height(); ++i) {
                for (size_t j = 0; j < new_left_basis.width(); ++j) {
                    new_left_basis(i, j) = (*left_basis)(i, j);
                }
            }
            *left_basis = new_left_basis;
        }
        while (values.size() > (signal_size - kernel_size + stride) / stride) {
            values.pop_back();
        }
        return values;
    }
    if (C_out <= C_in) {
        Matrix<long double> A(C_out * (signal_size - kernel_size + 1), C_in * kernel_size);
        for (size_t i = 0; i < signal_size - kernel_size + 1; i += stride) {
            for (size_t j = 0; j < kernel_size; ++j) {
                for (size_t c_in = 0; c_in < C_in; ++c_in) {
                    for (size_t c_out = 0; c_out < C_out; ++c_out) {
                        A(i * C_out + c_out, j * C_in + c_in) = kernels[c_in][c_out](0, j);
                    }
                }
            }
        }

        for (size_t i = 0; i < signal_size - kernel_size + 1; i += stride) {
            for (size_t j = 0; j < C_out - 1; ++j) {
                auto v = left_segment_reflection(A, C_out * i + j, C_out * (i + 1) - 1, j, true, 1e-20);
                if (left_basis) {
                    mult_right_segment_reflection(*left_basis, v, C_out * i + j, C_out * (i + 1) - 1);
                }
            }
            for (size_t j = 0; j < C_out; ++j) {
                auto v = right_segment_reflection(A, C_out * i + j, C_in * (kernel_size - 1) + j,
                                                  C_in * (kernel_size)-1, false, 1e-20);
                if (right_basis) {
                    mult_left_segment_reflection(*right_basis, v, C_in * (i + kernel_size - 1) + j,
                                                 C_in * (i + kernel_size) - 1);
                }

                Vector<long double> tmp(C_out * std::min(kernel_size, signal_size - kernel_size + 1 - i));
                for (size_t ind = 0; ind < tmp.size(); ++ind) {
                    size_t start_column = C_in * (kernel_size - 1 - ind / C_out) + j;
                    size_t end_column = C_in * (kernel_size - ind / C_out) - 1;
                    for (size_t j = start_column; j <= end_column; ++j) {
                        tmp[ind] += A(C_out * i + ind, j) * v(0, j - start_column);
                    }
                }

                for (size_t ind = 0; ind < tmp.size(); ++ind) {
                    size_t start_column = C_in * (kernel_size - 1 - ind / C_out) + j;
                    size_t end_column = C_in * (kernel_size - ind / C_out) - 1;
                    for (size_t k = start_column; k <= end_column; ++k) {
                        A(ind + C_out * i, k) -= 2.0 * tmp[ind] * v(0, k - start_column);
                    }
                }
            }
        }

        size_t band_width = C_in * signal_size - (C_in - C_out) - A.height() + 1;
        Matrix<long double> upper_band(A.height(), band_width);

        for (size_t i = 0; i < signal_size - kernel_size + 1; ++i) {
            for (size_t j = 0; j < C_out; ++j) {
                for (size_t k = j; k <= C_in * (kernel_size - 1) + j; ++k) {
                    upper_band(i * C_out + j, (C_in - C_out) * i + k - j) = A(i * C_out + j, k);
                }
            }
        }

        auto values = svd_banded_reduction(upper_band, left_basis, right_basis, eps);
        if (left_basis) {
            Matrix<long double> new_left_basis(left_basis->height(),
                                               C_out * ((signal_size - kernel_size + stride) / stride));
            for (size_t i = 0; i < new_left_basis.height(); ++i) {
                for (size_t j = 0; j < new_left_basis.width(); ++j) {
                    new_left_basis(i, j) = (*left_basis)(i, j);
                }
            }
            *left_basis = new_left_basis;
        }
        while (values.size() > C_out * ((signal_size - kernel_size + stride) / stride)) {
            values.pop_back();
        }
        return values;
    }

    size_t up_band_size = C_in * kernel_size;
    size_t down_band_size = C_out * (signal_size - kernel_size + 1) - C_in * (signal_size - kernel_size) - 1;
    Matrix<long double> A(C_out * (signal_size - kernel_size + 1), up_band_size + down_band_size);
    for (size_t i = 0; i < signal_size - kernel_size + 1; i += stride) {
        for (size_t j = 0; j < kernel_size; ++j) {
            for (size_t c_in = 0; c_in < C_in; ++c_in) {
                for (size_t c_out = 0; c_out < C_out; ++c_out) {
                    size_t row = i * C_out + c_out;
                    size_t column = (j + i) * C_in + c_in;
                    if (column >= row) {
                        A(row, column - row + down_band_size) = kernels[c_in][c_out](0, j);
                    } else {
                        A(row, down_band_size - (row - column)) = kernels[c_in][c_out](0, j);
                    }
                }
            }
        }
    }

    band_down_diag_reduction(A, down_band_size, up_band_size, left_basis, right_basis, eps);
    Matrix<long double> upper_band(std::min(A.height(), C_in * signal_size), up_band_size);
    for (size_t i = 0; i < upper_band.height(); ++i) {
        for (size_t j = 0; j < up_band_size; ++j) {
            upper_band(i, j) = A(i, j + down_band_size);
        }
    }
    auto values = svd_banded_reduction(upper_band, left_basis, right_basis, eps);
    if (left_basis) {
        Matrix<long double> new_left_basis(left_basis->height(),
                                           C_out * ((signal_size - kernel_size + stride) / stride));
        for (size_t i = 0; i < new_left_basis.height(); ++i) {
            for (size_t j = 0; j < new_left_basis.width(); ++j) {
                new_left_basis(i, j) = (*left_basis)(i, j);
            }
        }
        *left_basis = new_left_basis;
    }
    while (values.size() > C_out * ((signal_size - kernel_size + stride) / stride)) {
        values.pop_back();
    }
    return values;
}
}  // namespace convolution_svd