// Minimal tensor / matrix sandbox (Unit II): creation, reshape, broadcasting,
// inner products, matmul, transpose. No external dependencies.
#pragma once
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace fq {

struct Matrix {
    size_t r = 0, c = 0;
    std::vector<double> d;
    Matrix() = default;
    Matrix(size_t rows, size_t cols, double v = 0.0) : r(rows), c(cols), d(rows * cols, v) {}
    double& operator()(size_t i, size_t j) { return d[i * c + j]; }
    double operator()(size_t i, size_t j) const { return d[i * c + j]; }
    size_t size() const { return d.size(); }

    Matrix T() const {
        Matrix t(c, r);
        for (size_t i = 0; i < r; ++i)
            for (size_t j = 0; j < c; ++j) t(j, i) = (*this)(i, j);
        return t;
    }
    // Tensor reshaping (row-major, total element count must match).
    Matrix reshape(size_t nr, size_t nc) const {
        if (nr * nc != d.size()) throw std::runtime_error("reshape: size mismatch");
        Matrix m(nr, nc);
        m.d = d;
        return m;
    }
};

inline Matrix matmul(const Matrix& a, const Matrix& b) {
    if (a.c != b.r) throw std::runtime_error("matmul: shape mismatch");
    Matrix o(a.r, b.c);
    for (size_t i = 0; i < a.r; ++i)
        for (size_t k = 0; k < a.c; ++k) {
            double v = a(i, k);
            for (size_t j = 0; j < b.c; ++j) o(i, j) += v * b(k, j);
        }
    return o;
}

// Broadcasting: add a 1 x C row vector to every row of an N x C matrix.
inline Matrix addRowBroadcast(const Matrix& a, const Matrix& row) {
    if (row.r != 1 || row.c != a.c) throw std::runtime_error("broadcast: shape mismatch");
    Matrix o = a;
    for (size_t i = 0; i < a.r; ++i)
        for (size_t j = 0; j < a.c; ++j) o(i, j) += row(0, j);
    return o;
}

inline Matrix colSum(const Matrix& a) {
    Matrix s(1, a.c);
    for (size_t i = 0; i < a.r; ++i)
        for (size_t j = 0; j < a.c; ++j) s(0, j) += a(i, j);
    return s;
}

inline Matrix hadamard(const Matrix& a, const Matrix& b) {
    if (a.r != b.r || a.c != b.c) throw std::runtime_error("hadamard: shape mismatch");
    Matrix o = a;
    for (size_t i = 0; i < o.d.size(); ++i) o.d[i] *= b.d[i];
    return o;
}

// Inner product of two same-size tensors (flattened).
inline double innerProduct(const Matrix& a, const Matrix& b) {
    if (a.d.size() != b.d.size()) throw std::runtime_error("inner: size mismatch");
    double s = 0;
    for (size_t i = 0; i < a.d.size(); ++i) s += a.d[i] * b.d[i];
    return s;
}

inline void print(const Matrix& m, const char* name, int maxRows = 6) {
    std::cout << name << " (" << m.r << "x" << m.c << ")\n";
    for (size_t i = 0; i < m.r && (int)i < maxRows; ++i) {
        std::cout << "  ";
        for (size_t j = 0; j < m.c; ++j) std::cout << m(i, j) << (j + 1 < m.c ? ", " : "");
        std::cout << "\n";
    }
}

}  // namespace fq
