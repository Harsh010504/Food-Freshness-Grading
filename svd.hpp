// One-sided Jacobi SVD (Unit I: SVD). Used for feature diagnostics / PCA-like
// analysis of the freshness feature matrix.
#pragma once
#include <algorithm>
#include <numeric>
#include "tensor.hpp"

namespace fq {

struct SVDResult {
    std::vector<double> sigma;  // singular values, descending
    Matrix V;                   // right singular vectors as columns (same order as sigma)
};

// Requires rows >= cols. Returns singular values and V.
inline SVDResult svd(const Matrix& A, int maxSweeps = 60, double tol = 1e-12) {
    if (A.r < A.c) throw std::runtime_error("svd: need rows >= cols");
    Matrix U = A;
    const size_t n = A.c, m = A.r;
    Matrix V(n, n);
    for (size_t i = 0; i < n; ++i) V(i, i) = 1.0;
    for (int sweep = 0; sweep < maxSweeps; ++sweep) {
        double off = 0;
        for (size_t p = 0; p + 1 < n; ++p)
            for (size_t q = p + 1; q < n; ++q) {
                double a = 0, b = 0, g = 0;
                for (size_t i = 0; i < m; ++i) {
                    a += U(i, p) * U(i, p);
                    b += U(i, q) * U(i, q);
                    g += U(i, p) * U(i, q);
                }
                if (a <= 0 || b <= 0) continue;
                double rel = std::fabs(g) / std::sqrt(a * b);
                if (rel < tol) continue;
                off = std::max(off, rel);
                double zeta = (b - a) / (2.0 * g);
                double t = (zeta >= 0 ? 1.0 : -1.0) / (std::fabs(zeta) + std::sqrt(1 + zeta * zeta));
                double c = 1.0 / std::sqrt(1 + t * t), s = c * t;
                for (size_t i = 0; i < m; ++i) {
                    double up = U(i, p), uq = U(i, q);
                    U(i, p) = c * up - s * uq;
                    U(i, q) = s * up + c * uq;
                }
                for (size_t i = 0; i < n; ++i) {
                    double vp = V(i, p), vq = V(i, q);
                    V(i, p) = c * vp - s * vq;
                    V(i, q) = s * vp + c * vq;
                }
            }
        if (off < tol) break;
    }
    std::vector<double> sig(n);
    for (size_t j = 0; j < n; ++j) {
        double s = 0;
        for (size_t i = 0; i < m; ++i) s += U(i, j) * U(i, j);
        sig[j] = std::sqrt(s);
    }
    std::vector<size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return sig[a] > sig[b]; });
    SVDResult res;
    res.V = Matrix(n, n);
    for (size_t k = 0; k < n; ++k) {
        res.sigma.push_back(sig[idx[k]]);
        for (size_t i = 0; i < n; ++i) res.V(i, k) = V(i, idx[k]);
    }
    return res;
}

}  // namespace fq
