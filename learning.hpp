// Unit III learning rules: Hebbian (Oja), competitive (winner-take-all) and a tiny
// RBM trained with contrastive divergence (energy-based / Boltzmann concept).
#pragma once
#include <random>
#include "tensor.hpp"

namespace fq {

// Oja's rule: normalized Hebbian learning; weight vector converges to the first principal component.
inline std::vector<double> ojaRule(const Matrix& Xc, double lr, int epochs, unsigned seed) {
    std::mt19937 rng(seed); std::normal_distribution<double> nd(0, 0.1);
    std::vector<double> w(Xc.c);
    for (auto& v : w) v = nd(rng);
    for (int e = 0; e < epochs; ++e)
        for (size_t i = 0; i < Xc.r; ++i) {
            double y = 0;
            for (size_t j = 0; j < Xc.c; ++j) y += w[j] * Xc(i, j);
            for (size_t j = 0; j < Xc.c; ++j) w[j] += lr * y * (Xc(i, j) - y * w[j]);  // Hebb + decay
        }
    return w;
}

// Competitive learning: k prototypes, only the winner moves toward the input.
inline Matrix competitive(const Matrix& X, size_t k, double lr, int epochs, unsigned seed) {
    std::mt19937 rng(seed);
    Matrix P(k, X.c);
    std::uniform_int_distribution<size_t> pick(0, X.r - 1);
    for (size_t c = 0; c < k; ++c) for (size_t j = 0; j < X.c; ++j) P(c, j) = X(pick(rng), j);
    for (int e = 0; e < epochs; ++e)
        for (size_t i = 0; i < X.r; ++i) {
            size_t w = 0; double best = 1e300;
            for (size_t c = 0; c < k; ++c) {
                double d = 0;
                for (size_t j = 0; j < X.c; ++j) d += (X(i, j) - P(c, j)) * (X(i, j) - P(c, j));
                if (d < best) { best = d; w = c; }
            }
            for (size_t j = 0; j < X.c; ++j) P(w, j) += lr * (X(i, j) - P(w, j));
        }
    return P;
}

inline size_t nearest(const Matrix& P, const Matrix& X, size_t i) {
    size_t w = 0; double best = 1e300;
    for (size_t c = 0; c < P.r; ++c) {
        double d = 0;
        for (size_t j = 0; j < X.c; ++j) d += (X(i, j) - P(c, j)) * (X(i, j) - P(c, j));
        if (d < best) { best = d; w = c; }
    }
    return w;
}

// Tiny binary RBM with CD-1. Returns reconstruction error before and after training.
inline std::pair<double, double> rbmDemo(const Matrix& Xbin, size_t hidden, int epochs, double lr, unsigned seed) {
    std::mt19937 rng(seed); std::normal_distribution<double> nd(0, 0.05);
    std::uniform_real_distribution<double> u(0, 1);
    size_t nv = Xbin.c;
    Matrix W(nv, hidden), bv(1, nv), bh(1, hidden);
    for (auto& v : W.d) v = nd(rng);
    auto sig = [](double z) { return 1.0 / (1.0 + std::exp(-z)); };
    auto recErr = [&]() {
        double e = 0;
        for (size_t i = 0; i < Xbin.r; ++i) {
            std::vector<double> h(hidden);
            for (size_t k = 0; k < hidden; ++k) { double z = bh(0, k); for (size_t j = 0; j < nv; ++j) z += Xbin(i, j) * W(j, k); h[k] = sig(z); }
            for (size_t j = 0; j < nv; ++j) { double z = bv(0, j); for (size_t k = 0; k < hidden; ++k) z += h[k] * W(j, k); double d = sig(z) - Xbin(i, j); e += d * d; }
        }
        return e / (double)(Xbin.r * nv);
    };
    double before = recErr();
    for (int ep = 0; ep < epochs; ++ep)
        for (size_t i = 0; i < Xbin.r; ++i) {
            std::vector<double> h0(hidden), hs(hidden), v1(nv), h1(hidden);
            for (size_t k = 0; k < hidden; ++k) { double z = bh(0, k); for (size_t j = 0; j < nv; ++j) z += Xbin(i, j) * W(j, k); h0[k] = sig(z); hs[k] = u(rng) < h0[k] ? 1.0 : 0.0; }
            for (size_t j = 0; j < nv; ++j) { double z = bv(0, j); for (size_t k = 0; k < hidden; ++k) z += hs[k] * W(j, k); v1[j] = sig(z); }
            for (size_t k = 0; k < hidden; ++k) { double z = bh(0, k); for (size_t j = 0; j < nv; ++j) z += v1[j] * W(j, k); h1[k] = sig(z); }
            for (size_t j = 0; j < nv; ++j) for (size_t k = 0; k < hidden; ++k) W(j, k) += lr * (Xbin(i, j) * h0[k] - v1[j] * h1[k]);
            for (size_t j = 0; j < nv; ++j) bv(0, j) += lr * (Xbin(i, j) - v1[j]);
            for (size_t k = 0; k < hidden; ++k) bh(0, k) += lr * (h0[k] - h1[k]);
        }
    return {before, recErr()};
}

}  // namespace fq
