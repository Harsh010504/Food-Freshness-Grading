// From-scratch MLP (Unit I/III): forward pass, softmax cross-entropy, backprop,
// SGD / Momentum / Nesterov optimizers, initialization schemes, L2 regularization.
#pragma once
#include <algorithm>
#include <fstream>
#include <random>
#include <sstream>
#include "tensor.hpp"

namespace fq {

enum class Act { ReLU, Sigmoid, Tanh };
enum class Init { Zero, SmallNormal, Xavier, He };
enum class OptType { SGD, Momentum, Nesterov };

inline const char* name(Act a) { return a == Act::ReLU ? "relu" : a == Act::Sigmoid ? "sigmoid" : "tanh"; }
inline const char* name(Init i) {
    return i == Init::Zero ? "zero" : i == Init::SmallNormal ? "small-normal" : i == Init::Xavier ? "xavier" : "he";
}
inline const char* name(OptType o) { return o == OptType::SGD ? "sgd" : o == OptType::Momentum ? "momentum" : "nesterov"; }

struct MLP {
    std::vector<size_t> sizes;
    Act act = Act::ReLU;
    std::vector<Matrix> W, b;  // W[l]: in x out ; b[l]: 1 x out

    MLP() = default;
    MLP(std::vector<size_t> s, Act a, Init init, unsigned seed) : sizes(std::move(s)), act(a) {
        std::mt19937 rng(seed);
        for (size_t l = 0; l + 1 < sizes.size(); ++l) {
            size_t in = sizes[l], out = sizes[l + 1];
            Matrix w(in, out);
            double stdv = 0;
            switch (init) {
                case Init::Zero: stdv = 0; break;
                case Init::SmallNormal: stdv = 0.01; break;
                case Init::Xavier: stdv = std::sqrt(2.0 / (in + out)); break;
                case Init::He: stdv = std::sqrt(2.0 / in); break;
            }
            std::normal_distribution<double> nd(0.0, stdv > 0 ? stdv : 1.0);
            if (stdv > 0) for (auto& v : w.d) v = nd(rng);
            W.push_back(w);
            b.push_back(Matrix(1, out));
        }
    }

    double f(double z) const {
        switch (act) {
            case Act::ReLU: return z > 0 ? z : 0;
            case Act::Sigmoid: return 1.0 / (1.0 + std::exp(-z));
            default: return std::tanh(z);
        }
    }
    // derivative expressed through the activation output a = f(z)
    double df(double a) const {
        switch (act) {
            case Act::ReLU: return a > 0 ? 1.0 : 0.0;
            case Act::Sigmoid: return a * (1 - a);
            default: return 1 - a * a;
        }
    }

    static Matrix softmax(const Matrix& z) {
        Matrix p = z;
        for (size_t i = 0; i < z.r; ++i) {
            double mx = z(i, 0);
            for (size_t j = 1; j < z.c; ++j) mx = std::max(mx, z(i, j));
            double s = 0;
            for (size_t j = 0; j < z.c; ++j) s += (p(i, j) = std::exp(z(i, j) - mx));
            for (size_t j = 0; j < z.c; ++j) p(i, j) /= s;
        }
        return p;
    }

    // cache[0] = input, cache[l] = activation after layer l (last = probabilities).
    Matrix forward(const Matrix& X, std::vector<Matrix>* cache = nullptr) const {
        Matrix a = X;
        if (cache) { cache->clear(); cache->push_back(a); }
        for (size_t l = 0; l < W.size(); ++l) {
            Matrix z = addRowBroadcast(matmul(a, W[l]), b[l]);
            if (l + 1 == W.size()) a = softmax(z);
            else { for (auto& v : z.d) v = f(v); a = z; }
            if (cache) cache->push_back(a);
        }
        return a;
    }

    double loss(const Matrix& P, const std::vector<int>& y, double l2) const {
        double L = 0;
        for (size_t i = 0; i < P.r; ++i) L -= std::log(std::max(P(i, (size_t)y[i]), 1e-12));
        L /= (double)P.r;
        double reg = 0;
        for (auto& w : W) for (double v : w.d) reg += v * v;
        return L + 0.5 * l2 * reg;
    }

    void backward(const std::vector<Matrix>& cache, const std::vector<int>& y, double l2,
                  std::vector<Matrix>& gW, std::vector<Matrix>& gb) const {
        const size_t L = W.size(), N = cache[0].r;
        gW.assign(L, Matrix()); gb.assign(L, Matrix());
        Matrix dZ = cache[L];  // probabilities
        for (size_t i = 0; i < N; ++i) dZ(i, (size_t)y[i]) -= 1.0;
        for (auto& v : dZ.d) v /= (double)N;
        for (size_t l = L; l-- > 0;) {
            gW[l] = matmul(cache[l].T(), dZ);
            for (size_t k = 0; k < gW[l].d.size(); ++k) gW[l].d[k] += l2 * W[l].d[k];
            gb[l] = colSum(dZ);
            if (l > 0) {
                Matrix dA = matmul(dZ, W[l].T());
                for (size_t k = 0; k < dA.d.size(); ++k) dA.d[k] *= df(cache[l].d[k]);
                dZ = dA;
            }
        }
    }

    std::vector<int> predict(const Matrix& X) const {
        Matrix P = forward(X);
        std::vector<int> out(P.r);
        for (size_t i = 0; i < P.r; ++i) {
            size_t best = 0;
            for (size_t j = 1; j < P.c; ++j) if (P(i, j) > P(i, best)) best = j;
            out[i] = (int)best;
        }
        return out;
    }

    void save(std::ostream& os) const {
        os << sizes.size();
        for (auto s : sizes) os << ' ' << s;
        os << ' ' << (int)act << '\n';
        os.precision(17);
        for (size_t l = 0; l < W.size(); ++l) {
            for (double v : W[l].d) os << v << ' ';
            os << '\n';
            for (double v : b[l].d) os << v << ' ';
            os << '\n';
        }
    }
    static MLP load(std::istream& is) {
        MLP m; size_t n; is >> n; m.sizes.resize(n);
        for (auto& s : m.sizes) is >> s;
        int a; is >> a; m.act = (Act)a;
        for (size_t l = 0; l + 1 < n; ++l) {
            Matrix w(m.sizes[l], m.sizes[l + 1]), bb(1, m.sizes[l + 1]);
            for (auto& v : w.d) is >> v;
            for (auto& v : bb.d) is >> v;
            m.W.push_back(w); m.b.push_back(bb);
        }
        return m;
    }
};

struct Optimizer {
    OptType type; double lr, mu;
    std::vector<Matrix> vW, vb;
    Optimizer(OptType t, double lr_, double mu_ = 0.9) : type(t), lr(lr_), mu(mu_) {}

    void step(MLP& m, const std::vector<Matrix>& gW, const std::vector<Matrix>& gb) {
        if (vW.empty())
            for (size_t l = 0; l < m.W.size(); ++l) {
                vW.push_back(Matrix(m.W[l].r, m.W[l].c));
                vb.push_back(Matrix(m.b[l].r, m.b[l].c));
            }
        for (size_t l = 0; l < m.W.size(); ++l) {
            apply(m.W[l], gW[l], vW[l]);
            apply(m.b[l], gb[l], vb[l]);
        }
    }
    void apply(Matrix& p, const Matrix& g, Matrix& v) {
        for (size_t k = 0; k < p.d.size(); ++k) {
            if (type == OptType::SGD) p.d[k] -= lr * g.d[k];
            else if (type == OptType::Momentum) { v.d[k] = mu * v.d[k] - lr * g.d[k]; p.d[k] += v.d[k]; }
            else {  // Nesterov (look-ahead form)
                double vp = v.d[k];
                v.d[k] = mu * v.d[k] - lr * g.d[k];
                p.d[k] += -mu * vp + (1 + mu) * v.d[k];
            }
        }
    }
};

// Numerical gradient check: max relative error between analytic and central-difference gradients.
inline double gradientCheck(MLP& m, const Matrix& X, const std::vector<int>& y, double l2, double eps = 1e-5) {
    std::vector<Matrix> cache, gW, gb;
    m.forward(X, &cache);
    m.backward(cache, y, l2, gW, gb);
    double worst = 0;
    for (size_t l = 0; l < m.W.size(); ++l)
        for (size_t k = 0; k < m.W[l].d.size(); ++k) {
            double old = m.W[l].d[k];
            m.W[l].d[k] = old + eps; double lp = m.loss(m.forward(X), y, l2);
            m.W[l].d[k] = old - eps; double lm = m.loss(m.forward(X), y, l2);
            m.W[l].d[k] = old;
            double num = (lp - lm) / (2 * eps), ana = gW[l].d[k];
            double rel = std::fabs(num - ana) / std::max(1e-8, std::fabs(num) + std::fabs(ana));
            worst = std::max(worst, rel);
        }
    return worst;
}

}  // namespace fq
