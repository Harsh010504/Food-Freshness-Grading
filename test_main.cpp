#include <cstdlib>
#include <iostream>
#include "../include/data.hpp"
#include "../include/learning.hpp"
#include "../include/svd.hpp"
#include "../include/train.hpp"
using namespace fq;

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { std::cerr << "FAIL: " #cond " (line " << __LINE__ << ")\n"; ++failures; } } while (0)

int main() {
    // tensor ops
    Matrix a(2, 3); for (size_t i = 0; i < 6; ++i) a.d[i] = (double)i;
    CHECK(a.T().r == 3 && a.T()(2, 1) == 5);
    CHECK(matmul(a, a.T())(1, 1) == 9 + 16 + 25);
    Matrix row(1, 3); row.d = {1, 1, 1};
    CHECK(addRowBroadcast(a, row)(1, 2) == 6);
    CHECK(a.reshape(3, 2)(2, 1) == 5);
    CHECK(innerProduct(a, a) == 55);

    // SVD: sigma^2 sum equals Frobenius norm^2, and sigma sorted
    Dataset d = makeSynthetic(120, 3);
    Scaler sc; sc.fit(d.X); Matrix X = sc.apply(d.X);
    SVDResult s = svd(X);
    double fro = innerProduct(X, X), sum = 0;
    for (double v : s.sigma) sum += v * v;
    CHECK(std::fabs(fro - sum) / fro < 1e-8);
    for (size_t i = 1; i < s.sigma.size(); ++i) CHECK(s.sigma[i - 1] >= s.sigma[i]);

    // gradient check for every activation
    for (Act act : {Act::Sigmoid, Act::Tanh, Act::ReLU}) {
        MLP m({kFeatures, 5, 3}, act, Init::Xavier, 11);
        CHECK(gradientCheck(m, X, d.y, 1e-3) < 1e-5);
    }

    // training learns above chance, optimizers all converge
    Split sp = split(makeSynthetic(900, 7), 42);
    Scaler sc2; sc2.fit(sp.train.X);
    for (OptType o : {OptType::SGD, OptType::Momentum, OptType::Nesterov}) {
        TrainConfig c; c.opt = o;
        TrainResult r = train(c, sc2.apply(sp.train.X), sp.train.y, sc2.apply(sp.val.X), sp.val.y);
        CHECK(r.valAcc > 0.7);
    }
    // reproducibility: same seed -> identical weights
    TrainConfig c;
    auto r1 = train(c, sc2.apply(sp.train.X), sp.train.y, sc2.apply(sp.val.X), sp.val.y);
    auto r2 = train(c, sc2.apply(sp.train.X), sp.train.y, sc2.apply(sp.val.X), sp.val.y);
    CHECK(r1.model.W[0].d == r2.model.W[0].d);

    // no train/test leakage: sizes add up
    CHECK(sp.train.X.r + sp.val.X.r + sp.test.X.r == 900);

    // Oja rule converges to unit norm
    auto w = ojaRule(sc2.apply(sp.train.X), 0.002, 30, 1);
    double n = 0; for (double v : w) n += v * v;
    CHECK(std::fabs(std::sqrt(n) - 1.0) < 0.1);

    if (failures == 0) std::cout << "ALL TESTS PASSED\n";
    return failures == 0 ? 0 : 1;
}
