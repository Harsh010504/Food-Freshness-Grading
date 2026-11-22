// Training loop with mini-batches, validation tracking and experiment helpers.
#pragma once
#include "data.hpp"
#include "mlp.hpp"

namespace fq {

struct TrainConfig {
    std::vector<size_t> hidden = {16, 8};
    Act act = Act::ReLU;
    Init init = Init::He;
    OptType opt = OptType::Nesterov;
    double lr = 0.05, momentum = 0.9, l2 = 1e-4, augmentSigma = 0.0;
    int epochs = 60; size_t batch = 32; unsigned seed = 42;
};

struct TrainResult { MLP model; double trainAcc, valAcc, trainLoss, valLoss; };

inline TrainResult train(const TrainConfig& cfg, const Matrix& Xtr, const std::vector<int>& ytr,
                         const Matrix& Xva, const std::vector<int>& yva, bool verbose = false) {
    std::vector<size_t> sizes = {Xtr.c};
    for (auto h : cfg.hidden) sizes.push_back(h);
    sizes.push_back(3);
    MLP m(sizes, cfg.act, cfg.init, cfg.seed);
    Optimizer opt(cfg.opt, cfg.lr, cfg.momentum);
    std::mt19937 rng(cfg.seed + 1);
    std::vector<size_t> idx(Xtr.r); std::iota(idx.begin(), idx.end(), 0);
    std::normal_distribution<double> noise(0, cfg.augmentSigma > 0 ? cfg.augmentSigma : 1.0);
    for (int e = 0; e < cfg.epochs; ++e) {
        std::shuffle(idx.begin(), idx.end(), rng);
        for (size_t s = 0; s < idx.size(); s += cfg.batch) {
            size_t n = std::min(cfg.batch, idx.size() - s);
            Matrix xb(n, Xtr.c); std::vector<int> yb(n);
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < Xtr.c; ++j) xb(i, j) = Xtr(idx[s + i], j) + (cfg.augmentSigma > 0 ? noise(rng) : 0.0);
                yb[i] = ytr[idx[s + i]];
            }
            std::vector<Matrix> cache, gW, gb;
            m.forward(xb, &cache);
            m.backward(cache, yb, cfg.l2, gW, gb);
            opt.step(m, gW, gb);
        }
        if (verbose && (e % 10 == 0 || e + 1 == cfg.epochs)) {
            std::cout << "  epoch " << e << "  train_loss=" << m.loss(m.forward(Xtr), ytr, cfg.l2)
                      << "  val_acc=" << evaluate(m.predict(Xva), yva).acc << "\n";
        }
    }
    TrainResult r{m, 0, 0, 0, 0};
    r.trainAcc = evaluate(m.predict(Xtr), ytr).acc;
    r.valAcc = evaluate(m.predict(Xva), yva).acc;
    r.trainLoss = m.loss(m.forward(Xtr), ytr, cfg.l2);
    r.valLoss = m.loss(m.forward(Xva), yva, cfg.l2);
    return r;
}

}  // namespace fq
