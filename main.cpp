// Food Quality & Freshness Grading System - CLI
// Usage: freshness <math|train|compare|learning|predict|demo> [options]
#include <cstring>
#include <iomanip>
#include "../include/data.hpp"
#include "../include/learning.hpp"
#include "../include/svd.hpp"
#include "../include/train.hpp"

using namespace fq;

static std::string arg(int argc, char** argv, const std::string& key, const std::string& def = "") {
    for (int i = 2; i + 1 < argc; ++i) if (key == argv[i]) return argv[i + 1];
    return def;
}

static Dataset loadData(int argc, char** argv) {
    std::string csv = arg(argc, argv, "--csv");
    if (!csv.empty()) return loadCsv(csv);
    return makeSynthetic((size_t)std::stoi(arg(argc, argv, "--n", "900")), 7);
}

static void printMetrics(const char* title, const Metrics& m) {
    std::cout << title << ": accuracy=" << std::fixed << std::setprecision(3) << m.acc
              << "  macroF1=" << m.macroF1 << "\n  confusion (rows=true, cols=pred) ";
    for (auto* n : kClassNames) std::cout << n << " ";
    std::cout << "\n";
    for (int i = 0; i < 3; ++i) {
        std::cout << "    " << std::setw(8) << kClassNames[i] << ": ";
        for (int j = 0; j < 3; ++j) std::cout << std::setw(5) << m.cm[i][j];
        std::cout << "\n";
    }
}

// ---- Unit II: tensor sandbox + gradient check + SVD ----
static int cmdMath() {
    std::cout << "== Tensor sandbox ==\n";
    Matrix a(2, 3), row(1, 3);
    for (size_t i = 0; i < 6; ++i) a.d[i] = (double)i + 1;
    row.d = {10, 20, 30};
    print(a, "A");
    print(addRowBroadcast(a, row), "A + row (broadcast)");
    print(a.reshape(3, 2), "A reshaped to 3x2");
    print(matmul(a, a.T()), "A * A^T");
    std::cout << "inner(A,A) = " << innerProduct(a, a) << "\n";

    std::cout << "\n== Gradient check (sigmoid MLP, 5-6-3) ==\n";
    Dataset d = makeSynthetic(30, 1);
    Scaler sc; sc.fit(d.X);
    MLP m({kFeatures, 6, 3}, Act::Sigmoid, Init::Xavier, 3);
    std::cout << "max relative error = " << std::scientific << gradientCheck(m, sc.apply(d.X), d.y, 1e-3)
              << " (should be < 1e-6)\n" << std::fixed;

    std::cout << "\n== SVD of standardized feature matrix ==\n";
    SVDResult s = svd(sc.apply(makeSynthetic(300, 2).X));
    double tot = 0; for (double v : s.sigma) tot += v * v;
    for (size_t i = 0; i < s.sigma.size(); ++i)
        std::cout << "  sigma" << i + 1 << " = " << std::setprecision(4) << s.sigma[i]
                  << "  explained variance = " << 100.0 * s.sigma[i] * s.sigma[i] / tot << "%\n";
    return 0;
}

// ---- Unit III: controlled experiments ----
static int cmdCompare(int argc, char** argv) {
    Dataset all = loadData(argc, argv);
    Split sp = split(all, 42);
    Scaler sc; sc.fit(sp.train.X);
    Matrix Xtr = sc.apply(sp.train.X), Xva = sc.apply(sp.val.X);
    auto row = [&](const std::string& label, TrainConfig c) {
        TrainResult r = train(c, Xtr, sp.train.y, Xva, sp.val.y);
        std::cout << "  " << std::left << std::setw(22) << label << std::right
                  << " train_acc=" << r.trainAcc << " val_acc=" << r.valAcc
                  << " val_loss=" << r.valLoss << " gap=" << r.trainAcc - r.valAcc << "\n";
    };
    std::cout << std::fixed << std::setprecision(3) << "== Optimizers ==\n";
    for (OptType o : {OptType::SGD, OptType::Momentum, OptType::Nesterov}) { TrainConfig c; c.opt = o; row(name(o), c); }
    std::cout << "== Initialization ==\n";
    for (Init i : {Init::Zero, Init::SmallNormal, Init::Xavier, Init::He}) { TrainConfig c; c.init = i; row(name(i), c); }
    std::cout << "== Activation ==\n";
    for (Act a : {Act::ReLU, Act::Sigmoid, Act::Tanh}) { TrainConfig c; c.act = a; row(name(a), c); }
    std::cout << "== L2 regularization ==\n";
    for (double l : {0.0, 1e-4, 1e-2}) { TrainConfig c; c.l2 = l; row("l2=" + std::to_string(l), c); }
    std::cout << "== Data augmentation (feature jitter) ==\n";
    for (double s : {0.0, 0.1, 0.3}) { TrainConfig c; c.augmentSigma = s; row("sigma=" + std::to_string(s), c); }
    std::cout << "== Learning rate (hyperparameter) ==\n";
    for (double lr : {0.001, 0.01, 0.05, 0.5}) { TrainConfig c; c.lr = lr; row("lr=" + std::to_string(lr), c); }
    return 0;
}

static int cmdTrain(int argc, char** argv) {
    Dataset all = loadData(argc, argv);
    Split sp = split(all, 42);
    Scaler sc; sc.fit(sp.train.X);
    TrainConfig cfg;
    std::cout << "Train/val/test = " << sp.train.X.r << "/" << sp.val.X.r << "/" << sp.test.X.r << " (seed 42)\n";
    TrainResult r = train(cfg, sc.apply(sp.train.X), sp.train.y, sc.apply(sp.val.X), sp.val.y, true);
    Metrics te = evaluate(r.model.predict(sc.apply(sp.test.X)), sp.test.y);
    printMetrics("TEST", te);
    std::string out = arg(argc, argv, "--out", "model.txt");
    std::ofstream f(out); f.precision(17);
    for (size_t j = 0; j < kFeatures; ++j) f << sc.mean[j] << ' ' << sc.stdv[j] << '\n';
    r.model.save(f);
    std::cout << "Saved model -> " << out << "\n";
    return 0;
}

static int cmdLearning() {
    std::cout << std::fixed << std::setprecision(3);
    Dataset d = makeSynthetic(600, 5);
    Scaler sc; sc.fit(d.X); Matrix Xs = sc.apply(d.X);
    std::cout << "== Hebbian (Oja) vs SVD first principal direction ==\n";
    auto w = ojaRule(Xs, 0.002, 30, 1);
    SVDResult s = svd(Xs);
    double dot = 0, nw = 0;
    for (size_t j = 0; j < w.size(); ++j) { dot += w[j] * s.V(j, 0); nw += w[j] * w[j]; }
    std::cout << "  |w| = " << std::sqrt(nw) << " (-> 1),  |cos(w, v1_SVD)| = " << std::fabs(dot) / std::sqrt(nw) << " (-> 1)\n";
    std::cout << "== Competitive learning (unsupervised, 3 prototypes) ==\n";
    Matrix P = competitive(Xs, 3, 0.05, 20, 4);
    int tab[3][3] = {{0}};
    for (size_t i = 0; i < Xs.r; ++i) tab[nearest(P, Xs, i)][d.y[i]]++;
    int pure = 0;
    for (int c = 0; c < 3; ++c) pure += *std::max_element(tab[c], tab[c] + 3);
    std::cout << "  cluster purity vs grade labels = " << (double)pure / (double)Xs.r << " (no labels were used for learning)\n";
    std::cout << "== RBM (energy-based, CD-1) on binarized features ==\n";
    Matrix B(Xs.r, Xs.c);
    for (size_t i = 0; i < Xs.r; ++i) for (size_t j = 0; j < Xs.c; ++j) B(i, j) = Xs(i, j) > 0 ? 1.0 : 0.0;
    auto e = rbmDemo(B, 4, 30, 0.05, 9);
    std::cout << "  reconstruction MSE: before=" << e.first << " after=" << e.second << "\n";
    std::cout << "Note: Hebbian/competitive/RBM use no error signal from labels; error-correction (backprop) does.\n";
    return 0;
}

static int cmdPredict(int argc, char** argv) {
    std::string img = arg(argc, argv, "--image"), mp = arg(argc, argv, "--model", "model.txt");
    if (img.empty()) { std::cerr << "usage: freshness predict --image file.ppm [--model model.txt]\n"; return 1; }
    std::ifstream f(mp);
    if (!f) { std::cerr << "model not found: " << mp << " (run `freshness train` first)\n"; return 1; }
    Scaler sc; sc.mean.resize(kFeatures); sc.stdv.resize(kFeatures);
    for (size_t j = 0; j < kFeatures; ++j) f >> sc.mean[j] >> sc.stdv[j];
    MLP m = MLP::load(f);
    auto ft = imageFeatures(img);
    Matrix x(1, kFeatures);
    for (size_t j = 0; j < kFeatures; ++j) x(0, j) = ft[j];
    Matrix p = m.forward(sc.apply(x));
    int best = 0;
    for (int c = 1; c < 3; ++c) if (p(0, (size_t)c) > p(0, (size_t)best)) best = c;
    std::cout << "Grade: " << kClassNames[best] << "\n";
    for (int c = 0; c < 3; ++c) std::cout << "  " << kClassNames[c] << ": " << p(0, (size_t)c) << "\n";
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "freshness <math|train|compare|learning|predict|demo> [--csv f.csv] [--n 900] [--out model.txt]\n";
        return 1;
    }
    std::string cmd = argv[1];
    try {
        if (cmd == "math") return cmdMath();
        if (cmd == "train") return cmdTrain(argc, argv);
        if (cmd == "compare") return cmdCompare(argc, argv);
        if (cmd == "learning") return cmdLearning();
        if (cmd == "predict") return cmdPredict(argc, argv);
        if (cmd == "demo") { cmdMath(); std::cout << "\n"; cmdLearning(); std::cout << "\n"; cmdCompare(argc, argv); std::cout << "\n"; return cmdTrain(argc, argv); }
    } catch (const std::exception& e) { std::cerr << "error: " << e.what() << "\n"; return 1; }
    std::cerr << "unknown command: " << cmd << "\n";
    return 1;
}
