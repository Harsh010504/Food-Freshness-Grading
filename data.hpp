// Dataset utilities: synthetic freshness features, CSV loading, split, standardization,
// metrics (accuracy, macro-F1, confusion matrix) and PPM image feature extraction.
#pragma once
#include <algorithm>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
#include "tensor.hpp"

namespace fq {

constexpr size_t kFeatures = 5;
inline const char* kClassNames[3] = {"Fresh", "Medium", "Spoiled"};
inline const char* kFeatureNames[kFeatures] = {"meanR", "meanG", "meanB", "texture", "brownFrac"};

struct Dataset {
    Matrix X;
    std::vector<int> y;
};

// Synthetic stand-in for image-derived features. REPLACE with real data via --csv.
inline Dataset makeSynthetic(size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> nd(0, 1);
    // class centers: [meanR, meanG, meanB, texture, brownFrac]
    const double C[3][kFeatures] = {{0.55, 0.70, 0.25, 0.10, 0.05},
                                    {0.62, 0.58, 0.28, 0.18, 0.25},
                                    {0.55, 0.38, 0.25, 0.30, 0.60}};
    const double S = 0.09;  // noise level -> class overlap
    Dataset d; d.X = Matrix(n, kFeatures); d.y.resize(n);
    for (size_t i = 0; i < n; ++i) {
        int c = (int)(i % 3);
        d.y[i] = c;
        for (size_t j = 0; j < kFeatures; ++j) d.X(i, j) = C[c][j] + S * nd(rng);
    }
    return d;
}

// CSV format: header line, then rows "f1,f2,f3,f4,f5,label" with label in {0,1,2}.
inline Dataset loadCsv(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::string line; std::getline(in, line);
    std::vector<std::vector<double>> rows; std::vector<int> y;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line); std::string tok; std::vector<double> r;
        while (std::getline(ss, tok, ',')) r.push_back(std::stod(tok));
        if (r.size() != kFeatures + 1) throw std::runtime_error("bad CSV row: " + line);
        y.push_back((int)r.back()); r.pop_back(); rows.push_back(r);
    }
    Dataset d; d.X = Matrix(rows.size(), kFeatures); d.y = y;
    for (size_t i = 0; i < rows.size(); ++i) for (size_t j = 0; j < kFeatures; ++j) d.X(i, j) = rows[i][j];
    return d;
}

struct Split { Dataset train, val, test; };

inline Dataset subset(const Dataset& d, const std::vector<size_t>& idx) {
    Dataset o; o.X = Matrix(idx.size(), d.X.c); o.y.resize(idx.size());
    for (size_t i = 0; i < idx.size(); ++i) {
        for (size_t j = 0; j < d.X.c; ++j) o.X(i, j) = d.X(idx[i], j);
        o.y[i] = d.y[idx[i]];
    }
    return o;
}

// Shuffled 70/15/15 split with a fixed seed (reproducible, no overlap => no leakage).
inline Split split(const Dataset& d, unsigned seed) {
    std::vector<size_t> idx(d.X.r); std::iota(idx.begin(), idx.end(), 0);
    std::mt19937 rng(seed); std::shuffle(idx.begin(), idx.end(), rng);
    size_t nt = idx.size() * 70 / 100, nv = idx.size() * 15 / 100;
    Split s;
    s.train = subset(d, {idx.begin(), idx.begin() + nt});
    s.val = subset(d, {idx.begin() + nt, idx.begin() + nt + nv});
    s.test = subset(d, {idx.begin() + nt + nv, idx.end()});
    return s;
}

struct Scaler {
    std::vector<double> mean, stdv;
    void fit(const Matrix& X) {
        mean.assign(X.c, 0); stdv.assign(X.c, 0);
        for (size_t j = 0; j < X.c; ++j) {
            for (size_t i = 0; i < X.r; ++i) mean[j] += X(i, j);
            mean[j] /= (double)X.r;
            for (size_t i = 0; i < X.r; ++i) stdv[j] += (X(i, j) - mean[j]) * (X(i, j) - mean[j]);
            stdv[j] = std::sqrt(stdv[j] / (double)X.r) + 1e-9;
        }
    }
    Matrix apply(const Matrix& X) const {
        Matrix o = X;
        for (size_t i = 0; i < X.r; ++i) for (size_t j = 0; j < X.c; ++j) o(i, j) = (X(i, j) - mean[j]) / stdv[j];
        return o;
    }
};

// Data augmentation: Gaussian jitter on features (simulates lighting / sensor noise).
inline Dataset augment(const Dataset& d, double sigma, unsigned seed) {
    std::mt19937 rng(seed); std::normal_distribution<double> nd(0, sigma);
    Dataset o = d;
    for (auto& v : o.X.d) v += nd(rng);
    return o;
}

struct Metrics { double acc = 0, macroF1 = 0; int cm[3][3] = {{0}}; };

inline Metrics evaluate(const std::vector<int>& pred, const std::vector<int>& y) {
    Metrics m; int ok = 0;
    for (size_t i = 0; i < y.size(); ++i) { m.cm[y[i]][pred[i]]++; ok += pred[i] == y[i]; }
    m.acc = y.empty() ? 0 : (double)ok / (double)y.size();
    double f1sum = 0;
    for (int c = 0; c < 3; ++c) {
        double tp = m.cm[c][c], fp = 0, fn = 0;
        for (int k = 0; k < 3; ++k) { if (k != c) { fp += m.cm[k][c]; fn += m.cm[c][k]; } }
        double p = tp + fp > 0 ? tp / (tp + fp) : 0, r = tp + fn > 0 ? tp / (tp + fn) : 0;
        f1sum += p + r > 0 ? 2 * p * r / (p + r) : 0;
    }
    m.macroF1 = f1sum / 3.0;
    return m;
}

// Binary PPM (P6) reader -> 5 image features. Lets the CLI grade a real photo.
inline std::vector<double> imageFeatures(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open image " + path);
    std::string magic; int w, h, mx;
    in >> magic >> w >> h >> mx;
    if (magic != "P6") throw std::runtime_error("only binary PPM (P6) supported; convert with ImageMagick: convert in.jpg out.ppm");
    in.get();
    std::vector<unsigned char> px((size_t)w * h * 3);
    in.read((char*)px.data(), (std::streamsize)px.size());
    double R = 0, G = 0, B = 0, g1 = 0, g2 = 0, brown = 0; size_t n = (size_t)w * h;
    for (size_t i = 0; i < n; ++i) {
        double r = px[3 * i] / (double)mx, g = px[3 * i + 1] / (double)mx, b = px[3 * i + 2] / (double)mx;
        R += r; G += g; B += b;
        double gr = 0.299 * r + 0.587 * g + 0.114 * b; g1 += gr; g2 += gr * gr;
        if (r > g && g > b && r < 0.7 && g < 0.5) brown += 1;  // crude brown/dark-spot detector
    }
    double mg = g1 / (double)n;
    return {R / (double)n, G / (double)n, B / (double)n, std::sqrt(std::max(0.0, g2 / (double)n - mg * mg)), brown / (double)n};
}

}  // namespace fq
