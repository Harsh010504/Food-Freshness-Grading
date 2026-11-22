# Food Quality & Freshness Grading System

Deep Learning PBL (CSAI3027P) - Project 26, Food Processing domain. Pure C++17, **no external dependencies**.

Grades food as **Fresh / Medium / Spoiled** from image-derived features (mean R, G, B, texture, brown-spot fraction).

## What is implemented
| Syllabus area | Where |
|---|---|
| Tensors, broadcasting, inner product, reshape (Unit II) | `include/tensor.hpp`, `freshness math` |
| SVD feature analysis (Unit I) | `include/svd.hpp` (Jacobi SVD) |
| MLP, backprop, softmax cross-entropy (Unit I) | `include/mlp.hpp` |
| SGD, Momentum, Nesterov (Unit I) | `Optimizer` in `mlp.hpp` |
| Gradient checking | `gradientCheck()`, unit tests |
| Init / activation / L2 / augmentation / LR ablations (Unit III) | `freshness compare` |
| Hebbian (Oja), competitive, RBM energy-based (Unit III) | `include/learning.hpp`, `freshness learning` |
| Metrics: accuracy, macro-F1, confusion matrix | `include/data.hpp` |
| CLI app: train / predict on a PPM photo | `src/main.cpp` |

## Build and run
```bash
cmake -S . -B build && cmake --build build
ctest --test-dir build --output-on-failure
./build/freshness demo                    # everything
./build/freshness train --out model.txt
./build/freshness predict --image food.ppm --model model.txt
```
Without CMake: `g++ -std=c++17 -O2 src/main.cpp -o freshness`.

Convert a JPG to PPM: `convert photo.jpg photo.ppm` (ImageMagick).

## Using real data
Default data is **synthetic** (for reproducible demo). For real results, build a CSV with header and rows `meanR,meanG,meanB,texture,brownFrac,label` (label 0/1/2) and run `freshness train --csv data.csv`. Document dataset source and license in `docs/DATASET_CARD.md`.

## Roadmap (remaining syllabus items)
- CNN / ResNet + YOLO via **LibTorch C++** (classification, segmentation, detection)
- LSTM shelf-life regressor (metric: MAE), VAE/GAN minority augmenter
- Benchmark two detectors, latency/memory, model card, report

## Run on GitHub
Push the repo; `.github/workflows/ci.yml` builds, tests and runs the demo on every push (see Actions tab).
