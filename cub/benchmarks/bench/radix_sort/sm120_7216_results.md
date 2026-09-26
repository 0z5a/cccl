# CCCL #7216 SM120 register packing: RTX 5090 validation

Date: 2026-09-27. Baseline: `NVIDIA/cccl@d6842f00dd44f434ecc761c81a4efbfd9a6249c8`; candidate: `0z5a/cccl@c88db366a229bb77296fb2715fd3711e26039302` plus this report. Both binaries used CUDA 13.0.88 `nvcc -O3 -std=c++17 -arch=sm_120` and identical harness source. The test ran on GPU 2 of an eight-GPU host: GeForce RTX 5090, UUID `GPU-292ec65d-5654-8757-1164-5c05c383431a`, driver 580.82.07. GPU 2 was idle immediately before and after the run. Files were kept under `/home/gongji/0z5a-work` on local ext4.

The candidate packs 8/16-bit keys into 32-bit registers after the existing warp-striped load, then extracts items for ranking and scattering. The path is gated to SM120, keys-only integer sorts and full tiles.

## Complete `cub::DeviceRadixSort::SortKeys` timing on RTX 5090

Each process ran five warmups and 20 CUDA-event batches of 30 complete sorts, with workspace preallocated. The first timed batch was excluded. Eight fresh processes per size alternated as `B C C B C B B C`, where B is baseline and C is candidate. Times below are medians of four process medians. Speedup is the median of four adjacent baseline/candidate ratios; below 1× means slower. Input uses fixed seed 7216 and is checked against CPU `std::sort` before timing.

| Key | N | Baseline (µs) | Candidate (µs) | Paired speedup |
| --- | ---: | ---: | ---: | ---: |
| `uint8_t` | 1,048,576 | 29.188 | 30.191 | 0.966× |
| `uint16_t` | 1,048,576 | 38.182 | 38.402 | 0.993× |
| `int8_t` | 1,048,576 | 27.749 | 28.032 | 0.993× |
| `int16_t` | 1,048,576 | 38.375 | 38.386 | 0.996× |
| `uint8_t` | 16,777,216 | 88.884 | 89.160 | 0.996× |
| `uint16_t` | 16,777,216 | 156.492 | 156.242 | 1.001× |
| `int8_t` | 16,777,216 | 74.446 | 90.901 | **0.819×** |
| `int16_t` | 16,777,216 | 124.734 | 126.784 | 0.984× |

The 16M `int8_t` paired ratios were 0.8189×, 0.8186×, 0.8195× and 0.8186×. The 1M timings, especially `uint8_t`, vary more between processes and do not support a precise small-effect claim.

## Correctness and code generation

Baseline and candidate each passed a 1,344-case matrix: four integer key types, 14 sizes including zero and tile boundaries, six distributions, ascending/descending and full/partial bit ranges. Their output digest files matched byte for byte (SHA-256 `e6133682ab40ee6ee484171e96a87a6c2a791063da6dca970f9dde0aad73ab3b`). Full-width cases also matched CPU sorting. Both benchmarks compiled and passed CPU reference checks. `git diff --check` passed.

| Onesweep key | Baseline registers/thread | Candidate registers/thread | Baseline spill load/store bytes | Candidate spill load/store bytes |
| --- | ---: | ---: | ---: | ---: |
| `uint8_t` | 100 | 91 | 0 / 0 | 0 / 0 |
| `uint16_t` | 87 | 98 | 0 / 0 | 0 / 0 |
| `int8_t` | 64 | 84 | 20 / 20 | 0 / 0 |
| `int16_t` | 64 | 64 | 0 / 0 | 0 / 0 |

For the 512-thread onesweep policy and 65,536-register SM limit, the static `int8_t` counts allow at most two baseline CTAs versus one candidate CTA per SM. This is a plausible explanation for the slowdown; no occupancy counter was collected.

## Independent SM120 precheck

An earlier run on an RTX PRO 6000 Blackwell Server Edition used the same fixed baseline and candidate source. Its complete-sort results also showed no useful gain:

| Key | N | Baseline (µs) | Candidate (µs) | Speedup |
| --- | ---: | ---: | ---: | ---: |
| `uint8_t` | 1,048,576 | 18.748 | 18.741 | 1.000× |
| `uint16_t` | 1,048,576 | 30.297 | 30.247 | 1.002× |
| `int8_t` | 1,048,576 | 19.036 | 18.988 | 1.003× |
| `int16_t` | 1,048,576 | 30.550 | 30.826 | 0.991× |
| `uint8_t` | 16,777,216 | 106.287 | 106.572 | 0.997× |
| `uint16_t` | 16,777,216 | 183.900 | 183.595 | 1.002× |
| `int8_t` | 16,777,216 | 90.261 | 108.744 | **0.830×** |
| `int16_t` | 16,777,216 | 150.176 | 151.071 | 0.994× |

## Decision

The target RTX 5090 gate is complete and negative: this packing approach causes a repeatable 16M `int8_t` regression. Keep the fork PR as Draft evidence; do not promote this candidate as an optimization. The measurements cover the complete CUB sort call, not a model-level E2E workload. Raw CSVs, build logs and harnesses are retained in the local evidence directory `cccl-7216-sm120-evidence/rtx5090-20260927`. No model weights were downloaded, no existing process was stopped and no installed environment was changed.
