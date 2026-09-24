# CCCL #7216 SM120 register packing: preliminary negative result

Date: 2026-09-23. Base: `NVIDIA/cccl@d6842f00dd44f434ecc761c81a4efbfd9a6249c8`; candidate branch: `perf/7216-sm120-key-packing`. Compiler: CUDA 13.0 `nvcc`, `-O3 -std=c++17 -arch=sm_120`. The experiment ran on one **RTX PRO 6000 Blackwell Server Edition**, physical GPU 1 of the newly supplied two-GPU host. It is not the requested RTX 5090/8-GPU result.

The candidate packs each thread's 8/16-bit keys into 32-bit registers after the existing warp-striped load, then extracts items on demand for rank, callback and shared scatter. It is gated to SM120, keys-only integer sorts and full tiles. No production API, lookback protocol or PDL call was changed.

## Full `cub::DeviceRadixSort::SortKeys` timing

All times are microseconds per complete sort, with workspace preallocated. Each run used five warmups and 20 CUDA-event batches of 30 sorts. The table reports the median after excluding the first batch; 16M combines two runs with opposite candidate/baseline order. Speedup is baseline divided by candidate; values below 1 mean regression. Random input seed was fixed at 7216. Baseline and candidate binaries were compiled from separate worktrees with identical flags and harness source.

| Keys | N | Baseline µs | Packed µs | Speedup |
| --- | ---: | ---: | ---: | ---: |
| `uint8_t` | 1,048,576 | 18.748 | 18.741 | 1.000× |
| `uint16_t` | 1,048,576 | 30.297 | 30.247 | 1.002× |
| `int8_t` | 1,048,576 | 19.036 | 18.988 | 1.003× |
| `int16_t` | 1,048,576 | 30.550 | 30.826 | 0.991× |
| `uint8_t` | 16,777,216 | 106.287 | 106.572 | 0.997× |
| `uint16_t` | 16,777,216 | 183.900 | 183.595 | 1.002× |
| `int8_t` | 16,777,216 | 90.261 | 108.744 | 0.830× |
| `int16_t` | 16,777,216 | 150.176 | 151.071 | 0.994× |

The `int8_t` 16M regression reproduced in both orders: 0.829× and 0.836×. Earlier GPU 0 samples are excluded because an existing vLLM workload began using that GPU during measurement. GPU 1 showed 0% utilization immediately before its run. These measurements are a precheck, not paired 5090 acceptance data.

## Code generation and correctness

| Onesweep key | Baseline registers/thread | Packed registers/thread | Baseline spill bytes load/store | Packed spill bytes load/store |
| --- | ---: | ---: | ---: | ---: |
| `uint8_t` | 100 | 91 | 0 / 0 | 0 / 0 |
| `uint16_t` | 87 | 98 | 0 / 0 | 0 / 0 |
| `int8_t` | 64 | 84 | 20 / 20 | 0 / 0 |
| `int16_t` | 64 | 64 | 0 / 0 | 0 / 0 |

Both binaries passed CPU full-width sort comparison. A separate 1,344-case matrix covered four integer types, 14 sizes including zero, warp and tile boundaries, six distributions, both orders and full/partial bit ranges. Baseline and candidate output digests matched exactly for every case. The candidate compiled with no errors and `git diff --check` passed. Device properties report 65,536 registers per SM; the selected onesweep policy uses 512 threads per CTA. From the static register counts, `int8_t` changes from at most two resident CTAs to at most one. This is a plausible explanation for its slowdown, not a measured occupancy counter.

## Decision

This packed representation does not produce a useful net gain on the available GPU and has a reproducible regression. This Draft PR on the 0z5a fork preserves the experiment for review. It is not ready for an upstream performance PR. The target RTX 5090 and its 0z5a environment remain necessary before closing the original claim. No model weights were downloaded. No existing process was stopped and no installed environment was changed.

The raw CSV samples, build logs and temporary harnesses are retained in the local evidence directory. The candidate source is contained in this branch.
