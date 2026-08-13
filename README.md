# HPC101-AscendC-FusedRMSNorm

Fused Add + RMS Normalization kernel optimization on Huawei Ascend 910B NPU using Ascend C.

## Task
Implement `fused_add_rmsnorm` operator on Ascend NPU. Choose one of three frameworks:
- **Ascend C** (selected) — custom kernel with tiling
- Triton-Ascend
- TileLang-Ascend

Correctness: 5 hidden test cases (all pass)
Performance: Measured against baseline

## Implementation
- `src/ascendc/op_kernel/fused_add_rms_norm.cpp` — Ascend C kernel implementation
- `src/ascendc/op_kernel/fused_add_rms_norm_tiling.h` — Tiling configuration
- `src/ascendc/op_host/fused_add_rms_norm.cpp` — Host-side operator logic

## Build & Test
```bash
hpc submit -p lab3p5 bash checker/run.sh        # Correctness
hpc submit -p lab3p5 bash checker/profile.sh     # Performance profiling
```

## Structure
- `src/ascendc/` — Ascend C implementation (selected)
- `src/triton/` — Triton-Ascend implementation (reference)
- `src/tilelang/` — TileLang-Ascend implementation (reference)
- `checker/` — Test and profiling scripts
