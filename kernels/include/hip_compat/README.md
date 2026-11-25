# hip_compat

Headers used only by the ROCm/HIP build (`-DUSE_HIP=ON`); the CUDA build never sees this
directory.

- `cuda_runtime.h`, `cuda_fp16.h`, `cuda_bf16.h`, `cooperative_groups.h` and
  `cuda/pipeline` stand in for the CUDA toolkit headers of the same names. The directory
  is placed ahead of the include path, so sources that include them by name get the HIP
  version.
- `flashinfer/math.cuh`, `flashinfer/mma.cuh` and `flashinfer/vec_dtypes.cuh` replace
  the FlashInfer headers of the same names, whose PTX has no HIP spelling. FlashInfer's
  own headers reach them by a sibling-relative include, which the include path cannot
  shadow, so `quest/ops/CMakeLists.txt` force-includes them instead; each keeps the
  original include guard, which turns the original into a no-op.
- The `mma.cuh` replacement emulates `mma.sync` and `ldmatrix` with lane shuffles. The
  prefill kernel uses it only on GPUs without the 16x16x16 matrix instructions it needs
  (gfx908, RDNA2, gfx1250); on gfx90a, gfx94x and gfx950 (MFMA) and RDNA3/RDNA4 (WMMA) it
  runs its tiles on those instructions instead, through
  `kernels/include/prefill/prefill_matrix_core.cuh`.

The kernel sources keep their CUDA spelling. The few HIP-specific changes they still need
are guarded by `USE_HIP`.
