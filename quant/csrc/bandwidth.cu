// Streaming-read kernel for measuring peak DRAM read bandwidth (plan section 6.3).
// Every byte of the input is read once. Each thread keeps four 16-byte loads in flight in a grid-stride
// loop and XOR-accumulates them; each block writes one 32-bit partial, which keeps the loads from being
// removed by the compiler. GEMV is read-dominated, so this is the peak its efficiency is measured against.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdint>

namespace {

__device__ __forceinline__ uint32_t xor4(uint4 v) { return v.x ^ v.y ^ v.z ^ v.w; }

__global__ void stream_read_kernel(const uint4* __restrict__ vec, int64_t n_vec,
                                   const uint8_t* __restrict__ tail, int64_t n_tail,
                                   uint32_t* __restrict__ partials) {
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    uint32_t acc = 0;
    for (; i + 3 * stride < n_vec; i += 4 * stride) {
        const uint4 a = vec[i];
        const uint4 b = vec[i + stride];
        const uint4 c = vec[i + 2 * stride];
        const uint4 d = vec[i + 3 * stride];
        acc ^= xor4(a) ^ xor4(b) ^ xor4(c) ^ xor4(d);
    }
    for (; i < n_vec; i += stride) acc ^= xor4(vec[i]);

    // Bytes after the last full vector, as little-endian 32-bit words padded with zeros.
    if (blockIdx.x == 0) {
        for (int64_t j = threadIdx.x; j < n_tail; j += blockDim.x)
            acc ^= static_cast<uint32_t>(tail[j]) << (8 * (j % 4));
    }

    for (int offset = 16; offset > 0; offset /= 2) acc ^= __shfl_xor_sync(0xffffffff, acc, offset);
    __shared__ uint32_t warp_acc[32];
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    if (lane == 0) warp_acc[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        acc = lane < static_cast<int>((blockDim.x + 31) / 32) ? warp_acc[lane] : 0;
        for (int offset = 16; offset > 0; offset /= 2) acc ^= __shfl_xor_sync(0xffffffff, acc, offset);
        if (lane == 0) partials[blockIdx.x] = acc;
    }
}

}  // namespace

torch::Tensor stream_read(const torch::Tensor& x, int64_t blocks, int64_t threads) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "x must be a contiguous CUDA tensor");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, "x must be 16-byte aligned");
    TORCH_CHECK(blocks > 0, "blocks must be positive");
    TORCH_CHECK(threads % 32 == 0 && threads > 0 && threads <= 1024, "threads must be a multiple of 32, at most 1024");
    const c10::cuda::CUDAGuard guard(x.device());
    const int64_t nbytes = x.numel() * x.element_size();
    const int64_t n_vec = nbytes / 16;
    const auto* base = static_cast<const uint8_t*>(x.data_ptr());
    auto partials = torch::empty({blocks}, x.options().dtype(torch::kInt32));
    stream_read_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const uint4*>(base), n_vec, base + n_vec * 16, nbytes - n_vec * 16,
        reinterpret_cast<uint32_t*>(partials.data_ptr<int32_t>()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return partials;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("stream_read", &stream_read, "Read every byte of x once; returns per-block XOR partials (int32)",
          py::arg("x"), py::arg("blocks"), py::arg("threads") = 256);
}
