// v0 batch-1 GEMV kernels for the decode path (K-PLAN-001, K-PLAN-002 stage 2).
// One thread per output row, scalar loads, FP32 accumulation. This is the correctness baseline: adjacent
// threads read rows K bytes apart, so weight reads are uncoalesced and the kernels are deliberately slow.
//
//   int8_gemv:          y[n] = cast(scales[n] * sum_k float(q[n,k]) * float(x[k]) + bias[n])
//   gemv_16bit_control: y[n] = cast(sum_k float(w[n,k]) * float(x[k]) + bias[n])
//
// The scale is applied after the dot product, as in QuantizedLinear's reference backend. Correctness is
// defined for finite inputs whose unscaled FP32 accumulator stays finite and whose outputs are
// representable in the activation dtype.
#include <torch/extension.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdint>
#include <limits>
#include <optional>

namespace {

constexpr int kThreads = 128;

template <typename scalar_t>
__global__ void int8_gemv_v0_kernel(const scalar_t* __restrict__ x, const int8_t* __restrict__ q,
                                    const float* __restrict__ scales, const float* __restrict__ bias,
                                    scalar_t* __restrict__ y, int64_t n_rows, int64_t k_cols) {
    const int64_t n = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (n >= n_rows) return;
    const int8_t* row = q + n * k_cols;
    float acc = 0.0f;
    for (int64_t k = 0; k < k_cols; ++k) acc += static_cast<float>(row[k]) * static_cast<float>(x[k]);
    float out = acc * scales[n];
    if (bias != nullptr) out += bias[n];
    y[n] = static_cast<scalar_t>(out);
}

template <typename scalar_t>
__global__ void gemv16_control_v0_kernel(const scalar_t* __restrict__ x, const scalar_t* __restrict__ w,
                                         const float* __restrict__ bias, scalar_t* __restrict__ y,
                                         int64_t n_rows, int64_t k_cols) {
    const int64_t n = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (n >= n_rows) return;
    const scalar_t* row = w + n * k_cols;
    float acc = 0.0f;
    for (int64_t k = 0; k < k_cols; ++k) acc += static_cast<float>(row[k]) * static_cast<float>(x[k]);
    float out = acc;
    if (bias != nullptr) out += bias[n];
    y[n] = static_cast<scalar_t>(out);
}

void check_activation(const torch::Tensor& x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kHalf || x.scalar_type() == torch::kBFloat16,
                "x must be float16 or bfloat16, got ", x.scalar_type());
    TORCH_CHECK(x.dim() == 2 && x.size(0) == 1, "x must have shape [1, K], got ", x.sizes());
    TORCH_CHECK(x.size(1) > 0, "K must be positive");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
}

// Checks a [N, K] weight against x and returns N.
int64_t check_weight(const torch::Tensor& w, const char* name, const torch::Tensor& x) {
    TORCH_CHECK(w.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(w.device() == x.device(), name, " is on ", w.device(), " but x is on ", x.device());
    TORCH_CHECK(w.dim() == 2, name, " must have shape [N, K], got ", w.sizes());
    TORCH_CHECK(w.size(0) > 0, "N must be positive");
    TORCH_CHECK(w.size(1) == x.size(1), "x has K=", x.size(1), " but ", name, " has K=", w.size(1));
    TORCH_CHECK(w.is_contiguous(), name, " must be contiguous (row-major [N, K])");
    return w.size(0);
}

void check_row_vector(const torch::Tensor& v, const char* name, int64_t n_rows, const torch::Tensor& x) {
    TORCH_CHECK(v.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(v.device() == x.device(), name, " is on ", v.device(), " but x is on ", x.device());
    TORCH_CHECK(v.scalar_type() == torch::kFloat, name, " must be float32, got ", v.scalar_type());
    TORCH_CHECK(v.dim() == 1 && v.size(0) == n_rows, name, " must have shape [", n_rows, "], got ", v.sizes());
    TORCH_CHECK(v.is_contiguous(), name, " must be contiguous");
}

unsigned int grid_for(int64_t n_rows) {
    const int64_t blocks = (n_rows + kThreads - 1) / kThreads;
    TORCH_CHECK(blocks <= std::numeric_limits<int32_t>::max(), "N=", n_rows, " is too large");
    return static_cast<unsigned int>(blocks);
}

}  // namespace

torch::Tensor int8_gemv(const torch::Tensor& x, const torch::Tensor& qweight, const torch::Tensor& scales,
                        const std::optional<torch::Tensor>& bias) {
    check_activation(x);
    TORCH_CHECK(qweight.scalar_type() == torch::kChar, "qweight must be int8, got ", qweight.scalar_type());
    const int64_t n_rows = check_weight(qweight, "qweight", x);
    check_row_vector(scales, "scales", n_rows, x);
    if (bias.has_value()) check_row_vector(*bias, "bias", n_rows, x);

    const c10::cuda::CUDAGuard guard(x.device());
    auto y = torch::empty({1, n_rows}, x.options());
    const auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_REDUCED_FLOATING_TYPES(x.scalar_type(), "int8_gemv_v0", [&] {
        int8_gemv_v0_kernel<scalar_t><<<grid_for(n_rows), kThreads, 0, stream>>>(
            x.data_ptr<scalar_t>(), qweight.data_ptr<int8_t>(), scales.data_ptr<float>(),
            bias.has_value() ? bias->data_ptr<float>() : nullptr, y.data_ptr<scalar_t>(), n_rows, x.size(1));
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

torch::Tensor gemv_16bit_control(const torch::Tensor& x, const torch::Tensor& weight,
                                 const std::optional<torch::Tensor>& bias) {
    check_activation(x);
    TORCH_CHECK(weight.scalar_type() == x.scalar_type(), "weight must have the dtype of x (", x.scalar_type(),
                "), got ", weight.scalar_type());
    const int64_t n_rows = check_weight(weight, "weight", x);
    if (bias.has_value()) check_row_vector(*bias, "bias", n_rows, x);

    const c10::cuda::CUDAGuard guard(x.device());
    auto y = torch::empty({1, n_rows}, x.options());
    const auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_REDUCED_FLOATING_TYPES(x.scalar_type(), "gemv16_control_v0", [&] {
        gemv16_control_v0_kernel<scalar_t><<<grid_for(n_rows), kThreads, 0, stream>>>(
            x.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(),
            bias.has_value() ? bias->data_ptr<float>() : nullptr, y.data_ptr<scalar_t>(), n_rows, x.size(1));
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("int8_gemv", &int8_gemv,
          "v0 INT8 GEMV: x [1, K] fp16/bf16, qweight [N, K] int8, scales [N] fp32, optional bias [N] fp32 -> [1, N]",
          py::arg("x"), py::arg("qweight"), py::arg("scales"), py::arg("bias") = py::none());
    m.def("gemv_16bit_control", &gemv_16bit_control,
          "v0 16-bit control GEMV: x [1, K], weight [N, K] of x's dtype, optional bias [N] fp32 -> [1, N]",
          py::arg("x"), py::arg("weight"), py::arg("bias") = py::none());
}
