// Copyright (c) Facebook, Inc.
// Modernized to remove THC and use ATen/c10 CUDA APIs.
// Based on: https://github.com/pytorch/pytorch/blob/master/modules/detectron/sigmoid_focal_loss_op.cu

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ceil_div.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>  // C10_CUDA_CHECK

#include <cfloat>
#include <algorithm>  // std::min

// TODO make it in a common file
#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

template <typename T>
__global__ void SigmoidFocalLossForward(const int nthreads,
                                        const T* logits,
                                        const int* targets,
                                        const int num_classes,
                                        const float gamma,
                                        const float alpha,
                                        const int /*num*/,
                                        T* losses) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    const int n = i / num_classes;
    const int d = i % num_classes;        // current class [0..num_classes-1]
    const int t = targets[n];             // target class [1..num_classes], 0 is bg

    // Decide positive or negative
    const T c1 = (t == (d + 1)) ? T(1) : T(0);
    const T c2 = ((t >= 0) && (t != (d + 1))) ? T(1) : T(0);

    const T zp = static_cast<T>(alpha);
    const T zn = static_cast<T>(1.0f - alpha);

    // p = sigmoid(x) = 1/(1+exp(-x)) ; use T overloads (float/double)
    const T x = logits[i];
    const T p = T(1) / (T(1) + exp(-x));

    // (1-p)^gamma * log(p) ; guard log with small eps to avoid -inf/NaN
    const T eps = T(1e-12);
    const T term1 = pow(max(T(1) - p, T(0)), T(gamma)) * log(max(p, eps));

    // p^gamma * log(1-p) using numerically-stable log1p form:
    // log(1 - p) = -x if x >= 0 else log1p(-exp(x)) etc.
    // Use the common stable expression from original code:
    const T stable = (-x) * (x >= T(0)) - log(T(1) + exp(x - T(2) * x * (x >= T(0))));
    const T term2 = pow(max(p, T(0)), T(gamma)) * stable;

    losses[i] = T(0);
    losses[i] += -c1 * term1 * zp;
    losses[i] += -c2 * term2 * zn;
  }
}

template <typename T>
__global__ void SigmoidFocalLossBackward(const int nthreads,
                                         const T* logits,
                                         const int* targets,
                                         const T* d_losses,
                                         const int num_classes,
                                         const float gamma,
                                         const float alpha,
                                         const int /*num*/,
                                         T* d_logits) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    const int n = i / num_classes;
    const int d = i % num_classes;        // current class
    const int t = targets[n];             // target class

    // Decide positive or negative
    const T c1 = (t == (d + 1)) ? T(1) : T(0);
    const T c2 = ((t >= 0) && (t != (d + 1))) ? T(1) : T(0);

    const T zp = static_cast<T>(alpha);
    const T zn = static_cast<T>(1.0f - alpha);

    const T x = logits[i];
    const T p = T(1) / (T(1) + exp(-x));

    const T eps = T(1e-12);

    // (1-p)^g * (1 - p - g*p*log(p))
    const T term1 =
        pow(max(T(1) - p, T(0)), T(gamma)) *
        (T(1) - p - (p * T(gamma) * log(max(p, eps))));

    // p^g * ( g*(1-p)*log(1-p) - p )
    const T stable = (-x) * (x >= T(0)) - log(T(1) + exp(x - T(2) * x * (x >= T(0))));
    const T term2 =
        pow(max(p, T(0)), T(gamma)) * (stable * (T(1) - p) * T(gamma) - p);

    T g = T(0);
    g += -c1 * term1 * zp;
    g += -c2 * term2 * zn;

    d_logits[i] = g * d_losses[i];
  }
}

// Forward wrapper
at::Tensor SigmoidFocalLoss_forward_cuda(const at::Tensor& logits,
                                         const at::Tensor& targets,
                                         const int num_classes,
                                         const float gamma,
                                         const float alpha) {
  TORCH_CHECK(logits.is_cuda(),  "logits must be a CUDA tensor");
  TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
  TORCH_CHECK(logits.dim() == 2, "logits should be NxClass");
  TORCH_CHECK(logits.size(1) == num_classes,
              "logits.size(1) (", logits.size(1), ") must equal num_classes (", num_classes, ")");

  const int num_samples = static_cast<int>(logits.size(0));
  auto losses = at::empty({num_samples, logits.size(1)}, logits.options());
  const int64_t losses_size = static_cast<int64_t>(num_samples) * logits.size(1);

  c10::cuda::CUDAGuard device_guard(logits.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  const int blocks = static_cast<int>(std::min<int64_t>(
      at::ceil_div(losses_size, static_cast<int64_t>(512)),
      static_cast<int64_t>(4096)));
  dim3 grid(blocks);
  dim3 block(512);

  if (losses.numel() == 0) {
    C10_CUDA_CHECK(cudaGetLastError());
    return losses;
  }

  AT_DISPATCH_FLOATING_TYPES(logits.scalar_type(), "SigmoidFocalLoss_forward", [&] {
    SigmoidFocalLossForward<scalar_t><<<grid, block, 0, stream>>>(
        static_cast<int>(losses_size),
        logits.contiguous().data_ptr<scalar_t>(),
        targets.contiguous().data_ptr<int>(),
        num_classes,
        gamma,
        alpha,
        num_samples,
        losses.data_ptr<scalar_t>());
  });
  C10_CUDA_CHECK(cudaGetLastError());
  return losses;
}

// Backward wrapper
at::Tensor SigmoidFocalLoss_backward_cuda(const at::Tensor& logits,
                                          const at::Tensor& targets,
                                          const at::Tensor& d_losses,
                                          const int num_classes,
                                          const float gamma,
                                          const float alpha) {
  TORCH_CHECK(logits.is_cuda(),   "logits must be a CUDA tensor");
  TORCH_CHECK(targets.is_cuda(),  "targets must be a CUDA tensor");
  TORCH_CHECK(d_losses.is_cuda(), "d_losses must be a CUDA tensor");
  TORCH_CHECK(logits.dim() == 2,  "logits should be NxClass");
  TORCH_CHECK(logits.size(1) == num_classes,
              "logits.size(1) (", logits.size(1), ") must equal num_classes (", num_classes, ")");

  const int num_samples = static_cast<int>(logits.size(0));
  auto d_logits = at::zeros({num_samples, num_classes}, logits.options());
  const int64_t d_logits_size = static_cast<int64_t>(num_samples) * num_classes;

  c10::cuda::CUDAGuard device_guard(logits.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  const int blocks = static_cast<int>(std::min<int64_t>(
      at::ceil_div(d_logits_size, static_cast<int64_t>(512)),
      static_cast<int64_t>(4096)));
  dim3 grid(blocks);
  dim3 block(512);

  if (d_logits.numel() == 0) {
    C10_CUDA_CHECK(cudaGetLastError());
    return d_logits;
  }

  AT_DISPATCH_FLOATING_TYPES(logits.scalar_type(), "SigmoidFocalLoss_backward", [&] {
    SigmoidFocalLossBackward<scalar_t><<<grid, block, 0, stream>>>(
        static_cast<int>(d_logits_size),
        logits.contiguous().data_ptr<scalar_t>(),
        targets.contiguous().data_ptr<int>(),
        d_losses.contiguous().data_ptr<scalar_t>(),
        num_classes,
        gamma,
        alpha,
        num_samples,
        d_logits.data_ptr<scalar_t>());
  });
  C10_CUDA_CHECK(cudaGetLastError());
  return d_logits;
}
