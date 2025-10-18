// Copyright (c) Facebook, Inc.
// Modernized to remove THC and use ATen/c10 CUDA APIs.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ceil_div.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>  // C10_CUDA_CHECK

#include <algorithm>  // std::min
#include <cfloat>     // FLT_MAX
#include <cmath>      // ceilf, floorf
#include <cstring>    // memset (if ever needed)

// TODO make it in a common file
#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

template <typename T>
__global__ void RoIPoolFForward(const int nthreads, const T* bottom_data,
    const T spatial_scale, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const T* bottom_rois, T* top_data, int* argmax_data) {

  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c  = (index / pooled_width / pooled_height) % channels;
    int n  = index / pooled_width / pooled_height / channels;

    const T* offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = static_cast<int>(offset_bottom_rois[0]);

    int roi_start_w = static_cast<int>(::roundf(static_cast<float>(offset_bottom_rois[1] * spatial_scale)));
    int roi_start_h = static_cast<int>(::roundf(static_cast<float>(offset_bottom_rois[2] * spatial_scale)));
    int roi_end_w   = static_cast<int>(::roundf(static_cast<float>(offset_bottom_rois[3] * spatial_scale)));
    int roi_end_h   = static_cast<int>(::roundf(static_cast<float>(offset_bottom_rois[4] * spatial_scale)));

    // Force malformed ROIs to be 1x1
    int roi_width  = max(roi_end_w - roi_start_w + 1, 1);
    int roi_height = max(roi_end_h - roi_start_h + 1, 1);

    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width)  / static_cast<T>(pooled_width);

    int hstart = static_cast<int>(::floorf(static_cast<float>(ph) * static_cast<float>(bin_size_h)));
    int wstart = static_cast<int>(::floorf(static_cast<float>(pw) * static_cast<float>(bin_size_w)));
    int hend   = static_cast<int>(::ceilf(static_cast<float>(ph + 1) * static_cast<float>(bin_size_h)));
    int wend   = static_cast<int>(::ceilf(static_cast<float>(pw + 1) * static_cast<float>(bin_size_w)));

    // Add roi offsets and clip to input boundaries
    hstart = min(max(hstart + roi_start_h, 0), height);
    hend   = min(max(hend   + roi_start_h, 0), height);
    wstart = min(max(wstart + roi_start_w, 0), width);
    wend   = min(max(wend   + roi_start_w, 0), width);
    bool is_empty = (hend <= hstart) || (wend <= wstart);

    // Define an empty pooling region to be zero
    T maxval = is_empty ? static_cast<T>(0) : static_cast<T>(-FLT_MAX);
    // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
    int maxidx = -1;

    const T* offset_bottom_data = bottom_data + (roi_batch_ind * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        int bottom_index = h * width + w;
        T v = offset_bottom_data[bottom_index];
        if (v > maxval) {
          maxval = v;
          maxidx = bottom_index;
        }
      }
    }
    top_data[index]   = maxval;
    argmax_data[index]= maxidx;
  }
}

template <typename T>
__global__ void RoIPoolFBackward(const int nthreads, const T* top_diff,
    const int* argmax_data, const int /*num_rois*/, const T /*spatial_scale*/,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, T* bottom_diff,
    const T* bottom_rois) {

  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c  = (index / pooled_width / pooled_height) % channels;
    int n  = index / pooled_width / pooled_height / channels;

    const T* offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = static_cast<int>(offset_bottom_rois[0]);

    int bottom_offset = (roi_batch_ind * channels + c) * height * width;
    int top_offset    = (n * channels + c) * pooled_height * pooled_width;

    const T*  offset_top_diff   = top_diff + top_offset;
    T*        offset_bottom_diff= bottom_diff + bottom_offset;
    const int* offset_argmax    = argmax_data + top_offset;

    int argmax = offset_argmax[ph * pooled_width + pw];
    if (argmax != -1) {
      atomicAdd(offset_bottom_diff + argmax,
                static_cast<T>(offset_top_diff[ph * pooled_width + pw]));
    }
  }
}

// Forward
std::tuple<at::Tensor, at::Tensor> ROIPool_forward_cuda(const at::Tensor& input,
                                const at::Tensor& rois,
                                const float spatial_scale,
                                const int pooled_height,
                                const int pooled_width) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(rois.is_cuda(),  "rois must be a CUDA tensor");

  const auto num_rois = rois.size(0);
  const auto channels = input.size(1);
  const auto height   = input.size(2);
  const auto width    = input.size(3);

  auto output = at::empty({num_rois, channels, pooled_height, pooled_width}, input.options());
  auto argmax = at::zeros({num_rois, channels, pooled_height, pooled_width}, input.options().dtype(at::kInt));
  const int64_t output_size = static_cast<int64_t>(num_rois) * pooled_height * pooled_width * channels;

  c10::cuda::CUDAGuard device_guard(input.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  const int blocks = static_cast<int>(std::min<int64_t>(
      at::ceil_div(output_size, static_cast<int64_t>(512)),
      static_cast<int64_t>(4096)));
  dim3 grid(blocks);
  dim3 block(512);

  if (output.numel() == 0) {
    C10_CUDA_CHECK(cudaGetLastError());
    return std::make_tuple(output, argmax);
  }

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "ROIPool_forward", [&] {
    RoIPoolFForward<scalar_t><<<grid, block, 0, stream>>>(
         static_cast<int>(output_size),
         input.contiguous().data_ptr<scalar_t>(),
         static_cast<scalar_t>(spatial_scale),
         static_cast<int>(channels),
         static_cast<int>(height),
         static_cast<int>(width),
         pooled_height,
         pooled_width,
         rois.contiguous().data_ptr<scalar_t>(),
         output.data_ptr<scalar_t>(),
         argmax.data_ptr<int>());
  });
  C10_CUDA_CHECK(cudaGetLastError());
  return std::make_tuple(output, argmax);
}

// Backward
// TODO remove the dependency on input and use instead its sizes -> save memory
at::Tensor ROIPool_backward_cuda(const at::Tensor& grad,
                                 const at::Tensor& input,
                                 const at::Tensor& rois,
                                 const at::Tensor& argmax,
                                 const float spatial_scale,
                                 const int pooled_height,
                                 const int pooled_width,
                                 const int batch_size,
                                 const int channels,
                                 const int height,
                                 const int width) {
  TORCH_CHECK(grad.is_cuda(), "grad must be a CUDA tensor");
  TORCH_CHECK(rois.is_cuda(), "rois must be a CUDA tensor");
  (void)input; (void)spatial_scale; // not used in this kernel version

  const auto num_rois = rois.size(0);

  auto grad_input = at::zeros({batch_size, channels, height, width}, grad.options());

  c10::cuda::CUDAGuard device_guard(grad.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  const int64_t grad_elems = grad.numel();
  const int blocks = static_cast<int>(std::min<int64_t>(
      at::ceil_div(grad_elems, static_cast<int64_t>(512)),
      static_cast<int64_t>(4096)));
  dim3 grid(blocks);
  dim3 block(512);

  // handle possibly empty gradients
  if (grad.numel() == 0) {
    C10_CUDA_CHECK(cudaGetLastError());
    return grad_input;
  }

  AT_DISPATCH_FLOATING_TYPES(grad.scalar_type(), "ROIPool_backward", [&] {
    RoIPoolFBackward<scalar_t><<<grid, block, 0, stream>>>(
         static_cast<int>(grad.numel()),
         grad.contiguous().data_ptr<scalar_t>(),
         argmax.data_ptr<int>(),
         static_cast<int>(num_rois),
         static_cast<scalar_t>(spatial_scale),
         channels,
         height,
         width,
         pooled_height,
         pooled_width,
         grad_input.data_ptr<scalar_t>(),
         rois.contiguous().data_ptr<scalar_t>());
  });
  C10_CUDA_CHECK(cudaGetLastError());
  return grad_input;
}
