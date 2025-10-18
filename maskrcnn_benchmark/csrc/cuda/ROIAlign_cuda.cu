// Copyright (c) Facebook, Inc.
// Modernized to remove THC and use ATen/c10 CUDA APIs.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ceil_div.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>  // C10_CUDA_CHECK

#include <cmath>      // ceilf
#include <cstring>    // memset (if ever needed)
#include <algorithm>  // std::min

// TODO make it in a common file
#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

template <typename T>
__device__ T bilinear_interpolate(const T* bottom_data,
    const int height, const int width,
    T y, T x,
    const int /*index*/) {

  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    return 0;
  }

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  int y_low = static_cast<int>(y);
  int x_low = static_cast<int>(x);
  int y_high;
  int x_high;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = static_cast<T>(y_low);
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = static_cast<T>(x_low);
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = static_cast<T>(1.) - ly, hx = static_cast<T>(1.) - lx;

  // do bilinear interpolation
  T v1 = bottom_data[y_low * width + x_low];
  T v2 = bottom_data[y_low * width + x_high];
  T v3 = bottom_data[y_high * width + x_low];
  T v4 = bottom_data[y_high * width + x_high];
  T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

template <typename T>
__global__ void RoIAlignForward(const int nthreads, const T* bottom_data,
    const T spatial_scale, const int channels,
    const int height, const int width,
    const int pooled_height, const int pooled_width,
    const int sampling_ratio,
    const T* bottom_rois, T* top_data) {

  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c  = (index / pooled_width / pooled_height) % channels;
    int n  = index / pooled_width / pooled_height / channels;

    const T* offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = static_cast<int>(offset_bottom_rois[0]);

    // Do not use rounding; this implementation detail is critical
    T roi_start_w = offset_bottom_rois[1] * spatial_scale;
    T roi_start_h = offset_bottom_rois[2] * spatial_scale;
    T roi_end_w   = offset_bottom_rois[3] * spatial_scale;
    T roi_end_h   = offset_bottom_rois[4] * spatial_scale;

    // Force malformed ROIs to be 1x1
    T roi_width  = max(roi_end_w - roi_start_w, static_cast<T>(1.));
    T roi_height = max(roi_end_h - roi_start_h, static_cast<T>(1.));
    T bin_size_h = roi_height / static_cast<T>(pooled_height);
    T bin_size_w = roi_width  / static_cast<T>(pooled_width);

    const T* offset_bottom_data =
        bottom_data + (roi_batch_ind * channels + c) * height * width;

    // Sample grid per bin (integral approximation)
    int roi_bin_grid_h = (sampling_ratio > 0)
        ? sampling_ratio
        : static_cast<int>(ceilf(static_cast<float>(roi_height) / static_cast<float>(pooled_height)));
    int roi_bin_grid_w = (sampling_ratio > 0)
        ? sampling_ratio
        : static_cast<int>(ceilf(static_cast<float>(roi_width) / static_cast<float>(pooled_width)));

    // Average pooling inside the bin
    const T count = static_cast<T>(roi_bin_grid_h * roi_bin_grid_w);

    T output_val = 0.;
    for (int iy = 0; iy < roi_bin_grid_h; iy++) {
      const T y = roi_start_h + ph * bin_size_h
                + static_cast<T>(iy + .5f) * bin_size_h / static_cast<T>(roi_bin_grid_h);
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const T x = roi_start_w + pw * bin_size_w
                  + static_cast<T>(ix + .5f) * bin_size_w / static_cast<T>(roi_bin_grid_w);
        T val = bilinear_interpolate(offset_bottom_data, height, width, y, x, index);
        output_val += val;
      }
    }
    output_val /= count;
    top_data[index] = output_val;
  }
}

template <typename T>
__device__ void bilinear_interpolate_gradient(
    const int height, const int width,
    T y, T x,
    T & w1, T & w2, T & w3, T & w4,
    int & x_low, int & x_high, int & y_low, int & y_high,
    const int /*index*/) {

  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    w1 = w2 = w3 = w4 = 0.;
    x_low = x_high = y_low = y_high = -1;
    return;
  }

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  y_low = static_cast<int>(y);
  x_low = static_cast<int>(x);

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = static_cast<T>(y_low);
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = static_cast<T>(x_low);
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = static_cast<T>(1.) - ly, hx = static_cast<T>(1.) - lx;

  w1 = hy * hx; w2 = hy * lx; w3 = ly * hx; w4 = ly * lx;
}

template <typename T>
__global__ void RoIAlignBackwardFeature(const int nthreads, const T* top_diff,
    const int num_rois, const T spatial_scale,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width,
    const int sampling_ratio,
    T* bottom_diff,
    const T* bottom_rois) {

  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c  = (index / pooled_width / pooled_height) % channels;
    int n  = index / pooled_width / pooled_height / channels;

    const T* offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = static_cast<int>(offset_bottom_rois[0]);

    // Do not use rounding; this implementation detail is critical
    T roi_start_w = offset_bottom_rois[1] * spatial_scale;
    T roi_start_h = offset_bottom_rois[2] * spatial_scale;
    T roi_end_w   = offset_bottom_rois[3] * spatial_scale;
    T roi_end_h   = offset_bottom_rois[4] * spatial_scale;

    // Force malformed ROIs to be 1x1
    T roi_width  = max(roi_end_w - roi_start_w, static_cast<T>(1.));
    T roi_height = max(roi_end_h - roi_start_h, static_cast<T>(1.));
    T bin_size_h = roi_height / static_cast<T>(pooled_height);
    T bin_size_w = roi_width  / static_cast<T>(pooled_width);

    T* offset_bottom_diff =
        bottom_diff + (roi_batch_ind * channels + c) * height * width;

    int top_offset = (n * channels + c) * pooled_height * pooled_width;
    const T* offset_top_diff = top_diff + top_offset;
    const T top_diff_this_bin = offset_top_diff[ph * pooled_width + pw];

    // Sample grid per bin (integral approximation)
    int roi_bin_grid_h = (sampling_ratio > 0)
        ? sampling_ratio
        : static_cast<int>(ceilf(static_cast<float>(roi_height) / static_cast<float>(pooled_height)));
    int roi_bin_grid_w = (sampling_ratio > 0)
        ? sampling_ratio
        : static_cast<int>(ceilf(static_cast<float>(roi_width) / static_cast<float>(pooled_width)));

    // Average pooling inside the bin
    const T count = static_cast<T>(roi_bin_grid_h * roi_bin_grid_w);

    for (int iy = 0; iy < roi_bin_grid_h; iy++) {
      const T y = roi_start_h + ph * bin_size_h
                + static_cast<T>(iy + .5f) * bin_size_h / static_cast<T>(roi_bin_grid_h);
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const T x = roi_start_w + pw * bin_size_w
                  + static_cast<T>(ix + .5f) * bin_size_w / static_cast<T>(roi_bin_grid_w);

        T w1, w2, w3, w4;
        int x_low, x_high, y_low, y_high;

        bilinear_interpolate_gradient(height, width, y, x,
                                      w1, w2, w3, w4,
                                      x_low, x_high, y_low, y_high,
                                      index);

        T g1 = top_diff_this_bin * w1 / count;
        T g2 = top_diff_this_bin * w2 / count;
        T g3 = top_diff_this_bin * w3 / count;
        T g4 = top_diff_this_bin * w4 / count;

        if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
          atomicAdd(offset_bottom_diff + y_low  * width + x_low , static_cast<T>(g1));
          atomicAdd(offset_bottom_diff + y_low  * width + x_high, static_cast<T>(g2));
          atomicAdd(offset_bottom_diff + y_high * width + x_low , static_cast<T>(g3));
          atomicAdd(offset_bottom_diff + y_high * width + x_high, static_cast<T>(g4));
        }
      }
    }
  }
}

// Forward
at::Tensor ROIAlign_forward_cuda(const at::Tensor& input,
                                 const at::Tensor& rois,
                                 const float spatial_scale,
                                 const int pooled_height,
                                 const int pooled_width,
                                 const int sampling_ratio) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(rois.is_cuda(),  "rois must be a CUDA tensor");

  const auto num_rois = rois.size(0);
  const auto channels = input.size(1);
  const auto height   = input.size(2);
  const auto width    = input.size(3);

  auto output = at::empty({num_rois, channels, pooled_height, pooled_width}, input.options());
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
    return output;
  }

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "ROIAlign_forward", [&] {
    RoIAlignForward<scalar_t><<<grid, block, 0, stream>>>(
        static_cast<int>(output_size),
        input.contiguous().data_ptr<scalar_t>(),
        static_cast<scalar_t>(spatial_scale),
        static_cast<int>(channels),
        static_cast<int>(height),
        static_cast<int>(width),
        pooled_height,
        pooled_width,
        sampling_ratio,
        rois.contiguous().data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>());
  });
  C10_CUDA_CHECK(cudaGetLastError());
  return output;
}

// Backward
// TODO remove the dependency on input and use its sizes instead -> save memory
at::Tensor ROIAlign_backward_cuda(const at::Tensor& grad,
                                  const at::Tensor& rois,
                                  const float spatial_scale,
                                  const int pooled_height,
                                  const int pooled_width,
                                  const int batch_size,
                                  const int channels,
                                  const int height,
                                  const int width,
                                  const int sampling_ratio) {
  TORCH_CHECK(grad.is_cuda(), "grad must be a CUDA tensor");
  TORCH_CHECK(rois.is_cuda(), "rois must be a CUDA tensor");

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

  AT_DISPATCH_FLOATING_TYPES(grad.scalar_type(), "ROIAlign_backward", [&] {
    RoIAlignBackwardFeature<scalar_t><<<grid, block, 0, stream>>>(
        static_cast<int>(grad.numel()),
        grad.contiguous().data_ptr<scalar_t>(),
        static_cast<int>(num_rois),
        static_cast<scalar_t>(spatial_scale),
        channels,
        height,
        width,
        pooled_height,
        pooled_width,
        sampling_ratio,
        grad_input.data_ptr<scalar_t>(),
        rois.contiguous().data_ptr<scalar_t>());
  });
  C10_CUDA_CHECK(cudaGetLastError());
  return grad_input;
}
