#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

// THC removed
// #include <THC/THC.h>
// #include <THC/THCDeviceUtils.cuh>

#include <vector>
#include <iostream>
#include <cmath>

void DeformablePSROIPoolForward(
    const at::Tensor data, const at::Tensor bbox, const at::Tensor trans,
    at::Tensor out, at::Tensor top_count, const int batch, const int channels,
    const int height, const int width, const int num_bbox,
    const int channels_trans, const int no_trans, const float spatial_scale,
    const int output_dim, const int group_size, const int pooled_size,
    const int part_size, const int sample_per_part, const float trans_std);

void DeformablePSROIPoolBackwardAcc(
    const at::Tensor out_grad, const at::Tensor data, const at::Tensor bbox,
    const at::Tensor trans, const at::Tensor top_count, at::Tensor in_grad,
    at::Tensor trans_grad, const int batch, const int channels,
    const int height, const int width, const int num_bbox,
    const int channels_trans, const int no_trans, const float spatial_scale,
    const int output_dim, const int group_size, const int pooled_size,
    const int part_size, const int sample_per_part, const float trans_std);

void deform_psroi_pooling_cuda_forward(
    at::Tensor input, at::Tensor bbox, at::Tensor trans, at::Tensor out,
    at::Tensor top_count, const int no_trans, const float spatial_scale,
    const int output_dim, const int group_size, const int pooled_size,
    const int part_size, const int sample_per_part, const float trans_std)
{
  // Ensure we’re on the right CUDA device (recommended)
  c10::cuda::CUDAGuard device_guard(input.device());

  TORCH_CHECK(input.is_contiguous(), "input tensor has to be contiguous");

  const int batch    = static_cast<int>(input.size(0));
  const int channels = static_cast<int>(input.size(1));
  const int height   = static_cast<int>(input.size(2));
  const int width    = static_cast<int>(input.size(3));
  const int channels_trans = no_trans ? 2 : static_cast<int>(trans.size(1));

  const int num_bbox = static_cast<int>(bbox.size(0));
  TORCH_CHECK(num_bbox == out.size(0),
              "Output shape and bbox number won't match: (",
              out.size(0), " vs ", num_bbox, ").");

  DeformablePSROIPoolForward(
      input, bbox, trans, out, top_count, batch, channels, height, width,
      num_bbox, channels_trans, no_trans, spatial_scale, output_dim, group_size,
      pooled_size, part_size, sample_per_part, trans_std);
}

void deform_psroi_pooling_cuda_backward(
    at::Tensor out_grad, at::Tensor input, at::Tensor bbox, at::Tensor trans,
    at::Tensor top_count, at::Tensor input_grad, at::Tensor trans_grad,
    const int no_trans, const float spatial_scale, const int output_dim,
    const int group_size, const int pooled_size, const int part_size,
    const int sample_per_part, const float trans_std)
{
  // Guard device using the forward input’s device
  c10::cuda::CUDAGuard device_guard(input.device());

  TORCH_CHECK(out_grad.is_contiguous(), "out_grad tensor has to be contiguous");
  TORCH_CHECK(input.is_contiguous(),    "input tensor has to be contiguous");

  const int batch    = static_cast<int>(input.size(0));
  const int channels = static_cast<int>(input.size(1));
  const int height   = static_cast<int>(input.size(2));
  const int width    = static_cast<int>(input.size(3));
  const int channels_trans = no_trans ? 2 : static_cast<int>(trans.size(1));

  const int num_bbox = static_cast<int>(bbox.size(0));
  TORCH_CHECK(num_bbox == out_grad.size(0),
              "Output gradient shape and bbox number won't match: (",
              out_grad.size(0), " vs ", num_bbox, ").");

  DeformablePSROIPoolBackwardAcc(
      out_grad, input, bbox, trans, top_count, input_grad, trans_grad, batch,
      channels, height, width, num_bbox, channels_trans, no_trans,
      spatial_scale, output_dim, group_size, pooled_size, part_size,
      sample_per_part, trans_std);
}
