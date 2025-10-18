// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ceil_div.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h> // For C10_CUDA_CHECK
#include <c10/cuda/CUDACachingAllocator.h>

// #include <THC/THC.h>
// #include <THC/THCDeviceUtils.cuh>

#include <vector>
#include <iostream>
#include <cstring>

int const threadsPerBlock = sizeof(unsigned long long) * 8;

__device__ inline float devIoU(float const * const a, float const * const b) {
  float left = max(a[0], b[0]), right = min(a[2], b[2]);
  float top = max(a[1], b[1]), bottom = min(a[3], b[3]);
  float width = max(right - left + 1, 0.f), height = max(bottom - top + 1, 0.f);
  float interS = width * height;
  float Sa = (a[2] - a[0] + 1) * (a[3] - a[1] + 1);
  float Sb = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
  return interS / (Sa + Sb - interS + 1e-10f);
}

__global__ void nms_kernel(const int n_boxes, const float nms_overlap_thresh,
                           const float *dev_boxes, unsigned long long *dev_mask) {
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  // if (row_start > col_start) return;

  const int row_size =
        min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
        min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

  __shared__ float block_boxes[threadsPerBlock * 5];
  if (threadIdx.x < col_size) {
    block_boxes[threadIdx.x * 5 + 0] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 0];
    block_boxes[threadIdx.x * 5 + 1] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 1];
    block_boxes[threadIdx.x * 5 + 2] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 2];
    block_boxes[threadIdx.x * 5 + 3] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 3];
    block_boxes[threadIdx.x * 5 + 4] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 4];
  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
    const float *cur_box = dev_boxes + cur_box_idx * 5;
    int i = 0;
    unsigned long long t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;
    }
    for (i = start; i < col_size; i++) {
      if (devIoU(cur_box, block_boxes + i * 5) > nms_overlap_thresh) {
        t |= 1ULL << i;
      }
    }
    // const int col_blocks = THCCeilDiv(n_boxes, threadsPerBlock);
    const int col_blocks = (n_boxes + threadsPerBlock - 1) / threadsPerBlock;
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}

// boxes is a N x 5 tensor
at::Tensor nms_cuda(const at::Tensor boxes, float nms_overlap_thresh) {
  using scalar_t = float;
  TORCH_CHECK(boxes.is_cuda(), "boxes must be a CUDA tensor");
  TORCH_CHECK(boxes.dim() == 2 && boxes.size(1) == 5, "boxes must be Nx5");
  
  // AT_ASSERTM(boxes.device().is_cuda(), "boxes must be a CUDA tensor");
  auto scores = boxes.select(1, 4);
  auto order_t = std::get<1>(scores.sort(0, /* descending=*/true));
  auto boxes_sorted = boxes.index_select(0, order_t).contiguous();

  // int boxes_num = boxes.size(0);
  const int boxes_num = static_cast<int>(boxes_sorted.size(0));

  if (boxes_num == 0) {
    return at::empty({0}, boxes.options().dtype(at::kLong).device(at::kCPU));
  }

  c10::cuda::CUDAGuard device_guard(boxes.device());
  auto stream = c10::cuda::getCurrentCUDAStream();
  
  // const int col_blocks = THCCeilDiv(boxes_num, threadsPerBlock);
  const int col_blocks = static_cast<int>(at::ceil_div(static_cast<int64_t>(boxes_num),static_cast<int64_t>(threadsPerBlock)));

  scalar_t* boxes_dev = boxes_sorted.data_ptr<scalar_t>();

  // THCState *state = at::globalContext().lazyInitCUDA(); // TODO replace with getTHCState

  // unsigned long long* mask_dev = NULL;
  // //THCudaCheck(THCudaMalloc(state, (void**) &mask_dev,
  // //                      boxes_num * col_blocks * sizeof(unsigned long long)));

  // mask_dev = (unsigned long long*) THCudaMalloc(state, boxes_num * col_blocks * sizeof(unsigned long long));

  unsigned long long* mask_dev = nullptr;
  const size_t mask_bytes = static_cast<size_t>(boxes_num) * static_cast<size_t>(col_blocks) * sizeof(unsigned long long);
  mask_dev = static_cast<unsigned long long*>(c10::cuda::CUDACachingAllocator::raw_alloc(mask_bytes));

  // dim3 blocks(THCCeilDiv(boxes_num, threadsPerBlock),
  //             THCCeilDiv(boxes_num, threadsPerBlock));
  dim3 blocks(static_cast<unsigned>(col_blocks), static_cast<unsigned>(col_blocks));
  dim3 threads(threadsPerBlock);
  nms_kernel<<<blocks, threads, 0, stream>>>(boxes_num,
                                  nms_overlap_thresh,
                                  boxes_dev,
                                  mask_dev);

  C10_CUDA_CHECK(cudaGetLastError());

  // std::vector<unsigned long long> mask_host(boxes_num * col_blocks);
  // THCudaCheck(cudaMemcpy(&mask_host[0],
  //                       mask_dev,
  //                       sizeof(unsigned long long) * boxes_num * col_blocks,
  //                       cudaMemcpyDeviceToHost));

  std::vector<unsigned long long> mask_host(static_cast<size_t>(boxes_num) * col_blocks);
  C10_CUDA_CHECK(cudaMemcpy(mask_host.data(), mask_dev, mask_bytes, cudaMemcpyDeviceToHost));

  // std::vector<unsigned long long> remv(col_blocks);
  // memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks);
  std::vector<unsigned long long> remv(col_blocks, 0ULL);


  at::Tensor keep = at::empty({boxes_num}, boxes.options().dtype(at::kLong).device(at::kCPU));
  int64_t* keep_out = keep.data_ptr<int64_t>();

  int num_to_keep = 0;
  for (int i = 0; i < boxes_num; i++) {
    int nblock = i / threadsPerBlock;
    int inblock = i % threadsPerBlock;

    if (!(remv[nblock] & (1ULL << inblock))) {
      keep_out[num_to_keep++] = i;
      unsigned long long *p = &mask_host[0] + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv[j] |= p[j];
      }
    }
  }

  // THCudaFree(state, mask_dev);
  c10::cuda::CUDACachingAllocator::raw_delete(mask_dev);
  // TODO improve this part
  return std::get<0>(order_t.index({
                       keep.narrow(/*dim=*/0, /*start=*/0, /*length=*/num_to_keep).to(
                         order_t.device(), keep.scalar_type())
                     }).sort(0, false));
}
