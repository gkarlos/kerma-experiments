

#ifndef _BACKPROP_CUDA_KERNEL_H_
#define _BACKPROP_CUDA_KERNEL_H_

#include "backprop.h"
#include "cuda.h"
#include "math.h"
#include <stdio.h>

__global__ void bpnn_layerforward_CUDA(float *input_cuda,
                                       float *output_hidden_cuda,
                                       float *input_hidden_cuda,
                                       float *hidden_partial_sum, int in,
                                       int hid) {
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int index = (hid + 1) * HEIGHT * by + (hid + 1) * ty + tx + 1 + (hid + 1);

  int index_in = HEIGHT * by + ty + 1;

  __shared__ float input_node[HEIGHT];
  __shared__ float weight_matrix[HEIGHT][WIDTH];

  if (tx == 0)
    input_node[ty] = input_cuda[index_in];
  // 2/2/0/0

  __syncthreads();

  weight_matrix[ty][tx] = input_hidden_cuda[index];
  // 4/4/0/0

  __syncthreads();

  weight_matrix[ty][tx] = weight_matrix[ty][tx] * input_node[ty];
  // 7/7/0/0

  __syncthreads();

  for (int i = 1; i <= __log2f(HEIGHT); i++) {

    int power_two = __powf(2, i);

    if (ty % power_two == 0)
      weight_matrix[ty][tx] =
          weight_matrix[ty][tx] + weight_matrix[ty + power_two / 2][tx];

    __syncthreads();
  }
  // 10/10/3/0

  //__syncthreads();

  input_hidden_cuda[index] = weight_matrix[ty][tx];
  // 12/12/3/0

  __syncthreads();

  if (tx == 0) {
    hidden_partial_sum[by * hid + ty] = weight_matrix[tx][ty];
  }
  // 14/14/3/0
}

__global__ void bpnn_adjust_weights_cuda(float *delta, int hid, float *ly,
                                         int in, float *w, float *oldw) {

  int by = blockIdx.y;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int index = (hid + 1) * HEIGHT * by + (hid + 1) * ty + tx + 1 + (hid + 1);
  int index_y = HEIGHT * by + ty + 1;
  int index_x = tx + 1;
  // eta = 0.3;
  // momentum = 0.3;

  w[index] += ((ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]));
  // 5/5/0/0

  oldw[index] =
      ((ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]));
  // 9/9/0/0

  __syncthreads();

  if (ty == 0 && by == 0) {
    w[index_x] += ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]));
    // 13/13/0/0
    oldw[index_x] = ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]));
    // 16/16/0/0
  }
}
#endif
