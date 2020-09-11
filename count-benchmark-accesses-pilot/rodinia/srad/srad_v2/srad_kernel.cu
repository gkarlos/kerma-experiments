#include "srad.h"
#include <stdio.h>

__global__ void srad_cuda_1(float *E_C,
                            float *W_C,
                            float *N_C,
                            float *S_C,
                            float * J_cuda,
                            float * C_cuda,
                            int cols,
                            int rows,
                            float q0sqr
)
{

  //block id
  int bx = blockIdx.x;
  int by = blockIdx.y;

  //thread id
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  
  //indices
  int index   = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + tx;
  int index_n = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + tx - cols;
  int index_s = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * BLOCK_SIZE + tx;
  int index_w = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty - 1;
  int index_e = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + BLOCK_SIZE;

  float n, w, e, s, jc, g2, l, num, den, qsqr, c;

  //shared memory allocation
  __shared__ float temp[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float temp_result[BLOCK_SIZE][BLOCK_SIZE];

  __shared__ float north[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float south[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float  east[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float  west[BLOCK_SIZE][BLOCK_SIZE];

  //load data to shared memory
  north[ty][tx] = J_cuda[index_n]; 
  south[ty][tx] = J_cuda[index_s];
  // 4/4/0/0

  if ( by == 0 ){
    north[ty][tx] = J_cuda[BLOCK_SIZE * bx + tx]; 
    // 6/6/0/0
  }
  else if ( by == gridDim.y - 1 ){
    south[ty][tx] = J_cuda[cols * BLOCK_SIZE * (gridDim.y - 1) + BLOCK_SIZE * bx + cols * ( BLOCK_SIZE - 1 ) + tx];
    // 8/8/0/0
  }
   __syncthreads();
 
  west[ty][tx] = J_cuda[index_w];
  east[ty][tx] = J_cuda[index_e];
  // 10/10/0/0

  if ( bx == 0 ){
    west[ty][tx] = J_cuda[cols * BLOCK_SIZE * by + cols * ty]; 
    // 12/12/0/0
  }
  else if ( bx == gridDim.x - 1 ){
    east[ty][tx] = J_cuda[cols * BLOCK_SIZE * by + BLOCK_SIZE * ( gridDim.x - 1) + cols * ty + BLOCK_SIZE-1];
    // 14/14/0/0
  }
 
  __syncthreads();

  temp[ty][tx] = J_cuda[index];
  // 16/16/0/0

  __syncthreads();

  jc = temp[ty][tx];
  // 17/17/0/0

  if ( ty == 0 && tx == 0 ){ //nw
    n  = north[ty][tx] - jc;
    s  = temp[ty+1][tx] - jc;
    w  = west[ty][tx]  - jc;
    e  = temp[ty][tx+1] - jc;
    // 21/21/0/0
  }
  else if ( ty == 0 && tx == BLOCK_SIZE-1 ){ //ne
	n  = north[ty][tx] - jc;
    s  = temp[ty+1][tx] - jc;
    w  = temp[ty][tx-1] - jc;
    e  = east[ty][tx] - jc;
    // 25/25/0/0
  }
  else if ( ty == BLOCK_SIZE -1 && tx == BLOCK_SIZE - 1){ //se
	n  = temp[ty-1][tx] - jc;
    s  = south[ty][tx] - jc;
    w  = temp[ty][tx-1] - jc;
    e  = east[ty][tx]  - jc;
    // 29/29/0/0
  }
  else if ( ty == BLOCK_SIZE -1 && tx == 0 ){//sw
	n  = temp[ty-1][tx] - jc;
    s  = south[ty][tx] - jc;
    w  = west[ty][tx]  - jc; 
    e  = temp[ty][tx+1] - jc;
    // 33/33/0/0
  }
  else if ( ty == 0 ){ //n
	n  = north[ty][tx] - jc;
    s  = temp[ty+1][tx] - jc;
    w  = temp[ty][tx-1] - jc; 
    e  = temp[ty][tx+1] - jc;
    // 37/37/0/0
  }
  else if ( tx == BLOCK_SIZE -1 ){ //e
	n  = temp[ty-1][tx] - jc;
    s  = temp[ty+1][tx] - jc;
    w  = temp[ty][tx-1] - jc; 
    e  = east[ty][tx] - jc;
    // 41/41/0/0
  }
  else if ( ty == BLOCK_SIZE -1){ //s
	n  = temp[ty-1][tx] - jc;
    s  = south[ty][tx] - jc;
    w  = temp[ty][tx-1] - jc; 
    e  = temp[ty][tx+1] - jc;
    // 45/45/0/0
  }
  else if ( tx == 0 ){ //w
	n  = temp[ty-1][tx] - jc;
    s  = temp[ty+1][tx] - jc;
    w  = west[ty][tx] - jc; 
    e  = temp[ty][tx+1] - jc;
    // 49/49/0/0
  }
  else{  //the data elements which are not on the borders 
	n  = temp[ty-1][tx] - jc;
    s  = temp[ty+1][tx] - jc;
    w  = temp[ty][tx-1] - jc;
    e  = temp[ty][tx+1] - jc;
    //53/53/0/0
  }


  g2 = ( n * n + s * s + w * w + e * e ) / (jc * jc);

  l = ( n + s + w + e ) / jc;

  num  = (0.5*g2) - ((1.0/16.0)*(l*l)) ;
  den  = 1 + (.25*l);
  qsqr = num/(den*den);

  // diffusion coefficent (equ 33)
  den = (qsqr-q0sqr) / (q0sqr * (1+q0sqr)) ;
  c = 1.0 / (1.0+den) ;

  // saturate diffusion coefficent
  if (c < 0){
    temp_result[ty][tx] = 0;
  }
  else if (c > 1) {
    temp_result[ty][tx] = 1;
  }
  else {
    temp_result[ty][tx] = c;
  }
  //56/56/0/0

  __syncthreads();

  C_cuda[index] = temp_result[ty][tx];
  E_C[index] = e;
  W_C[index] = w;
  S_C[index] = s;
  N_C[index] = n;
  //62/62/0/0

}

__global__ void srad_cuda_2(float *E_C, 
                            float *W_C, 
                            float *N_C, 
                            float *S_C,	
                            float * J_cuda, 
                            float * C_cuda, 
                            int cols, 
                            int rows, 
                            float lambda,
                            float q0sqr
) 
{
	//block id
	int bx = blockIdx.x;
    int by = blockIdx.y;

	//thread id
    int tx = threadIdx.x;
    int ty = threadIdx.y;

	//indices
    int index   = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + tx;
	int index_s = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * BLOCK_SIZE + tx;
    int index_e = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + BLOCK_SIZE;
	float cc, cn, cs, ce, cw, d_sum;

	//shared memory allocation
	__shared__ float south_c[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float  east_c[BLOCK_SIZE][BLOCK_SIZE];

    __shared__ float c_cuda_temp[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float c_cuda_result[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float temp[BLOCK_SIZE][BLOCK_SIZE];

    //load data to shared memory
	temp[ty][tx] = J_cuda[index];
    // 2/2/0/0

    __syncthreads();
	 
	south_c[ty][tx] = C_cuda[index_s];
    // 4/4/0/0

	if ( by == gridDim.y - 1 ){
	  south_c[ty][tx] = C_cuda[cols * BLOCK_SIZE * (gridDim.y - 1) + BLOCK_SIZE * bx + cols * ( BLOCK_SIZE - 1 ) + tx];
	}
    // 6/6/0/0

	__syncthreads();

	east_c[ty][tx] = C_cuda[index_e];
    // 8/8/0/0

	if ( bx == gridDim.x - 1 ){
	  east_c[ty][tx] = C_cuda[cols * BLOCK_SIZE * by + BLOCK_SIZE * ( gridDim.x - 1) + cols * ty + BLOCK_SIZE-1];
	}
	//10/10/0/0
    __syncthreads();

    c_cuda_temp[ty][tx]      = C_cuda[index];
    //12/12/0/0

    __syncthreads();

	cc = c_cuda_temp[ty][tx];
    //13/13/0/0

    if ( ty == BLOCK_SIZE -1 && tx == BLOCK_SIZE - 1){ //se
        cn  = cc;
        cs  = south_c[ty][tx];
        cw  = cc; 
        ce  = east_c[ty][tx];
        // 15/15/0/0
    }
    else if ( tx == BLOCK_SIZE -1 ){ //e
        cn  = cc;
        cs  = c_cuda_temp[ty+1][tx];
        cw  = cc; 
        ce  = east_c[ty][tx];
        // 17/17/0/0
    }
    else if ( ty == BLOCK_SIZE -1){ //s
        cn  = cc;
        cs  = south_c[ty][tx];
        cw  = cc; 
        ce  = c_cuda_temp[ty][tx+1];
        // 19/19/0/0
    }
    else{ //the data elements which are not on the borders 
        cn  = cc;
        cs  = c_cuda_temp[ty+1][tx];
        cw  = cc; 
        ce  = c_cuda_temp[ty][tx+1];
        // 21/21/0/0
    }

   // divergence (equ 58)
   d_sum = cn * N_C[index] + cs * S_C[index] + cw * W_C[index] + ce * E_C[index];
   // 25/25/0/0

   // image update (equ 61)
   c_cuda_result[ty][tx] = temp[ty][tx] + 0.25 * lambda * d_sum;
   // 27/27/0/0

   __syncthreads();

   J_cuda[index] = c_cuda_result[ty][tx];
   // 29/29/0/0
}
