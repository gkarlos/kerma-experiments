#include <cuda.h>
#include <stdio.h>

#ifdef RD_WG_SIZE_0_0
        #define BLOCK_SIZE RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
        #define BLOCK_SIZE RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
        #define BLOCK_SIZE RD_WG_SIZE
#else
        #define BLOCK_SIZE 16
#endif


__global__ void lud_diagonal(float *m, int matrix_dim, int offset)
{
  int i,j;
  __shared__ float shadow[BLOCK_SIZE][BLOCK_SIZE];

  int array_offset = offset * matrix_dim + offset;

  for(i=0; i < BLOCK_SIZE; i++){
    shadow[i][threadIdx.x] = m[array_offset+threadIdx.x];
    array_offset += matrix_dim;
  }
  // 2/2/2/0

  __syncthreads();
  
  for(i=0; i < BLOCK_SIZE-1; i++) {

    if (threadIdx.x>i){
      for(j=0; j < i; j++)
        shadow[threadIdx.x][i] -= shadow[threadIdx.x][j] * shadow[j][i];
      // 6/6/6/0
      shadow[threadIdx.x][i] /= shadow[i][i];
      // 9/9/9/0
    }

    __syncthreads();

    if (threadIdx.x>i){

      for(j=0; j < i+1; j++)
        shadow[i+1][threadIdx.x] -= shadow[i+1][j] * shadow[j][threadIdx.x];
        // 13/13/13/0
    }
    __syncthreads();
  }

  /* 
     The first row is not modified, it
     is no need to write it back to the
     global memory

   */
  array_offset = (offset+1) * matrix_dim + offset;
  for(i=1; i < BLOCK_SIZE; i++){
    m[array_offset+threadIdx.x]=shadow[i][threadIdx.x];
    // 15/15/15/0
    array_offset += matrix_dim;
  }
}

__global__ void lud_perimeter(float *m, int matrix_dim, int offset)
{
  __shared__ float dia[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float peri_row[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float peri_col[BLOCK_SIZE][BLOCK_SIZE];

  int i,j, array_offset;
  int idx;

  if (threadIdx.x < BLOCK_SIZE) {
    idx = threadIdx.x;
    
    array_offset = offset*matrix_dim+offset;
    for (i=0; i < BLOCK_SIZE/2; i++){
      dia[i][idx]=m[array_offset+idx];
      //2/2/2/0
      array_offset += matrix_dim;
    }
    
    array_offset = offset*matrix_dim+offset;
    for (i=0; i < BLOCK_SIZE; i++) {
      peri_row[i][idx]=m[array_offset+(blockIdx.x+1)*BLOCK_SIZE+idx];
      //4/4/4/0
      array_offset += matrix_dim;
    }

  } else {
    idx = threadIdx.x-BLOCK_SIZE;
    
    array_offset = (offset+BLOCK_SIZE/2)*matrix_dim+offset;
    for (i=BLOCK_SIZE/2; i < BLOCK_SIZE; i++){
      dia[i][idx]=m[array_offset+idx];
      //6/6/6/0
      array_offset += matrix_dim;
    }
    
    array_offset = (offset+(blockIdx.x+1)*BLOCK_SIZE)*matrix_dim+offset;
    for (i=0; i < BLOCK_SIZE; i++) {
      peri_col[i][idx] = m[array_offset+idx];
      // 8/8/8/0
      array_offset += matrix_dim;
    }
  
  }
  __syncthreads();

  if (threadIdx.x < BLOCK_SIZE) { //peri-row
    idx=threadIdx.x;
    for(i=1; i < BLOCK_SIZE; i++){
      for (j=0; j < i; j++)
        peri_row[i][idx]-=dia[i][j]*peri_row[j][idx];
        // 12/12/12/0
    }
  } else { //peri-col
    idx=threadIdx.x - BLOCK_SIZE;
    for(i=0; i < BLOCK_SIZE; i++){
      for(j=0; j < i; j++)
        peri_col[idx][i]-=peri_col[idx][j]*dia[j][i];
      // 16/16/16/0
      peri_col[idx][i] /= dia[i][i];
      // 19/19/19/0
    }
  }

  __syncthreads();
    
  if (threadIdx.x < BLOCK_SIZE) { //peri-row
    idx=threadIdx.x;
    array_offset = (offset+1)*matrix_dim+offset;
    for(i=1; i < BLOCK_SIZE; i++){
      m[array_offset+(blockIdx.x+1)*BLOCK_SIZE+idx] = peri_row[i][idx];
      array_offset += matrix_dim;
      // 21/21/21/0
    }
  } else { //peri-col
    idx=threadIdx.x - BLOCK_SIZE;
    array_offset = (offset+(blockIdx.x+1)*BLOCK_SIZE)*matrix_dim+offset;
    for(i=0; i < BLOCK_SIZE; i++){
      m[array_offset+idx] =  peri_col[i][idx];
      // 23/23/23/0
      array_offset += matrix_dim;
    }
  }

}

__global__ void lud_internal(float *m, int matrix_dim, int offset)
{
  __shared__ float peri_row[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float peri_col[BLOCK_SIZE][BLOCK_SIZE];

  int i;
  float sum;

  int global_row_id = offset + (blockIdx.y+1)*BLOCK_SIZE;
  int global_col_id = offset + (blockIdx.x+1)*BLOCK_SIZE;

  peri_row[threadIdx.y][threadIdx.x] = m[(offset+threadIdx.y)*matrix_dim+global_col_id+threadIdx.x];
  peri_col[threadIdx.y][threadIdx.x] = m[(global_row_id+threadIdx.y)*matrix_dim+offset+threadIdx.x];

  // 4/4/0/0
  __syncthreads();

  sum = 0;
  for (i=0; i < BLOCK_SIZE; i++)
    sum += peri_col[threadIdx.y][i] * peri_row[i][threadIdx.x];
  // 6/6/2/0
  m[(global_row_id+threadIdx.y)*matrix_dim+global_col_id+threadIdx.x] -= sum;
  // 8/8/2/0


}


void lud_cuda(float *m, int matrix_dim)
{
  int i=0;
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  float *m_debug = (float*)malloc(matrix_dim*matrix_dim*sizeof(float));

  for (i=0; i < matrix_dim-BLOCK_SIZE; i += BLOCK_SIZE) {
      lud_diagonal<<<1, BLOCK_SIZE>>>(m, matrix_dim, i);
      lud_perimeter<<<(matrix_dim-i)/BLOCK_SIZE-1, BLOCK_SIZE*2>>>(m, matrix_dim, i);
      dim3 dimGrid((matrix_dim-i)/BLOCK_SIZE-1, (matrix_dim-i)/BLOCK_SIZE-1);
      lud_internal<<<dimGrid, dimBlock>>>(m, matrix_dim, i); 
  }
  lud_diagonal<<<1,BLOCK_SIZE>>>(m, matrix_dim, i);
}

