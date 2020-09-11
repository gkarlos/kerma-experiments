#ifndef SRAD_KERNELS_CU
#define SRAD_KERNELS_CU

#include "define.h"

// statistical kernel
__global__ void extract(long d_Ne, fp *d_I){ // pointer to input image (DEVICE GLOBAL MEMORY)

	// indexes
	int bx = blockIdx.x;											// get current horizontal block index
	int tx = threadIdx.x;											// get current horizontal thread index
	int ei = (bx * NUMBER_THREADS) + tx;					// unique thread id, more threads than actual elements !!!

	// copy input to output & log uncompress
	if(ei < d_Ne){															// do only for the number of elements, omit extra threads
		d_I[ei] = exp(d_I[ei]/255);							// exponentiate input IMAGE and copy to output image
	}
}

// statistical kernel
__global__ void prepare(long d_Ne,
											  fp *d_I,						// pointer to output image (DEVICE GLOBAL MEMORY)
											  fp *d_sums,					// pointer to input image (DEVICE GLOBAL MEMORY)
											  fp *d_sums2){

	// indexes
	int bx = blockIdx.x;											// get current horizontal block index
	int tx = threadIdx.x;											// get current horizontal thread index
	int ei = (bx * NUMBER_THREADS) + tx;					// unique thread id, more threads than actual elements !!!

	// copy input to output & log uncompress
	if(ei < d_Ne){                              // do only for the number of elements, omit extra threads
		d_sums[ei] = d_I[ei];
		d_sums2[ei] = d_I[ei] * d_I[ei];
	}
}

// statistical kernel
__global__ void compress(long d_Ne,
                         fp *d_I) {         // pointer to output image (DEVICE GLOBAL MEMORY)

	// indexes
	int bx = blockIdx.x;											// get current horizontal block index
	int tx = threadIdx.x;											// get current horizontal thread index
	int ei = (bx*NUMBER_THREADS)+tx;					// unique thread id, more threads than actual elements !!!

	// copy input to output & log uncompress
	if(ei<d_Ne){															// do only for the number of elements, omit extra threads
		d_I[ei] = log(d_I[ei])*255;						  // exponentiate input IMAGE and copy to output image
	}
}


// statistical kernel
__global__ void reduce(	long d_Ne,					// number of elements in array
										    int d_no,						// number of sums to reduce
										    int d_mul,					// increment
										    fp *d_sums,	        // pointer to partial sums variable (DEVICE GLOBAL MEMORY)
										    fp *d_sums2) {

	// indexes
  int bx = blockIdx.x;                      // get current horizontal block index
	int tx = threadIdx.x;											// get current horizontal thread index
	int ei = (bx*NUMBER_THREADS)+tx;					// unique thread id, more threads than actual elements !!!
	int nf = NUMBER_THREADS-(gridDim.x*NUMBER_THREADS-d_no);				// number of elements assigned to last block
	int df = 0;																// divisibility factor for the last block

	// statistical
	__shared__ fp d_psum[NUMBER_THREADS];			// data for block calculations allocated by every block in its shared memory
	__shared__ fp d_psum2[NUMBER_THREADS];

	// counters
	int i;

	// copy data to shared memory
	if(ei<d_no){															// do only for the number of elements, omit extra threads
		d_psum[tx] = d_sums[ei*d_mul];
		d_psum2[tx] = d_sums2[ei*d_mul];
	}
  // 4/4/0/0

	// reduction of sums if all blocks are full (rare case)	
  if(nf == NUMBER_THREADS){
		for(i=2; i<=NUMBER_THREADS; i=2*i){     // sum of every 2, 4, ..., NUMBER_THREADS elements
			if((tx+1) % i == 0){									// every ith
				d_psum[tx] = d_psum[tx] + d_psum[tx-i/2];
				d_psum2[tx] = d_psum2[tx] + d_psum2[tx-i/2];
			}
			__syncthreads();
		}
    //10/10/6/0

		// final sumation by last thread in every block
		if(tx==(NUMBER_THREADS-1)){											// block result stored in global memory
			d_sums[bx*d_mul*NUMBER_THREADS] = d_psum[tx];
			d_sums2[bx*d_mul*NUMBER_THREADS] = d_psum2[tx];
		}
    //14/14/6/0
	}
	else {                                     // reduction of sums if last block is not full (common case)
		if(bx != (gridDim.x - 1)) {              // for full blocks (all except for last block)
			for(i=2; i<=NUMBER_THREADS; i=2*i){		 // sum of every 2, 4, ..., NUMBER_THREADS elements
				// sum of elements
				if((tx+1) % i == 0){								 // every ith
					d_psum[tx] = d_psum[tx] + d_psum[tx-i/2];
					d_psum2[tx] = d_psum2[tx] + d_psum2[tx-i/2];
				}
				// synchronization
				__syncthreads();
			}
      //20/20/12/0

			// final sumation by last thread in every block
			if(tx==(NUMBER_THREADS-1)){						 // block result stored in global memory
				d_sums[bx*d_mul*NUMBER_THREADS] = d_psum[tx];
				d_sums2[bx*d_mul*NUMBER_THREADS] = d_psum2[tx];
			}
      //24/24/12/0
		}

		else {											             // for not full block (last block)
			// figure out divisibility
			for(int i=2; i<=NUMBER_THREADS; i=2*i){
        if(nf >= i) {
          df = i;
        }
      }

			// sum of every 2, 4, ..., NUMBER_THREADS elements
			for(int i=2; i<=df; i=2*i){											//
				// sum of elements (only busy threads)
				if((tx+1) % i == 0 && tx<df){								// every ith
					d_psum[tx] = d_psum[tx] + d_psum[tx-i/2];
					d_psum2[tx] = d_psum2[tx] + d_psum2[tx-i/2];
				}
				__syncthreads();											//
			}
      //30/30/18/0

			// remainder / final summation by last thread
			if(tx == (df-1)) {										//
				// compute the remainder and final summation by last busy thread
				for(i = ( bx * NUMBER_THREADS) + df; i < (bx*NUMBER_THREADS) + nf; i++){						//
					d_psum[tx] = d_psum[tx] + d_sums[i];
					d_psum2[tx] = d_psum2[tx] + d_sums2[i];
				}
        //36/36/24/0

				// final sumation by last thread in every block
				d_sums[bx*d_mul*NUMBER_THREADS] = d_psum[tx];
				d_sums2[bx*d_mul*NUMBER_THREADS] = d_psum2[tx];

        //40/40/24/0
			}
		}
	}

}

// BUG IN SRAD APPLICATIONS SEEMS TO BE SOMEWHERE IN THIS CODE, WRONG MEMORY ACCESS

// srad kernel
__global__ void srad(	fp d_lambda,
									    int d_Nr,
									    int d_Nc,
									    long d_Ne,
									    int *d_iN,
									    int *d_iS,
									    int *d_jE,
									    int *d_jW,
									    fp *d_dN,
									    fp *d_dS,
									    fp *d_dE,
									    fp *d_dW,
									    fp d_q0sqr
									    fp *d_c,
									    fp *d_I){

	// indexes
  int bx = blockIdx.x;										// get current horizontal block index
	int tx = threadIdx.x;										// get current horizontal thread index
	int ei = bx*NUMBER_THREADS+tx;					// more threads than actual elements !!!
	int row;																// column, x position
	int col;																// row, y position

	// variables
	fp d_Jc;
	fp d_dN_loc, d_dS_loc, d_dW_loc, d_dE_loc;
	fp d_c_loc;
	fp d_G2,d_L,d_num,d_den,d_qsqr;

	// figure out row/col location in new matrix
	row = (ei+1) % d_Nr - 1;													// (0-n) row
	col = (ei+1) / d_Nr + 1 - 1;												// (0-n) column
	if((ei+1) % d_Nr == 0){
		row = d_Nr - 1;
		col = col - 1;
	}

	if(ei<d_Ne){															  // make sure that only threads matching jobs run

		                                          // directional derivatives, ICOV, diffusion coefficent
		d_Jc = d_I[ei];														// get value of the current element
    //1/1/0/0

		// directional derivates (every element of IMAGE)(try to copy to shared memory or temp files)
		d_dN_loc = d_I[d_iN[row] + d_Nr*col] - d_Jc;						// north direction derivative
		d_dS_loc = d_I[d_iS[row] + d_Nr*col] - d_Jc;						// south direction derivative
		d_dW_loc = d_I[row + d_Nr*d_jW[col]] - d_Jc;						// west direction derivative
		d_dE_loc = d_I[row + d_Nr*d_jE[col]] - d_Jc;						// east direction derivative
    //9/5/0/0 

		// normalized discrete gradient mag squared (equ 52,53)
		d_G2 = (d_dN_loc*d_dN_loc + d_dS_loc*d_dS_loc + d_dW_loc*d_dW_loc + d_dE_loc*d_dE_loc) / (d_Jc*d_Jc);	// gradient (based on derivatives)

		// normalized discrete laplacian (equ 54)
		d_L = (d_dN_loc + d_dS_loc + d_dW_loc + d_dE_loc) / d_Jc;			// laplacian (based on derivatives)

		// ICOV (equ 31/35)
		d_num  = (0.5*d_G2) - ((1.0/16.0)*(d_L*d_L)) ;						// num (based on gradient and laplacian)
		d_den  = 1 + (0.25*d_L);												// den (based on laplacian)
		d_qsqr = d_num/(d_den*d_den);										// qsqr (based on num and den)

		// diffusion coefficent (equ 33) (every element of IMAGE)
		d_den = (d_qsqr-d_q0sqr) / (d_q0sqr * (1+d_q0sqr)) ;				// den (based on qsqr and q0sqr)
		d_c_loc = 1.0 / (1.0+d_den) ;										// diffusion coefficient (based on den)

		// saturate diffusion coefficent to 0-1 range
		if (d_c_loc < 0){													// if diffusion coefficient < 0
			d_c_loc = 0;													// ... set to 0
		}
		else if (d_c_loc > 1){												// if diffusion coefficient > 1
			d_c_loc = 1;													// ... set to 1
		}

		// save data to global memory
		d_dN[ei] = d_dN_loc; 
		d_dS[ei] = d_dS_loc; 
		d_dW[ei] = d_dW_loc; 
		d_dE[ei] = d_dE_loc;
		d_c[ei] = d_c_loc;
    //14/5/0/0
	}
}

__global__ void srad2(	fp d_lambda, 
										int d_Nr, 
										int d_Nc, 
										long d_Ne, 
										int *d_iN, 
										int *d_iS, 
										int *d_jE, 
										int *d_jW,
										fp *d_dN, 
										fp *d_dS, 
										fp *d_dE, 
										fp *d_dW, 
										fp *d_c, 
										fp *d_I){

	// indexes
  int bx = blockIdx.x;													// get current horizontal block index
	int tx = threadIdx.x;													// get current horizontal thread index
	int ei = bx*NUMBER_THREADS+tx;											// more threads than actual elements !!!
	int row;																// column, x position
	int col;																// row, y position

	// variables
	fp d_cN,d_cS,d_cW,d_cE;
	fp d_D;

	// figure out row/col location in new matrix
	row = (ei+1) % d_Nr - 1;												// (0-n) row
	col = (ei+1) / d_Nr + 1 - 1;											// (0-n) column
	if((ei+1) % d_Nr == 0){
		row = d_Nr - 1;
		col = col - 1;
	}

	if(ei<d_Ne){															// make sure that only threads matching jobs run

		// diffusion coefficent
		d_cN = d_c[ei];														// north diffusion coefficient
		d_cS = d_c[d_iS[row] + d_Nr*col];										// south diffusion coefficient
		d_cW = d_c[ei];														// west diffusion coefficient
		d_cE = d_c[row + d_Nr * d_jE[col]];									// east diffusion coefficient
    // 6/4/0/0

		// divergence (equ 58)
		d_D = d_cN * d_dN[ei] + d_cS * d_dS[ei] + d_cW * d_dW[ei] + d_cE * d_dE[ei];// divergence
    // 10/8/0/0
    
		// image update (equ 61) (every element of IMAGE)
		d_I[ei] = d_I[ei] + 0.25*d_lambda*d_D;								// updates image (based on input time step and divergence)
    // 12/10/0/0
	}

}

#endif