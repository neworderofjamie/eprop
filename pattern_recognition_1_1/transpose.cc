#include "transpose.h"

// CUDA includes
#include <curand_kernel.h>

// Auto-generated model code
#include "pattern_recognition_1_1_CODE/definitionsInternal.h"

// Anonymous namespace
namespace
{
// How large are (square) tiles used to calculate CUDA transpose
constexpr size_t TILE_DIM = 32;

// How 'high' are thread blocks
constexpr size_t BLOCK_HEIGHT = 8;

// Optimised CUDA transpose kernel based on https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/
__global__ void transposeCoalescedKernel(const float *d_in, float *d_out, unsigned int numInRows, unsigned int numInCols)
{
    // **NOTE** adding extra column prevents conflicts between 32 shared memory banks
    __shared__ float shTile[TILE_DIM][TILE_DIM + 1];

    {
        // Calculate coordinate of thread in input matrix
        const unsigned int x = (blockIdx.x * TILE_DIM) + threadIdx.x;
        const unsigned int y = (blockIdx.y * TILE_DIM) + threadIdx.y;
        
        // If thread isn't off the 'right' edge of the input matrix
        if(x < numInCols) {
            // Loop through input rows 
            for (unsigned int j = 0; j < TILE_DIM; j += BLOCK_HEIGHT) {
                // If thread isn't off the 'bottom' edge of the input matrix
                if((y + j) < numInRows) {
                    shTile[threadIdx.y + j][threadIdx.x] = d_in[((y + j) * numInCols) + x];
                }
            }
        }
    }
    
    __syncthreads();

    {
        // Calculate (transposed) coordinate of thread in output matrix
        const unsigned int x = (blockIdx.y * TILE_DIM) + threadIdx.x;
        const unsigned int y = (blockIdx.x * TILE_DIM) + threadIdx.y;
        
        // If thread isn't off the 'right' edge of the output matrix
        if(x < numInRows) {
            // Loop through output rows
            for (unsigned int j = 0; j < TILE_DIM; j += BLOCK_HEIGHT) {
                // If thread isn't off the 'bottom' edge of the output matrix
                if((y + j) < numInCols) {
                    d_out[((y + j) * numInRows) + x] = shTile[threadIdx.x][threadIdx.y + j];
                }
            }
        }
    }
}
}   // Anonymous namespace

//----------------------------------------------------------------------------
// BatchLearning
//----------------------------------------------------------------------------
namespace BatchLearning
{
void transposeCPU(const float *in, float *out, unsigned int numInRows, unsigned int numInCols)
{
    // Loop through rows and columns of matrix
    // **NOTE** this is super-naive
    for(unsigned int i = 0; i < numInRows; i++) {
        for(unsigned int j = 0; j < numInCols; j++) {
            out[(j * numInCols) + i] = in[(i * numInCols) + j];
        }
    }
}

void transposeCUDA(const float *d_in, float *d_out, unsigned int numInRows, unsigned int numInCols)
{
    // Calculate number of blocks required to process matrix
    const unsigned int numBlockX = (numInCols + TILE_DIM - 1) / TILE_DIM;
    const unsigned int numBlockY = (numInRows + TILE_DIM - 1) / TILE_DIM;
    
    const dim3 threads(32, BLOCK_HEIGHT);
    const dim3 grid(numBlockX, numBlockY);
    transposeCoalescedKernel<<<grid, threads>>>(d_in, d_out, numInRows, numInCols);
    CHECK_CUDA_ERRORS(cudaPeekAtLastError());
}
}   // namespace BatchLearning