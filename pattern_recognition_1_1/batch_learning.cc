#include "batch_learning.h"

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
__global__ void transposeKernel(const float *d_in, float *d_out, 
                                unsigned int numInRows, unsigned int numInCols)
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

__global__ void fixedRateLearningKernel(float *d_DeltaG, float *d_G, 
                                        unsigned int numSynapses, float learningRate)
{
    const unsigned int id = (blockIdx.x * 32) + threadIdx.x;
    if(id < numSynapses) {
        // Update weight
        d_G[id] += learningRate * d_DeltaG[id];
        
        // Zero delta
        d_DeltaG[id] = 0.0f;
    }
}
// Optimised CUDA transpose kernel based on https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/
__global__ void fixedRateLearningTransposeKernel(float *d_DeltaGIn, float *d_GIn, float *d_GOut, 
                                                 unsigned int numInRows, unsigned int numInCols, 
                                                 float learningRate)
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
                    // Read forward weight from global memory
                    const unsigned int idxIn = ((y + j) * numInCols) + x;
                    float gIn = d_GIn[idxIn];
                    
                    // Apply fixed learning rate learning
                    gIn += learningRate * d_DeltaGIn[idxIn];
                    
                    // Write to shared memory to transpose
                    shTile[threadIdx.y + j][threadIdx.x] = gIn;
                    
                    // Write updated weight back to shared memory
                    d_GIn[idxIn] = gIn;
                    
                    // Zero deltaGs
                    d_DeltaGIn[idxIn] = 0.0f;
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
                    d_GOut[((y + j) * numInRows) + x] = shTile[threadIdx.x][threadIdx.y + j];
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
void transposeCUDA(const float *d_in, float *d_out, unsigned int numInRows, unsigned int numInCols)
{
    // Calculate number of blocks required to process matrix
    const unsigned int numBlockX = (numInCols + TILE_DIM - 1) / TILE_DIM;
    const unsigned int numBlockY = (numInRows + TILE_DIM - 1) / TILE_DIM;
    
    const dim3 threads(32, BLOCK_HEIGHT);
    const dim3 grid(numBlockX, numBlockY);
    transposeKernel<<<grid, threads>>>(d_in, d_out, numInRows, numInCols);
    CHECK_CUDA_ERRORS(cudaPeekAtLastError());
}

void fixedRateLearningCUDA(float *d_DeltaG, float *d_G, unsigned int numRows, unsigned int numCols, float learningRate)
{
    const unsigned int numSynapses = numRows * numCols;
    const unsigned int numBlocks = (numSynapses + 31) / 32;
    
    const dim3 threads(32, 1);
    const dim3 grid(numBlocks, 1);
    fixedRateLearningKernel<<<grid, threads>>>(d_DeltaG, d_G, numSynapses, learningRate);
    CHECK_CUDA_ERRORS(cudaPeekAtLastError());
}

void fixedRateLearningTransposeCUDA(float *d_DeltaGIn, float *d_GIn, float *d_GOut, unsigned int numInRows, unsigned int numInCols, float learningRate)
{
    // Calculate number of blocks required to process matrix
    const unsigned int numBlockX = (numInCols + TILE_DIM - 1) / TILE_DIM;
    const unsigned int numBlockY = (numInRows + TILE_DIM - 1) / TILE_DIM;
    
    const dim3 threads(32, BLOCK_HEIGHT);
    const dim3 grid(numBlockX, numBlockY);
    fixedRateLearningTransposeKernel<<<grid, threads>>>(d_DeltaGIn, d_GIn, d_GOut, numInRows, numInCols, learningRate);
    CHECK_CUDA_ERRORS(cudaPeekAtLastError());
}

}   // namespace BatchLearning