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

class NOP
{
public:
    __forceinline__ __device__ bool updateParameter(float &, unsigned int)
    {
        return false;
    }
};

//----------------------------------------------------------------------------
// Simple 'operation' to use with transpose and update  
// kernels to perform learning with a fixed rate
//----------------------------------------------------------------------------
class FixedLearningRate
{
public:
    FixedLearningRate(float *gradients, float learningRate)
    :   m_Gradients(gradients), m_LearningRate(learningRate)
    {
    }
    
    __forceinline__ __device__ bool updateParameter(float &param, unsigned int idx)
    {
        // Subtract gradient to parameter, scaled by learning rate
        param -= (m_Gradients[idx] * m_LearningRate);
        
        // Zero gradient
        m_Gradients[idx] = 0.0f;
        return true;
    }
    
private:
    float *m_Gradients;
    const float m_LearningRate;
};

//----------------------------------------------------------------------------
// Simple 'operation' to apply Adam optimizer to parameter in transpose and update kernels
//----------------------------------------------------------------------------
class AdamOptimizer
{
public:
    AdamOptimizer(float *gradients, float *m, float *v, unsigned int t, float alpha = 0.001, 
                  float beta1 = 0.9, float beta2 = 0.999, float epsilon = 1E-8)
    :   m_Gradients(gradients), m_M(m), m_V(v), m_Alpha(alpha), 
        m_Beta1(beta1), m_Beta2(beta2), m_Epsilon(epsilon), 
        m_FirstMomentScale(1.0f / (1.0f - pow(m_Beta1, t + 1))),
        m_SecondMomentScale(1.0f / (1.0f - pow(m_Beta2, t + 1)))
    {
    }
    
    __forceinline__ __device__ bool updateParameter(float &param, unsigned int idx)
    {
        // Get gradients
        const float gradient = m_Gradients[idx];
        
        // Update biased first moment estimate
        const float mT = (m_Beta1 * m_M[idx]) + ((1.0f - m_Beta1) * gradient);
        
        // Update biased second moment estimate
        const float vT = (m_Beta2 * m_V[idx]) + ((1.0f - m_Beta2) * gradient * gradient);
        
        // Add gradient to parameter, scaled by learning rate
        param -= (m_Alpha * mT * m_FirstMomentScale) / (sqrt(vT * m_SecondMomentScale) + m_Epsilon);
        
        // Write moments back to memory
        m_M[idx] = mT;
        m_V[idx] = vT;
        
        // Zero gradient
        m_Gradients[idx] = 0.0f;
        return true;
    }
    
private:
    float *m_Gradients;
    float *m_M;
    float *m_V;
    const float m_Alpha;
    const float m_Beta1;
    const float m_Beta2;
    const float m_Epsilon;
    const float m_FirstMomentScale;
    const float m_SecondMomentScale;
};

// Optimised CUDA transpose kernel based on https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/
template<typename Operation>
__global__ void transposeKernel(float *d_in, float *d_out, 
                                unsigned int numInRows, unsigned int numInCols,
                                Operation operation)
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
                    float gIn = d_in[idxIn];
                    
                    // Update parameter - if it's changed update global memory
                    if(operation.updateParameter(gIn, idxIn)) {
                        d_in[idxIn] = gIn;
                    }
                    
                    // Write forward weight to share memory
                    shTile[threadIdx.y + j][threadIdx.x] = gIn;
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

template<typename Operation>
__global__ void updateKernel(float *d_G, unsigned int numSynapses, Operation operation)
{
    const unsigned int idx = (blockIdx.x * 32) + threadIdx.x;
    if(idx < numSynapses) {
        // Update parameter - if it's changed update global memory
        float gIn = d_G[idx];
        if(operation.updateParameter(gIn, idx)) {
            d_G[idx] = gIn;
        }
    }
}
}   // Anonymous namespace

//----------------------------------------------------------------------------
// BatchLearning
//----------------------------------------------------------------------------
namespace BatchLearning
{
void transposeCUDA(float *d_in, float *d_out, unsigned int numInRows, unsigned int numInCols)
{
    // Calculate number of blocks required to process matrix
    const unsigned int numBlockX = (numInCols + TILE_DIM - 1) / TILE_DIM;
    const unsigned int numBlockY = (numInRows + TILE_DIM - 1) / TILE_DIM;
    
    NOP nop;
    const dim3 threads(32, BLOCK_HEIGHT);
    const dim3 grid(numBlockX, numBlockY);
    transposeKernel<<<grid, threads>>>(d_in, d_out, numInRows, numInCols, nop);
    CHECK_CUDA_ERRORS(cudaPeekAtLastError());
}

void fixedRateLearningCUDA(float *d_DeltaG, float *d_G, unsigned int numRows, unsigned int numCols, float learningRate)
{
    const unsigned int numSynapses = numRows * numCols;
    const unsigned int numBlocks = (numSynapses + 31) / 32;
    
    FixedLearningRate fixedLearningRate(d_DeltaG, learningRate);
    
    const dim3 threads(32, 1);
    const dim3 grid(numBlocks, 1);
    updateKernel<<<grid, threads>>>(d_G, numSynapses, fixedLearningRate);
    CHECK_CUDA_ERRORS(cudaPeekAtLastError());
}

void fixedRateLearningTransposeCUDA(float *d_DeltaGIn, float *d_GIn, float *d_GOut, unsigned int numInRows, unsigned int numInCols, float learningRate)
{
    // Calculate number of blocks required to process matrix
    const unsigned int numBlockX = (numInCols + TILE_DIM - 1) / TILE_DIM;
    const unsigned int numBlockY = (numInRows + TILE_DIM - 1) / TILE_DIM;
    
    FixedLearningRate fixedLearningRate(d_DeltaGIn, learningRate);
    
    const dim3 threads(32, BLOCK_HEIGHT);
    const dim3 grid(numBlockX, numBlockY);
    transposeKernel<<<grid, threads>>>(d_GIn, d_GOut, numInRows, numInCols, fixedLearningRate);
    CHECK_CUDA_ERRORS(cudaPeekAtLastError());
}

void adamOptimizerCUDA(float *d_DeltaG, float *d_M, float *d_V, float *d_G, 
                       unsigned int numRows, unsigned int numCols, unsigned int t, 
                       float alpha, float beta1, float beta2, float epsilon)
{
    const unsigned int numSynapses = numRows * numCols;
    const unsigned int numBlocks = (numSynapses + 31) / 32;
    
    AdamOptimizer adam(d_DeltaG, d_M, d_V, t, alpha, beta1, beta2, epsilon);
    
    const dim3 threads(32, 1);
    const dim3 grid(numBlocks, 1);
    updateKernel<<<grid, threads>>>(d_G, numSynapses, adam);
    CHECK_CUDA_ERRORS(cudaPeekAtLastError());
}

void adamOptimizerTransposeCUDA(float *d_DeltaGIn, float *d_MIn, float *d_VIn, float *d_GIn, 
                                float *d_GOut, unsigned int numInRows, unsigned int numInCols, 
                                unsigned int t, float alpha, float beta1, 
                                float beta2, float epsilon)
{
    // Calculate number of blocks required to process matrix
    const unsigned int numBlockX = (numInCols + TILE_DIM - 1) / TILE_DIM;
    const unsigned int numBlockY = (numInRows + TILE_DIM - 1) / TILE_DIM;
    
    AdamOptimizer adam(d_DeltaGIn, d_MIn, d_VIn, t, alpha, beta1, beta2, epsilon);
    
    const dim3 threads(32, BLOCK_HEIGHT);
    const dim3 grid(numBlockX, numBlockY);
    transposeKernel<<<grid, threads>>>(d_GIn, d_GOut, numInRows, numInCols, adam);
    CHECK_CUDA_ERRORS(cudaPeekAtLastError());
}
}   // namespace BatchLearning