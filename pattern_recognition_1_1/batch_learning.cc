#include "batch_learning.h"

// Standard C++ includes
#include <iostream>
#include <numeric>

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

__device__ unsigned int g_NumDormantConnections;

class NOP
{
public:
    __forceinline__ __device__ bool updateParameter(float &, unsigned int)
    {
        return false;
    }
    
    __forceinline__ __device__ void moveParams(unsigned int, unsigned int)
    {
    }
};

//----------------------------------------------------------------------------
// FixedLearningRate
//----------------------------------------------------------------------------
//! Simple 'operation' to use with transpose and update kernels to perform learning with a fixed rate
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
    
    __forceinline__ __device__ void moveParams(unsigned int srcIdx, unsigned int dstIdx)
    {
        m_Gradients[dstIdx] = m_Gradients[srcIdx];
    }
    
private:
    float *m_Gradients;
    const float m_LearningRate;
};

//----------------------------------------------------------------------------
// AdamOptimizer
//----------------------------------------------------------------------------
//! Simple 'operation' to apply Adam optimizer to parameter in transpose and update kernels
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
    
    __forceinline__ __device__ void moveParams(unsigned int srcIdx, unsigned int dstIdx)
    {
        m_Gradients[dstIdx] = m_Gradients[srcIdx];
        m_M[dstIdx] = m_M[srcIdx];
        m_V[dstIdx] = m_M[dstIdx];
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

__global__ void buildDeepRBitmaskKernel(uint32_t *d_Bitmask, const unsigned int *d_RowLength,
                                        unsigned int numRows, unsigned int maxRowLength)
{
    const unsigned int idPre = (blockIdx.x * 32) + threadIdx.x;
}

template<typename Operation>
__global__ void deepRFirstPassKernel(float *d_G, float *d_EFiltered, unsigned int *d_RowLength, unsigned int *d_Ind, 
                                     unsigned int numRows, unsigned int maxRowLength, 
                                     Operation operation)
{
    const unsigned int idPre = (blockIdx.x * 32) + threadIdx.x;
    
    // Use first thread in block to zero shared memory dormant counter
     __shared__ unsigned int shNumDormant;
    if(threadIdx.x == 0) {
        shNumDormant = 0;
    }
    __syncthreads();
    
    // If there's a row for this thread to process
    if(idPre < numRows) {
        // Loop through synapses
        unsigned int numDormant = 0;
        unsigned int rowLength = d_RowLength[idPre];
        const unsigned int rowStartIdx = idPre * maxRowLength;
        for(unsigned int j = 0; j < rowLength; j++) {
            const unsigned int idx = rowStartIdx + j;
            
            // Cache parameter and its sign in register
            float gIn = d_G[idx];
            const auto oldSign = signbit(gIn);
            
            // If update changes parameter
            // **TODO** L1 regularizer
            if(operation.updateParameter(gIn, idx)) {
                // If sign hasn't changed, update weight in memory
                if(signbit(gIn) == oldSign) {
                    d_G[idx] = gIn;
                }
                // Otherwise, make connection dormant
                else {
                    // Calculate index of last synapse in row
                    const unsigned int rowLastIdx = rowStartIdx + rowLength - 1;
                    
                    // Overwrite this synapse with one at end of row
                    d_Ind[idx] = d_Ind[rowLastIdx];
                    d_G[idx] = d_G[rowLastIdx];
                    d_EFiltered[idx] = d_EFiltered[rowLastIdx];
                    
                    // Instruct operation to do the same for any of its parameters
                    operation.moveParams(rowLastIdx, idx);
                    
                    // Decrement row length
                    rowLength--;
                    
                    // Increment row's dormant counter
                    numDormant++;
                }
            }
        }
        
        // Write back updated row length
        d_RowLength[idPre] = rowLength;
        
        // Update shared memory dormant synapse count
        if(numDormant > 0) {
            atomicAdd(&shNumDormant, numDormant);
        }
    }
    
    // Use first thread in block to atomic add shared memory counter to global total
    __syncthreads();
    if(threadIdx.x == 0 && shNumDormant > 0) {
        atomicAdd(&g_NumDormantConnections, shNumDormant);
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

void adamOptimizerDeepRCUDA(float *d_DeltaG, float *d_M, float *d_V, 
                            float *d_G, float *d_EFiltered,
                            unsigned int *rowLength, unsigned int *d_RowLength, unsigned int *d_Ind, 
                            unsigned int numRows, unsigned int maxRowLength, unsigned int t, 
                            std::mt19937 &rng, float alpha, float beta1, float beta2, float epsilon)
{
    // Calculate number of blocks required to process matrix
    const unsigned int numBlocks = (numRows + 31) / 32;
    
    AdamOptimizer adam(d_DeltaG, d_M, d_V, t, alpha, beta1, beta2, epsilon);
    
    // Zero device dormant count
    unsigned int numDormant = 0;
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbol(g_NumDormantConnections, &numDormant, sizeof(unsigned int)));
    
    // Launch kernel to perform first Deep-R pass
    const dim3 threads(32, 1);
    const dim3 grid(numBlocks, 1);
    deepRFirstPassKernel<<<grid, threads>>>(d_G, d_EFiltered, d_RowLength, d_Ind, 
                                            numRows, maxRowLength, adam);
    
    // Copy device dormant count back to host
    CHECK_CUDA_ERRORS(cudaMemcpyFromSymbol(&numDormant, g_NumDormantConnections, sizeof(unsigned int)));
    std::cout << numDormant << " synapses made dormant" << std::endl;
    
    // Copy row lengths back to host
    CHECK_CUDA_ERRORS(cudaMemcpy(rowLength, d_RowLength, numRows * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    
    // Count number of synapses
    const size_t numSynapses = std::accumulate(&rowLength[0], &rowLength[numRows], 0u);
    
    // From this, calculate how many padding synapses there are in data structure
    size_t numTotalPaddingSynapses = (maxRowLength * numRows) - numSynapses;
    std::cout << numTotalPaddingSynapses << " empty synapses" << std::endl;
    
    // Loop through rows of synaptic matrix
    for(unsigned int i = 0; i < (numRows - 1); i++) {
        const unsigned int numRowPaddingSynapses = maxRowLength - rowLength[i];
        const double probability = (double)numRowPaddingSynapses / (double)numTotalPaddingSynapses;

        // Create distribution to sample number of activations
        std::binomial_distribution<size_t> numActivationDist(numDormant, probability);

        // Sample number of activations
        const size_t numActivations = numActivationDist(rng);
        
        std::cout << "\t" << numActivations << std::endl;
        
        // Update counters
        numDormant -= numActivations;
        numTotalPaddingSynapses -= numRowPaddingSynapses;
    }
    assert(numDormant < maxRowLength - rowLength[numRows - 1]);
    std::cout << "\t" << numDormant << std::endl;
    
    //CHECK_CUDA_ERRORS(cudaPeekAtLastError());
    
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