#pragma once

//----------------------------------------------------------------------------
// BatchLearning
//----------------------------------------------------------------------------
namespace BatchLearning
{
//! Calculate transpose of matrix using CUDA
void transposeCUDA(float *d_in, float *d_out, 
                   unsigned int numInRows, unsigned int numInCols);

//! Apply fixed rate learning to dense weights
void fixedRateLearningCUDA(float *d_DeltaG, float *d_G, 
                           unsigned int numRows, unsigned int numCols, 
                           float learningRate);

//! Apply fixed rate learning to dense weights and then transfer to transpose
void fixedRateLearningTransposeCUDA(float *d_DeltaGIn, float *d_GIn, float *d_GOut, 
                                    unsigned int numInRows, unsigned int numInCols, 
                                    float learningRate);
}