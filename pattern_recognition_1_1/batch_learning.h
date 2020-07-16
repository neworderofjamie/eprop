#pragma once

//----------------------------------------------------------------------------
// BatchLearning
//----------------------------------------------------------------------------
namespace BatchLearning
{
//! Calculate transpose of matrix using CPU
void transposeCPU(const float *in, float *out, unsigned int numInRows, unsigned int numInCols);

//! Calculate transpose of matrix using CUDA
void transposeCUDA(const float *d_in, float *d_out, unsigned int numInRows, unsigned int numInCols);
}