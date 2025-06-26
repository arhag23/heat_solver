#ifndef TIME_STEP_HPP_
#define TIME_STEP_HPP_

#ifdef CUDA
#include <cuda_runtime.h>
#elif HIP
#include <hip/hip_runtime.h>
#endif

#if defined CUDA || defined HIP
__global__ void time_step(const double *prev, double *next, const int X_LEN,
                          const double DEL_X, const double DEL_T);
#else
void time_step(const double *prev, double *next, const int X_LEN,
               const double DEL_X, const double DEL_T);
#endif

#endif