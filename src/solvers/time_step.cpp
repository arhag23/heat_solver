#include "time_step.hpp"

#if defined CUDA || defined HIP
__global__ void time_step(const double *prev, double *next, const int X_LEN,
                          const double DEL_X, const double DEL_T) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i == 0) {
    next[0] = 2 * DEL_T * (prev[1] - prev[0]) / (DEL_X * DEL_X) + prev[0];
  } else if (i == X_LEN) {
    next[X_LEN] =
        2 * DEL_T * (prev[X_LEN - 1] - prev[X_LEN]) / (DEL_X * DEL_X) + prev[X_LEN];
  } else if (i < X_LEN) {
    double uxx = (prev[i + 1] + prev[i - 1] - 2 * prev[i]) / (DEL_X * DEL_X);
    next[i] = DEL_T * uxx + prev[i];
  }
}
#else
void time_step(const double *prev, double *next, const int X_LEN,
               const double DEL_X, const double DEL_T) {
  next[0] = 2 * DEL_T * (prev[1] - prev[0]) / (2 * DEL_X * DEL_X) + prev[0];
  next[X_LEN] =
      2 * DEL_T * (prev[X_LEN - 1] - prev[X_LEN]) / (2 * DEL_X * DEL_X) + prev[X_LEN];
  // next[0] = DEL_T * (prev[1] - prev[0]) / (DEL_X) + prev[0];
  // next[X_LEN] = DEL_T * (prev[X_LEN - 1] - prev[X_LEN]) / (DEL_X) +
  // prev[X_LEN];
  for (int i = 1; i < X_LEN; i++) {
    double uxx = (prev[i + 1] + prev[i - 1] - 2 * prev[i]) / (DEL_X * DEL_X);
    next[i] = DEL_T * uxx + prev[i];
  }
}
#endif