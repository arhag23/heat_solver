#include <algorithm>
#include <cmath>
#include <cstdio>
#include <utility>

#include "solver.hpp"
#include "time_step.hpp"



void difference_recurrence(const char *file_path, const double *init, const double L,
                           const double T, const double DEL_X,
                           const double DEL_T, const int PRINT_WIDTH,
                           const double FRAME_LEN) {
  FILE *fp = std::fopen(file_path, "w+");
  const int X_LEN = static_cast<int>(std::ceil(L / DEL_X));
  const int print_stride = std::max(static_cast<int>(L / DEL_X / PRINT_WIDTH), 1);
  const int print_iter = static_cast<int>(FRAME_LEN / DEL_T);
  const int iter = static_cast<int>(ceil(T / DEL_T));

  double *prev = new double[X_LEN + 1];
  double *next = new double[X_LEN + 1];

  for (int i = 0; i <= X_LEN; i++) {
    prev[i] = init[i];
    if (i % print_stride == 0)
      fprintf(fp, "%.6f ", prev[i]);
  }
  fputc('\n', fp);

#if defined CUDA || defined HIP
  double *d_prev, *d_next;
  dim3 gridDim(X_LEN / 256 + 1, 1, 1);
  dim3 blockDim(256, 1, 1);
#ifdef CUDA
  cudaMalloc(&d_prev, sizeof(double) * (X_LEN + 1));
  cudaMalloc(&d_next, sizeof(double) * (X_LEN + 1));
  cudaMemcpy(d_prev, prev, sizeof(double) * (X_LEN + 1),
             cudaMemcpyHostToDevice);
#elif defined HIP
  hipMalloc(&d_prev, sizeof(double) * (X_LEN + 1));
  hipMalloc(&d_next, sizeof(double) * (X_LEN + 1));
  hipMemcpy(d_prev, prev, sizeof(double) * (X_LEN + 1), hipMemcpyHostToDevice);
#endif
#endif

  for (int i = 0; i < iter / print_iter + 1; i++) {
    for (int j = i * print_iter; j < std::min((i + 1) * print_iter, iter);
         j++) {
#if defined CUDA || defined HIP
      time_step<<<gridDim, blockDim>>>(d_prev, d_next, X_LEN, DEL_X, DEL_T);
      std::swap(d_prev, d_next);
#else
      time_step(prev, next, X_LEN, DEL_X, DEL_T);
      std::swap(prev, next);
#endif
    }
#ifdef CUDA
    cudaMemcpy(prev, d_prev, sizeof(double) * (X_LEN + 1),
               cudaMemcpyDeviceToHost);
#elif defined HIP
    cudaMemcpy(prev, d_prev, sizeof(double) * (X_LEN + 1),
               cudaMemcpyDeviceToHost);
#endif
    for (int j = 0; j <= X_LEN; j++)
      if (j % print_stride == 0)
        fprintf(fp, "%.6f ", prev[j]);
    fputc('\n', fp);
  }

#ifdef CUDA
  cudaFree(d_prev) cudaFree(d_next);
#elif defined HIP
  hipFree(d_prev) hipFree(d_next);
#endif

  delete[] prev;
  delete[] next;
  fclose(fp);
}