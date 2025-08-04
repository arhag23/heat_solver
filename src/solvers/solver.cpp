#include <algorithm>
#include <cmath>
#include <cstdio>
#include <utility>

#include "solver.hpp"
#include "time_step.hpp"

void difference_recurrence(const char *file_path, const double *init,
                           const double L, const double T, const double DEL_X,
                           const double DEL_T, const double PRINT_DEL_X,
                           const double PRINT_DEL_T) {
  FILE *fp = std::fopen(file_path, "w+");
  const int X_LEN = static_cast<int>(std::ceil(L / DEL_X));
  const int print_stride = std::max(static_cast<int>(PRINT_DEL_X / DEL_X), 1);
  const int print_iter = std::max(static_cast<int>(PRINT_DEL_T / DEL_T), 1);
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
  cudaFree(d_prev);
  cudaFree(d_next);
#elif defined HIP
  hipFree(d_prev);
  hipFree(d_next);
#endif

  delete[] prev;
  delete[] next;
  fclose(fp);
}

void difference_recurrence_scaled(const char *file_path, const double *init,
                                  const double L, const double T,
                                  const double DEL_X, const double DEL_T,
                                  const double PRINT_DEL_X,
                                  const double PRINT_DEL_T) {
  FILE *fp = std::fopen(file_path, "w+");
  const double T_ACT = 2 * T / DEL_X;
  const int X_LEN = static_cast<int>(std::ceil(L / DEL_X));
  const int print_stride = std::max(static_cast<int>(PRINT_DEL_X / DEL_X), 1);
  const int print_iter =
      std::max(static_cast<int>(2 * PRINT_DEL_T / (DEL_T * DEL_X)), 1);
  const int iter = static_cast<int>(ceil(T_ACT / DEL_T));

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
      time_step_scaled<<<gridDim, blockDim>>>(d_prev, d_next, X_LEN, DEL_X,
                                              DEL_T);
      std::swap(d_prev, d_next);
#else
      time_step_scaled(prev, next, X_LEN, DEL_X, DEL_T);
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
  cudaFree(d_prev);
  cudaFree(d_next);
#elif defined HIP
  hipFree(d_prev);
  hipFree(d_next);
#endif

  delete[] prev;
  delete[] next;
  fclose(fp);
}
