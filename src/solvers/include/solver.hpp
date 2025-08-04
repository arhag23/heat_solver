#ifndef SOLVER_HPP_
#define SOLVER_HPP_

void difference_recurrence(const char *file_path, const double *init,
                           const double L, const double T, const double DEL_X,
                           const double DEL_T, const double PRINT_DEL_X = 0.1,
                           const double PRINT_DEL_T = 0.05);

void difference_recurrence_scaled(const char *file_path, const double *init,
                                  const double L, const double T,
                                  const double DEL_X, const double DEL_T,
                                  const double PRINT_DEL_X = 0.1,
                                  const double PRINT_DEL_T = 0.05);

#endif
