#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

#include "init_conds.hpp"
#include "solver.hpp"

int main(int argc, char *argv[]) {
  if (argc < 6) {
    printf("Usage is HeatSolver <Length> <Time> <Length Delta> <Time Delta> "
           "<Print Length Delta> <Print Time Delta> <Solver>"
           "<Output file>\n");
    return 1;
  }

  double DEL_X = std::stod(argv[3]), DEL_T = std::stod(argv[4]),
         T = std::stod(argv[2]), L = std::stod(argv[1]),
         PDEL_X = std::stod(argv[5]), PDEL_T = std::stod(argv[6]);
  char *solver = argv[7];
  char *out_path = argv[8];

  const int X_LEN = static_cast<unsigned>(std::ceil(L / DEL_X));
  double *init = new double[X_LEN];
  for (int i = 0; i <= X_LEN; i++)
    init[i] = vshape(i * DEL_X, L);

  if (strcmp(solver, "finite_diff") == 0)
    difference_recurrence(out_path, init, L, T, DEL_X, DEL_T, PDEL_X, PDEL_T);
  else if (strcmp(solver, "finite_diff_scaled") == 0)
    difference_recurrence_scaled(out_path, init, L, T, DEL_X, DEL_T, PDEL_X,
                                 PDEL_T);
  else
    printf("Invalid solver \"%s\"\n", solver);

  delete[] init;
  return 0;
}
