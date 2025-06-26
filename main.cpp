#include <cstdlib>
#include <cmath>

#include "solver.hpp"
#include "init_conds.hpp"

int main(int argc, char* argv[]) {
    double DEL_X = std::atof(argv[1]), DEL_T = std::atof(argv[2]), T = std::atof(argv[3]), L = std::atof(argv[4]);
    char* out_path = argv[5];

    const int X_LEN = static_cast<unsigned>(std::ceil(L / DEL_X));
    double* init = new double[X_LEN];
    for (int i = 0; i <= X_LEN; i++)
        init[i] = vshape(i * DEL_X, L);

    difference_recurrence("out/data.out", init, L, T, DEL_X, DEL_T, 500, 0.05);

    delete[] init;
}