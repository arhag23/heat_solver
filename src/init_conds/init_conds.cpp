#include <cmath>

#include "init_conds.hpp"

constexpr double PI = 3.1415926535897931;

double linear(const double x) {
    return x;
}

double vshape(const double x, const double L) {
    return std::fabs(x - L / 2);
}

double periodic(const double x, const double period) {
    return std::cos(2 * PI / period * x) + 1.5;
}
