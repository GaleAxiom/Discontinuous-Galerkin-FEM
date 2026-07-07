/**
 * @file euler_eigensystem.cpp
 * @brief Implementation of the x-direction 1D Euler flux-Jacobian eigensystem provider.
 */

#include "dgfem/solver/euler_eigensystem.hpp"

#include "dgfem/weak_forms/euler_weak_formulation.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace dgfem {

namespace {
constexpr int kRhoIdx = 0, kRhoUIdx = 1, kRhoVIdx = 2, kEIdx = 3;
}  // namespace

EulerXDirectionEigensystem::EulerXDirectionEigensystem(int n_vars) {
    config_.coupled_indices = {kRhoIdx, kRhoUIdx, kEIdx};
    for (int i = 0; i < n_vars; ++i) {
        if (i != kRhoIdx && i != kRhoUIdx && i != kEIdx) {
            config_.passive_indices.push_back(i);
        }
    }
}

LocalEigensystem
EulerXDirectionEigensystem::build(const std::vector<double>& direction,
                                  const std::vector<double>& cell_avg_primitives,
                                  double gamma) const {
    if (direction.size() != 2 || direction[0] != 1.0 || direction[1] != 0.0) {
        throw std::invalid_argument(
            "EulerXDirectionEigensystem::build: only direction {1.0, 0.0} is supported "
            "this pass (a genuine multi-direction provider does not exist yet).");
    }

    double rho = cell_avg_primitives[kRhoIdx];
    double u = cell_avg_primitives[kRhoUIdx];
    double p = cell_avg_primitives[kEIdx];

    double rho_safe = std::max(rho, 1e-12);
    double c = std::sqrt(std::max(gamma * p / rho_safe, 1e-24));
    double H = c * c / (gamma - 1.0) + 0.5 * u * u;

    LocalEigensystem es;
    es.n_coupled = 3;
    es.R = {{1.0, 1.0, 1.0}, {u - c, u, u + c}, {H - u * c, 0.5 * u * u, H + u * c}};

    double b1 = (gamma - 1.0) / (c * c);
    double b2 = 0.5 * b1 * u * u;
    es.L = {{(b2 + u / c) / 2.0, -(b1 * u + 1.0 / c) / 2.0, b1 / 2.0},
            {1.0 - b2, b1 * u, -b1},
            {(b2 - u / c) / 2.0, -(b1 * u - 1.0 / c) / 2.0, b1 / 2.0}};
    return es;
}

std::vector<double> cell_average_primitives(const DView2& u_elem, int n_vars, double gamma) {
    Vec4 cons4{u_elem(0, kRhoIdx), u_elem(0, kRhoUIdx), u_elem(0, kRhoVIdx), u_elem(0, kEIdx)};
    Vec4 prim4 = conserved_to_primitive(cons4, gamma);
    std::vector<double> prim(n_vars);
    prim[kRhoIdx] = prim4[0];
    prim[kRhoUIdx] = prim4[1];
    prim[kRhoVIdx] = prim4[2];
    prim[kEIdx] = prim4[3];
    double rho_safe = std::max(prim4[0], 1e-12);
    for (int v = 4; v < n_vars; ++v) {
        prim[v] = u_elem(0, v) / rho_safe;
    }
    return prim;
}

}  // namespace dgfem
