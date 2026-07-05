#include "dgfem/solver/dg_solver.hpp"
#include "dgfem/weak_forms/navier_stokes_weak_formulation.hpp"

#include <utility>

namespace dgfem {

NavierStokesDGSolver::NavierStokesDGSolver(std::shared_ptr<DGMesh> mesh, double gamma,
                                           double dynamic_viscosity, double prandtl,
                                           double penalty_prefactor)
    : CompressibleDGSolverBase(std::move(mesh),
                               std::make_shared<NavierStokesWeakFormulation>(
                                   gamma, dynamic_viscosity, prandtl, penalty_prefactor),
                               "Navier-Stokes") {}

std::vector<DView2> NavierStokesDGSolver::solve(std::function<Vec4(const Vec2&)> initial_condition,
                                                double T_final, double dt, int save_every) {
    return run_time_integration(std::move(initial_condition), T_final, dt, save_every);
}

}  // namespace dgfem
