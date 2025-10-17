#include "dgfem/solver/dg_solver.hpp"
#include "dgfem/weak_forms/euler_weak_formulation.hpp"

#include <memory>
#include <utility>

namespace dgfem {

EulerDGSolver::EulerDGSolver(std::shared_ptr<DGMesh> mesh, double gamma)
    : CompressibleDGSolverBase(std::move(mesh), std::make_shared<EulerWeakFormulation>(gamma),
                               "Euler") {}

EulerDGSolver::EulerDGSolver(std::shared_ptr<DGMesh> mesh,
                             std::shared_ptr<EulerWeakFormulation> weak_form)
    : CompressibleDGSolverBase(std::move(mesh), std::move(weak_form), "Euler") {}

std::vector<Eigen::MatrixXd>
EulerDGSolver::solve(std::function<Eigen::Vector4d(const Eigen::Vector2d&)> initial_condition,
                     double T_final, double dt, int save_every) {
    return run_time_integration(std::move(initial_condition), T_final, dt, save_every);
}

}  // namespace dgfem
