#include "dgfem/solver/dg_solver.hpp"

#include <Amesos2.hpp>
#include <cmath>

#include <sstream>
#include <stdexcept>

namespace dgfem {

using Stage = DGSolverBase::Stage;

LaplaceDGSolver::LaplaceDGSolver(std::shared_ptr<DGMesh> mesh, double penalty_parameter)
    : DGSolverBase(std::move(mesh), "Laplace"),
      weak_form_(std::make_shared<LaplaceWeakFormulation>(penalty_parameter)) {
    assembler_ = std::make_shared<DGAssembler>(mesh_, weak_form_);

    std::ostringstream oss;
    oss << "Initialized weak formulation with penalty parameter = " << penalty_parameter;
    log(Stage::Setup, oss.str());
}

DView1 LaplaceDGSolver::solve(std::function<double(const Vec2&)> source_func) {
    begin_stage(Stage::Assembly);
    assembler_->assemble(source_func);
    end_stage(Stage::Assembly);
    log(Stage::Assembly, "Assembled Laplace system.");

    auto system_matrix = assembler_->get_system_matrix();
    auto rhs = assembler_->get_rhs();

    if (system_matrix->getGlobalNumRows() == 0) {
        throw std::runtime_error("System matrix is empty");
    }

    begin_stage(Stage::Solve);
    auto x = Teuchos::rcp(new TpetraMultiVector(rhs->getMap(), 1));
    x->putScalar(0.0);

    // const_cast: Amesos2::create wants a non-const RCP<const CrsMatrix> is fine, but the
    // solver interface requires a non-const RHS/solution RCP pair with matching constness.
    auto solver = Amesos2::create<TpetraCrsMatrix, TpetraMultiVector>(
        "klu2", system_matrix, x, Teuchos::rcp_const_cast<TpetraMultiVector>(rhs));

    try {
        solver->symbolicFactorization().numericFactorization().solve();
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Laplace linear solve failed: ") + e.what());
    }

    end_stage(Stage::Solve);

    DView1 solution = tpetra_to_view(*x);

    std::ostringstream oss;
    oss << "Solver converged. Solution norm = " << norm(solution);
    log(Stage::Solve, oss.str());

    assembler_->distribute_solution(*x);
    increment_output_frames();

    return solution;
}

Teuchos::RCP<const TpetraCrsMatrix> LaplaceDGSolver::get_system_matrix() const {
    return assembler_->get_system_matrix();
}

Teuchos::RCP<const TpetraMultiVector> LaplaceDGSolver::get_rhs() const noexcept {
    return assembler_->get_rhs();
}

std::map<std::string, double>
LaplaceDGSolver::compute_error(std::function<double(const Vec2&)> exact_solution,
                               std::function<Vec2(const Vec2&)> exact_gradient) const {
    begin_stage(Stage::Postprocess);

    double L2_error_sq = 0.0;
    double H1_seminorm_sq = 0.0;

    auto dg_space = mesh_->get_dg_space();
    auto mapping = dg_space->get_mapping();
    auto mesh_solution = mesh_->get_solution();

    for (int elem_id = 0; elem_id < mesh_->get_n_elements(); ++elem_id) {
        const auto& elem_data = mesh_->get_element_data(elem_id);
        DView1 elem_coeffs = mesh_solution->get_element_coeffs(elem_id, 0);

        DView2 vertices = mesh_->get_element_vertices(elem_id);

        const DView1& weights = dg_space->get_volume_quad()->weights;
        const DView2& J_det = elem_data.at("J_det_vol");
        const DView2& phi = dg_space->get_volume_basis_values();
        const DView2& dphi_dx = elem_data.at("dphi_dx_vol");

        int n_quad = weights.extent(0);
        int n_basis = dg_space->get_basis()->get_n_basis();

        for (int q = 0; q < n_quad; ++q) {
            Vec2 xi_q = row2(dg_space->get_volume_quad()->points, q);
            Vec2 x_q = mapping->map_to_physical(vertices, xi_q);

            double u_h = 0.0;
            Vec2 grad_u_h{0.0, 0.0};

            for (int i = 0; i < n_basis; ++i) {
                u_h += elem_coeffs[i] * phi(q, i);
                grad_u_h += elem_coeffs[i] * row2(dphi_dx, q * n_basis + i);
            }

            double u_exact = exact_solution(x_q);
            Vec2 grad_u_exact = exact_gradient ? exact_gradient(x_q) : Vec2{0.0, 0.0};

            double w_q = weights[q] * std::abs(J_det(q, 0));

            L2_error_sq += w_q * std::pow(u_h - u_exact, 2.0);
            if (exact_gradient) {
                Vec2 diff = grad_u_h - grad_u_exact;
                H1_seminorm_sq += w_q * dot(diff, diff);
            }
        }
    }

    std::map<std::string, double> errors;
    errors["L2"] = std::sqrt(L2_error_sq);
    errors["H1"] = std::sqrt(H1_seminorm_sq + L2_error_sq);

    end_stage(Stage::Postprocess);

    std::ostringstream oss;
    oss << "Computed errors: L2 = " << errors["L2"] << ", H1 = " << errors["H1"];
    log(Stage::Postprocess, oss.str());

    return errors;
}

}  // namespace dgfem
