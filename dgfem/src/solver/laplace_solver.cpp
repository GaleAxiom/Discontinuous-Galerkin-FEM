#include "dgfem/solver/dg_solver.hpp"

#include <Eigen/SparseLU>
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

Eigen::VectorXd LaplaceDGSolver::solve(std::function<double(const Eigen::Vector2d&)> source_func) {
    begin_stage(Stage::Assembly);
    assembler_->assemble(source_func);
    end_stage(Stage::Assembly);
    log(Stage::Assembly, "Assembled Laplace system.");

    const auto& system_matrix = assembler_->get_system_matrix();
    const auto& rhs = assembler_->get_rhs();

    if (system_matrix.rows() == 0) {
        throw std::runtime_error("System matrix is empty");
    }

    begin_stage(Stage::Solve);
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.compute(system_matrix);

    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Matrix factorization failed");
    }

    Eigen::VectorXd solution = solver.solve(rhs);

    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Linear solve failed");
    }

    end_stage(Stage::Solve);

    std::ostringstream oss;
    oss << "Solver converged. Solution norm = " << solution.norm();
    log(Stage::Solve, oss.str());

    assembler_->distribute_solution(solution);
    increment_output_frames();

    return solution;
}

const Eigen::SparseMatrix<double>& LaplaceDGSolver::get_system_matrix() const {
    return assembler_->get_system_matrix();
}

const Eigen::VectorXd& LaplaceDGSolver::get_rhs() const noexcept {
    return assembler_->get_rhs();
}

std::map<std::string, double> LaplaceDGSolver::compute_error(
    std::function<double(const Eigen::Vector2d&)> exact_solution,
    std::function<Eigen::Vector2d(const Eigen::Vector2d&)> exact_gradient) const {
    begin_stage(Stage::Postprocess);

    double L2_error_sq = 0.0;
    double H1_seminorm_sq = 0.0;

    auto dg_space = mesh_->get_dg_space();
    auto mapping = dg_space->get_mapping();
    auto mesh_solution = mesh_->get_solution();

    for (int elem_id = 0; elem_id < mesh_->get_n_elements(); ++elem_id) {
        const auto& elem_data = mesh_->get_element_data(elem_id);
        Eigen::VectorXd elem_coeffs = mesh_solution->get_element_coeffs(elem_id, 0);

        Eigen::MatrixXd vertices(mesh_->get_elements().cols(), 2);
        for (int i = 0; i < mesh_->get_elements().cols(); ++i) {
            vertices.row(i) = mesh_->get_vertices().row(mesh_->get_elements()(elem_id, i));
        }

        const Eigen::VectorXd& weights = dg_space->get_volume_quad()->weights;
        const Eigen::VectorXd& J_det = elem_data.at("J_det_vol");
        const Eigen::MatrixXd& phi = dg_space->get_volume_basis_values();
        const Eigen::MatrixXd& dphi_dx = elem_data.at("dphi_dx_vol");

        int n_quad = weights.size();
        int n_basis = dg_space->get_basis()->get_n_basis();

        for (int q = 0; q < n_quad; ++q) {
            Eigen::Vector2d xi_q = dg_space->get_volume_quad()->points.row(q);
            Eigen::Vector2d x_q = mapping->map_to_physical(vertices, xi_q);

            double u_h = 0.0;
            Eigen::Vector2d grad_u_h = Eigen::Vector2d::Zero();

            for (int i = 0; i < n_basis; ++i) {
                u_h += elem_coeffs[i] * phi(q, i);
                grad_u_h += elem_coeffs[i] * dphi_dx.row(q * n_basis + i).transpose();
            }

            double u_exact = exact_solution(x_q);
            Eigen::Vector2d grad_u_exact =
                exact_gradient ? exact_gradient(x_q) : Eigen::Vector2d::Zero();

            double w_q = weights[q] * std::abs(J_det[q]);

            L2_error_sq += w_q * std::pow(u_h - u_exact, 2.0);
            if (exact_gradient) {
                H1_seminorm_sq += w_q * (grad_u_h - grad_u_exact).squaredNorm();
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
