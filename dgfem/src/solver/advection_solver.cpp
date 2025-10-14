#include "dgfem/reference/mapping.hpp"
#include "dgfem/solver/dg_solver.hpp"
#include "dgfem/solver/time_stepping.hpp"

#include <cmath>

#include <iomanip>
#include <sstream>

namespace dgfem {

using Stage = DGSolverBase::Stage;

AdvectionDGSolver::AdvectionDGSolver(std::shared_ptr<DGMesh> mesh,
                                     const Eigen::Vector2d& advection_velocity)
    : DGSolverBase(std::move(mesh), "Advection"),
      weak_form_(std::make_shared<AdvectionWeakFormulation>(advection_velocity)),
      advection_velocity_(advection_velocity) {
    assembler_ = std::make_shared<DGAssembler>(mesh_, weak_form_);

    std::ostringstream oss;
    oss << "Velocity = (" << advection_velocity_.transpose() << ")";
    log(Stage::Setup, oss.str());
}

const Eigen::SparseMatrix<double>& AdvectionDGSolver::get_system_matrix() const {
    return assembler_->get_system_matrix();
}

std::vector<Eigen::VectorXd>
AdvectionDGSolver::solve(std::function<double(const Eigen::Vector2d&)> initial_condition,
                         double T_final, double dt,
                         std::shared_ptr<BoundaryCondition> boundary_condition, int save_every) {
    std::ostringstream setup_msg;
    setup_msg << "Config: T_final = " << T_final << ", dt = " << dt
              << ", save_every = " << save_every;
    log(Stage::Setup, setup_msg.str());

    begin_stage(Stage::Projection);
    compute_mass_matrix_inverse_blocks();
    Eigen::VectorXd u = project_initial_condition(initial_condition);
    end_stage(Stage::Projection);
    log(Stage::Projection, "Projected initial condition onto DG space.");

    auto mesh_solution = mesh_->get_solution();
    mesh_solution->set_global_coeffs(u);

    begin_stage(Stage::Assembly);
    auto bc_func = [boundary_condition](const Eigen::Vector2d& x) -> double {
        if (boundary_condition) {
            return boundary_condition->evaluate(x);
        }
        return 0.0;
    };

    assembler_->assemble(nullptr, bc_func);
    L_operator_ = assembler_->get_system_matrix();
    F_boundary_ = assembler_->get_rhs();
    end_stage(Stage::Assembly);
    log(Stage::Assembly, "Assembled advection operator and boundary flux.");

    begin_stage(Stage::TimeStep);

    std::vector<Eigen::VectorXd> solution_frames;
    double t = 0.0;
    int step = 0;
    int n_steps = static_cast<int>(std::ceil(T_final / dt));

    solution_frames.push_back(u);

    while (t < T_final) {
        if (step % save_every == 0 && step > 0) {
            solution_frames.push_back(u);
        }

        u = time_step_ssp_rk3(u, dt);
        t += dt;
        step++;
        increment_steps();
    }

    if (solution_frames.back() != u) {
        solution_frames.push_back(u);
    }

    mesh_solution->set_global_coeffs(u);

    end_stage(Stage::TimeStep);

    increment_output_frames(static_cast<int>(solution_frames.size()));

    std::ostringstream timestep_msg;
    timestep_msg << "Completed time stepping in " << n_steps << " steps. Saved "
                 << solution_frames.size() << " frames.";
    log(Stage::TimeStep, timestep_msg.str());

    log(Stage::Output, "Updated mesh solution coefficients with final state.");

    return solution_frames;
}

void AdvectionDGSolver::compute_mass_matrix_inverse_blocks() {
    Eigen::SparseMatrix<double> M = assembler_->assemble_mass_matrix();

    int n_basis = mesh_->get_dg_space()->get_basis()->get_n_basis();
    M_inv_blocks_.clear();
    M_inv_blocks_.reserve(mesh_->get_n_elements());

    for (int elem_id = 0; elem_id < mesh_->get_n_elements(); ++elem_id) {
        int start = elem_id * n_basis;
        Eigen::MatrixXd M_block(n_basis, n_basis);
        for (int i = 0; i < n_basis; ++i) {
            for (int j = 0; j < n_basis; ++j) {
                M_block(i, j) = M.coeff(start + i, start + j);
            }
        }
        M_inv_blocks_.push_back(M_block.inverse());
    }
}

Eigen::VectorXd AdvectionDGSolver::apply_mass_inv(const Eigen::VectorXd& vec) const {
    int n_basis = mesh_->get_dg_space()->get_basis()->get_n_basis();
    Eigen::VectorXd result = Eigen::VectorXd::Zero(vec.size());

    for (int elem_id = 0; elem_id < mesh_->get_n_elements(); ++elem_id) {
        int start = elem_id * n_basis;
        result.segment(start, n_basis) = M_inv_blocks_[elem_id] * vec.segment(start, n_basis);
    }

    return result;
}

Eigen::VectorXd AdvectionDGSolver::project_initial_condition(
    std::function<double(const Eigen::Vector2d&)> u0_func) {
    auto dg_space = mesh_->get_dg_space();
    auto mapping = dg_space->get_mapping();
    int n_basis = dg_space->get_basis()->get_n_basis();

    Eigen::VectorXd F_proj = Eigen::VectorXd::Zero(mesh_->get_n_elements() * n_basis);

    for (int elem_id = 0; elem_id < mesh_->get_n_elements(); ++elem_id) {
        Eigen::MatrixXd vertices(mesh_->get_elements().cols(), 2);
        for (int i = 0; i < mesh_->get_elements().cols(); ++i) {
            vertices.row(i) = mesh_->get_vertices().row(mesh_->get_elements()(elem_id, i));
        }

        const Eigen::MatrixXd& phi = dg_space->get_volume_basis_values();
        const Eigen::VectorXd& weights = dg_space->get_volume_quad()->weights;
        const auto& elem_data = mesh_->get_element_data(elem_id);
        const Eigen::VectorXd& J_det = elem_data.at("J_det_vol");

        int n_quad = weights.size();
        Eigen::VectorXd F_local = Eigen::VectorXd::Zero(n_basis);

        for (int q = 0; q < n_quad; ++q) {
            Eigen::Vector2d xi_q = dg_space->get_volume_quad()->points.row(q);
            Eigen::Vector2d x_q = mapping->map_to_physical(vertices, xi_q);
            double u0_val = u0_func(x_q);
            double w_q = weights[q] * std::abs(J_det[q]);

            F_local += w_q * u0_val * phi.row(q).transpose();
        }

        int start = elem_id * n_basis;
        F_proj.segment(start, n_basis) = F_local;
    }

    return apply_mass_inv(F_proj);
}

Eigen::VectorXd AdvectionDGSolver::time_step_ssp_rk3(const Eigen::VectorXd& u_n, double dt) const {
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> rhs_func =
        [this](const Eigen::VectorXd& u) { return compute_rhs(u); };
    return SSP_RK::step_rk3<Eigen::VectorXd>(u_n, dt, rhs_func);
}

Eigen::VectorXd AdvectionDGSolver::compute_rhs(const Eigen::VectorXd& u) const {
    increment_rhs_evaluations();
    Eigen::VectorXd rhs = L_operator_ * u + F_boundary_;
    return apply_mass_inv(rhs);
}

}  // namespace dgfem
