#include "dgfem/reference/mapping.hpp"
#include "dgfem/solver/dg_solver.hpp"
#include "dgfem/solver/time_stepping.hpp"

#include <cmath>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <limits>
#include <sstream>
#include <utility>

namespace dgfem {

using Stage = DGSolverBase::Stage;

CompressibleDGSolverBase::CompressibleDGSolverBase(std::shared_ptr<DGMesh> mesh,
                                                   std::shared_ptr<EulerWeakFormulation> weak_form,
                                                   std::string solver_label)
    : DGSolverBase(std::move(mesh), solver_label), weak_form_(std::move(weak_form)),
      gamma_(weak_form_->get_gamma()), solver_label_(std::move(solver_label)) {
    assembler_ = std::make_shared<DGAssembler>(mesh_, weak_form_);

    std::ostringstream oss;
    oss << "Gamma = " << gamma_ << ", variables = " << weak_form_->get_n_vars();
    log(Stage::Setup, oss.str());
}

const Eigen::SparseMatrix<double>& CompressibleDGSolverBase::get_system_matrix() const {
    return assembler_->get_system_matrix();
}

CompressibleDGSolverBase::StateVector
CompressibleDGSolverBase::assemble_residual(const StateVector& u_coeffs) const {
    increment_rhs_evaluations();
    assembler_->assemble_euler_residual(u_coeffs, residual_buffer_);
    return residual_buffer_;
}

void CompressibleDGSolverBase::compute_mass_matrix_inverse_blocks() {
    int n_elem = mesh_->get_n_elements();
    M_inv_blocks_.clear();
    M_inv_blocks_.reserve(n_elem);

    for (int elem_id = 0; elem_id < n_elem; ++elem_id) {
        const auto& elem_data = mesh_->get_element_data(elem_id);
        Eigen::MatrixXd M_local =
            weak_form_->compute_mass_integral(elem_data, mesh_->get_dg_space());
        M_inv_blocks_.push_back(M_local.inverse());
    }
}

CompressibleDGSolverBase::StateVector
CompressibleDGSolverBase::apply_mass_inv(const StateVector& vec) const {
    StateVector result(vec.size());
    for (size_t elem_id = 0; elem_id < vec.size(); ++elem_id) {
        result[elem_id] = M_inv_blocks_[elem_id] * vec[elem_id];
    }
    return result;
}

CompressibleDGSolverBase::StateVector CompressibleDGSolverBase::project_initial_condition(
    std::function<Eigen::Vector4d(const Eigen::Vector2d&)> u0_func) {
    int n_elem = mesh_->get_n_elements();
    int n_basis = mesh_->get_dg_space()->get_basis()->get_n_basis();
    int n_vars = weak_form_->get_n_vars();

    auto dg_space = mesh_->get_dg_space();
    auto mapping = dg_space->get_mapping();

    StateVector F_proj(n_elem, Eigen::MatrixXd::Zero(n_basis, n_vars));

    for (int elem_id = 0; elem_id < n_elem; ++elem_id) {
        const auto& elem_data = mesh_->get_element_data(elem_id);

        Eigen::MatrixXd vertices(mesh_->get_elements().cols(), 2);
        for (int i = 0; i < mesh_->get_elements().cols(); ++i) {
            vertices.row(i) = mesh_->get_vertices().row(mesh_->get_elements()(elem_id, i));
        }

        const Eigen::VectorXd& weights = dg_space->get_volume_quad()->weights;
        const Eigen::MatrixXd& phi = dg_space->get_volume_basis_values();
        const Eigen::VectorXd& J_det = elem_data.at("J_det_vol");

        int n_quad = weights.size();
        Eigen::MatrixXd F_local = Eigen::MatrixXd::Zero(n_basis, n_vars);

        for (int q = 0; q < n_quad; ++q) {
            Eigen::Vector2d xi_q = dg_space->get_volume_quad()->points.row(q);
            Eigen::Vector2d x_q = mapping->map_to_physical(vertices, xi_q);
            Eigen::Vector4d U_q = u0_func(x_q);
            double w_q = weights[q] * std::abs(J_det[q]);

            for (int i = 0; i < n_basis; ++i) {
                F_local.row(i) += w_q * phi(q, i) * U_q.transpose();
            }
        }

        F_proj[elem_id] = F_local;
    }

    return apply_mass_inv(F_proj);
}

CompressibleDGSolverBase::StateVector
CompressibleDGSolverBase::time_step_ssp_rk3(const StateVector& u_n, double dt) const {
    std::function<StateVector(const StateVector&)> rhs_func = [this](const StateVector& u) {
        return apply_mass_inv(assemble_residual(u));
    };
    return SSP_RK::step_rk3<StateVector>(u_n, dt, rhs_func);
}

double CompressibleDGSolverBase::compute_max_cfl(const StateVector& u_coeffs, double dt) const {
    int n_elem = mesh_->get_n_elements();
    int n_basis = mesh_->get_dg_space()->get_basis()->get_n_basis();
    double max_cfl = 0.0;

    for (int elem_id = 0; elem_id < n_elem; ++elem_id) {
        Eigen::MatrixXd vertices(mesh_->get_elements().cols(), 2);
        for (int i = 0; i < mesh_->get_elements().cols(); ++i) {
            vertices.row(i) = mesh_->get_vertices().row(mesh_->get_elements()(elem_id, i));
        }

        double h_elem = std::numeric_limits<double>::max();
        for (int i = 0; i < vertices.rows(); ++i) {
            for (int j = i + 1; j < vertices.rows(); ++j) {
                double dist = (vertices.row(i) - vertices.row(j)).norm();
                h_elem = std::min(h_elem, dist);
            }
        }

        double max_wave_speed = 0.0;
        for (int i = 0; i < n_basis; ++i) {
            double rho = u_coeffs[elem_id](i, 0);
            double rho_u = u_coeffs[elem_id](i, 1);
            double rho_v = u_coeffs[elem_id](i, 2);
            double E = u_coeffs[elem_id](i, 3);

            double u = rho_u / rho;
            double v = rho_v / rho;
            double p = (gamma_ - 1.0) * (E - 0.5 * rho * (u * u + v * v));

            double c = std::sqrt(gamma_ * p / rho);
            double vel_mag = std::sqrt(u * u + v * v);
            double wave_speed = vel_mag + c;

            max_wave_speed = std::max(max_wave_speed, wave_speed);
        }

        double cfl_elem = dt * max_wave_speed / h_elem;
        max_cfl = std::max(max_cfl, cfl_elem);
    }

    return max_cfl;
}

std::vector<Eigen::MatrixXd> CompressibleDGSolverBase::run_time_integration(
    std::function<Eigen::Vector4d(const Eigen::Vector2d&)> initial_condition, double T_final,
    double dt, int save_every) {
    std::ostringstream intro;
    intro << "Starting " << solver_label_ << " integration (T_final = " << T_final
          << ", dt = " << dt << ", save_every = " << save_every << ")";
    log(Stage::Setup, intro.str());

    begin_stage(Stage::Setup);
    const auto mass_start = std::chrono::high_resolution_clock::now();
    compute_mass_matrix_inverse_blocks();
    const auto mass_end = std::chrono::high_resolution_clock::now();
    end_stage(Stage::Setup);

    {
        std::ostringstream msg;
        msg << "Mass matrix inverse blocks computed in " << std::fixed << std::setprecision(4)
            << std::chrono::duration<double>(mass_end - mass_start).count() << " s";
        log(Stage::Setup, msg.str());
    }

    begin_stage(Stage::Projection);
    const auto proj_start = std::chrono::high_resolution_clock::now();
    StateVector u_current = project_initial_condition(initial_condition);
    const auto proj_end = std::chrono::high_resolution_clock::now();
    end_stage(Stage::Projection);

    {
        std::ostringstream msg;
        msg << "Initial state projected in " << std::fixed << std::setprecision(4)
            << std::chrono::duration<double>(proj_end - proj_start).count() << " s";
        log(Stage::Projection, msg.str());
    }

    begin_stage(Stage::TimeStep);

    std::vector<StateVector> raw_frames;
    raw_frames.push_back(u_current);

    double t = 0.0;
    int step = 0;
    const int n_steps_target = static_cast<int>(std::ceil(T_final / dt));

    double max_cfl_encountered = compute_max_cfl(u_current, dt);
    double total_timestep_time = 0.0;

    {
        std::ostringstream msg;
        msg << "Initial CFL number = " << std::fixed << std::setprecision(4) << max_cfl_encountered;
        log(Stage::TimeStep, msg.str());
    }
    if (max_cfl_encountered > 1.0) {
        log(Stage::TimeStep, "WARNING: Initial CFL exceeds 1.0; results may be unstable.");
    }

    while (t < T_final) {
        if (step % save_every == 0 && step > 0) {
            raw_frames.push_back(u_current);
        }

        if (step % 10 == 0) {
            double rho_min = std::numeric_limits<double>::max();
            double rho_max = std::numeric_limits<double>::lowest();

            for (const auto& elem_coeffs : u_current) {
                rho_min = std::min(rho_min, elem_coeffs.col(0).minCoeff());
                rho_max = std::max(rho_max, elem_coeffs.col(0).maxCoeff());
            }

            double current_cfl = compute_max_cfl(u_current, dt);
            max_cfl_encountered = std::max(max_cfl_encountered, current_cfl);

            std::ostringstream progress;
            progress << std::fixed << std::setprecision(3);
            progress << "Step " << step << " / " << n_steps_target << ", t = " << t << " (rho in ["
                     << rho_min << ", " << rho_max << "], CFL = " << current_cfl << ")";
            log(Stage::TimeStep, progress.str());

            if (current_cfl > 1.0 && step % 50 == 0) {
                std::ostringstream warn;
                warn << "WARNING: CFL = " << current_cfl << " > 1.0 at step " << step;
                log(Stage::TimeStep, warn.str());
            }
        }

        const auto step_start = std::chrono::high_resolution_clock::now();
        u_current = time_step_ssp_rk3(u_current, dt);
        const auto step_end = std::chrono::high_resolution_clock::now();
        total_timestep_time += std::chrono::duration<double>(step_end - step_start).count();

        t += dt;
        step++;
        increment_steps();

        bool has_nan = false;
        for (const auto& elem_coeffs : u_current) {
            if (elem_coeffs.hasNaN()) {
                has_nan = true;
                break;
            }
        }

        if (has_nan) {
            log(Stage::TimeStep, "ERROR: Solution contains NaN values; aborting integration.");
            break;
        }
    }

    if (step % save_every != 0) {
        raw_frames.push_back(u_current);
    }

    end_stage(Stage::TimeStep);

    increment_output_frames(static_cast<int>(raw_frames.size()));

    {
        std::ostringstream summary;
        summary << solver_label_ << " completed with " << raw_frames.size() << " frames and "
                << step << " steps. Total stepping time = " << std::fixed << std::setprecision(4)
                << total_timestep_time << " s (avg " << std::setprecision(6)
                << (step > 0 ? total_timestep_time / step : 0.0) << " s/step).";
        log(Stage::TimeStep, summary.str());
    }

    {
        std::ostringstream msg;
        msg << "Maximum CFL encountered = " << std::fixed << std::setprecision(4)
            << max_cfl_encountered;
        if (max_cfl_encountered > 1.0) {
            msg << " (WARNING: > 1.0)";
        }
        log(Stage::TimeStep, msg.str());
    }

    begin_stage(Stage::Output);
    auto flattened = flatten_frames(raw_frames);
    end_stage(Stage::Output);

    log(Stage::Output, "Flattened solution frames for post-processing.");

    return flattened;
}

std::vector<Eigen::MatrixXd>
CompressibleDGSolverBase::flatten_frames(const std::vector<StateVector>& frames) const {
    std::vector<Eigen::MatrixXd> flat_frames;
    flat_frames.reserve(frames.size());

    for (const auto& frame : frames) {
        int n_elem = static_cast<int>(frame.size());
        if (n_elem == 0) {
            flat_frames.emplace_back();
            continue;
        }

        int n_basis = frame[0].rows();
        int n_vars = frame[0].cols();

        Eigen::MatrixXd flat_frame(n_elem, n_basis * n_vars);
        for (int e = 0; e < n_elem; ++e) {
            for (int i = 0; i < n_basis; ++i) {
                for (int v = 0; v < n_vars; ++v) {
                    flat_frame(e, i * n_vars + v) = frame[e](i, v);
                }
            }
        }

        flat_frames.push_back(flat_frame);
    }

    return flat_frames;
}

}  // namespace dgfem
