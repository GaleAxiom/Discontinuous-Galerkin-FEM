#include "dgfem/reference/mapping.hpp"
#include "dgfem/solver/dg_solver.hpp"
#include "dgfem/solver/time_stepping.hpp"

#include <cmath>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
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

Teuchos::RCP<const TpetraCrsMatrix> CompressibleDGSolverBase::get_system_matrix() const {
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
    // Pre-size then index-write rather than push_back inside the loop: push_back is not
    // index-safe under any real parallel backend, even though it's fine on Serial.
    M_inv_blocks_.assign(n_elem, DView2{});

    Kokkos::parallel_for("compute_mass_inv_blocks", n_elem, [&](const int elem_id) {
        const auto& elem_data = mesh_->get_element_data(elem_id);
        DView2 M_local = weak_form_->compute_mass_integral(elem_data, mesh_->get_dg_space());
        M_inv_blocks_[elem_id] = invert_dense(M_local);
    });
}

CompressibleDGSolverBase::StateVector
CompressibleDGSolverBase::apply_mass_inv(const StateVector& vec) const {
    StateVector result(vec.size());
    for (size_t elem_id = 0; elem_id < vec.size(); ++elem_id) {
        const DView2& M_inv = M_inv_blocks_[elem_id];
        const DView2& v = vec[elem_id];
        result[elem_id] = DView2("mass_inv_result", v.extent(0), v.extent(1));
        gemm('N', 'N', 1.0, M_inv, v, 0.0, result[elem_id]);
    }
    return result;
}

CompressibleDGSolverBase::StateVector
CompressibleDGSolverBase::project_initial_condition(std::function<Vec4(const Vec2&)> u0_func) {
    int n_elem = mesh_->get_n_elements();
    int n_basis = mesh_->get_dg_space()->get_basis()->get_n_basis();
    int n_vars = weak_form_->get_n_vars();

    auto dg_space = mesh_->get_dg_space();
    auto mapping = dg_space->get_mapping();

    StateVector F_proj;
    F_proj.reserve(n_elem);

    for (int elem_id = 0; elem_id < n_elem; ++elem_id) {
        const auto& elem_data = mesh_->get_element_data(elem_id);

        DView2 vertices = mesh_->get_element_vertices(elem_id);

        const DView1& weights = dg_space->get_volume_quad()->weights;
        const DView2& phi = dg_space->get_volume_basis_values();
        const DView2& J_det = elem_data.at("J_det_vol");

        int n_quad = static_cast<int>(weights.size());
        DView2 F_local("F_local", n_basis, n_vars);

        for (int q = 0; q < n_quad; ++q) {
            Vec2 xi_q = row2(dg_space->get_volume_quad()->points, q);
            Vec2 x_q = mapping->map_to_physical(vertices, xi_q);
            Vec4 U_q = u0_func(x_q);
            double w_q = weights[q] * std::abs(J_det(q, 0));

            for (int i = 0; i < n_basis; ++i) {
                for (int v = 0; v < n_vars; ++v) {
                    F_local(i, v) += w_q * phi(q, i) * U_q[v];
                }
            }
        }

        F_proj.push_back(F_local);
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

    auto dg_space = mesh_->get_dg_space();
    const DView2& phi = dg_space->get_volume_basis_values();
    int n_quad = static_cast<int>(phi.extent(0));

    double max_cfl = 0.0;
    int invalid_found = 0;

    // Two reduction targets in one parallel_reduce: the CFL max, and a flag for
    // "invalid density/pressure encountered" -- the original code short-circuited with
    // `return infinity` from inside the per-element loop, which a reduce lambda can't do
    // (it can only return from itself, not the enclosing function), so the invalid case
    // is tracked as a second reduced value and checked once the reduce completes.
    Kokkos::parallel_reduce(
        "compute_max_cfl", n_elem,
        [&](const int elem_id, double& local_max, int& local_invalid) {
            DView2 vertices = mesh_->get_element_vertices(elem_id);
            int n_verts = static_cast<int>(vertices.extent(0));

            double h_elem = std::numeric_limits<double>::max();
            for (int i = 0; i < n_verts; ++i) {
                for (int j = i + 1; j < n_verts; ++j) {
                    double dist = norm(row2(vertices, i) - row2(vertices, j));
                    h_elem = std::min(h_elem, dist);
                }
            }

            double max_wave_speed = 0.0;
            const DView2& u_elem = u_coeffs[elem_id];
            DView2 U_quad("U_quad", n_quad, static_cast<int>(u_elem.extent(1)));
            gemm('N', 'N', 1.0, phi, u_elem, 0.0, U_quad);

            for (int q = 0; q < n_quad; ++q) {
                double rho = U_quad(q, 0);
                double rho_u = U_quad(q, 1);
                double rho_v = U_quad(q, 2);
                double E = U_quad(q, 3);

                if (rho <= 0.0 || !std::isfinite(rho)) {
                    local_invalid = 1;
                    return;
                }

                double u = rho_u / rho;
                double v = rho_v / rho;
                double kinetic = 0.5 * (u * u + v * v);
                double p = (gamma_ - 1.0) * (E - rho * kinetic);

                if (p <= 0.0 || !std::isfinite(p)) {
                    local_invalid = 1;
                    return;
                }

                double c = std::sqrt(std::max(gamma_ * p / rho, 0.0));
                double vel_mag = std::sqrt(u * u + v * v);
                double wave_speed = vel_mag + c;

                max_wave_speed = std::max(max_wave_speed, wave_speed);
            }

            double cfl_elem = dt * max_wave_speed / h_elem;
            local_max = std::max(local_max, cfl_elem);
        },
        Kokkos::Max<double, Kokkos::HostSpace>(max_cfl),
        Kokkos::Sum<int, Kokkos::HostSpace>(invalid_found));

    if (invalid_found > 0) {
        return std::numeric_limits<double>::infinity();
    }
    return max_cfl;
}

std::pair<double, double>
CompressibleDGSolverBase::compute_density_range(const StateVector& u_coeffs) const {
    auto dg_space = mesh_->get_dg_space();
    const DView2& phi = dg_space->get_volume_basis_values();
    int n_quad = static_cast<int>(phi.extent(0));

    double rho_min = std::numeric_limits<double>::infinity();
    double rho_max = std::numeric_limits<double>::lowest();

    Kokkos::parallel_reduce(
        "compute_density_range", u_coeffs.size(),
        [&](const size_t idx, double& local_min, double& local_max) {
            const auto& elem_coeffs = u_coeffs[idx];
            DView2 U_quad("U_quad", n_quad, static_cast<int>(elem_coeffs.extent(1)));
            gemm('N', 'N', 1.0, phi, elem_coeffs, 0.0, U_quad);
            for (int q = 0; q < n_quad; ++q) {
                double rho = U_quad(q, 0);
                if (!std::isfinite(rho)) {
                    continue;
                }
                local_min = std::min(local_min, rho);
                local_max = std::max(local_max, rho);
            }
        },
        Kokkos::Min<double, Kokkos::HostSpace>(rho_min),
        Kokkos::Max<double, Kokkos::HostSpace>(rho_max));

    if (!std::isfinite(rho_min) || !std::isfinite(rho_max)) {
        double nan_value = std::numeric_limits<double>::quiet_NaN();
        return {nan_value, nan_value};
    }

    return {rho_min, rho_max};
}

std::vector<DView2>
CompressibleDGSolverBase::run_time_integration(std::function<Vec4(const Vec2&)> initial_condition,
                                               double T_final, double dt, int save_every) {
    if (dt <= 0.0) {
        throw std::invalid_argument("Time step dt must be positive");
    }
    if (T_final <= 0.0) {
        throw std::invalid_argument("T_final must be positive");
    }
    diverged_ = false;

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
        auto [rho_min, rho_max] = compute_density_range(u_current);
        std::ostringstream msg;
        msg << "Initial CFL number = " << std::fixed << std::setprecision(4) << max_cfl_encountered
            << ", rho in [" << rho_min << ", " << rho_max << "]";
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
            auto [rho_min, rho_max] = compute_density_range(u_current);
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

        const bool all_ok =
            std::all_of(u_current.begin(), u_current.end(),
                        [](const DView2& elem_coeffs) { return all_finite(elem_coeffs); });

        if (!all_ok) {
            diverged_ = true;
            log(Stage::TimeStep,
                "ERROR: Solution contains non-finite (NaN or Inf) values; aborting integration.");
            break;
        }
    }

    // Always push the true final state (converged or diverged-and-broken-out-of) after the
    // loop. The in-loop push above only ever appends the state as of the *previous* iteration
    // (before that iteration's update), so it never pushes what u_current holds right now --
    // there's no risk of double-pushing here. Guarding this on `step % save_every != 0` (as it
    // used to be) meant the real final frame was silently dropped whenever save_every evenly
    // divided the step count, which it does for the default save_every=1.
    raw_frames.push_back(u_current);

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

std::vector<DView2>
CompressibleDGSolverBase::flatten_frames(const std::vector<StateVector>& frames) const {
    std::vector<DView2> flat_frames;
    flat_frames.reserve(frames.size());

    for (const auto& frame : frames) {
        int n_elem = static_cast<int>(frame.size());
        if (n_elem == 0) {
            flat_frames.emplace_back();
            continue;
        }

        int n_basis = static_cast<int>(frame[0].extent(0));
        int n_vars = static_cast<int>(frame[0].extent(1));

        DView2 flat_frame("flat_frame", n_elem, n_basis * n_vars);
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
