/**
 * @file dg_solver.cpp
 * @brief Implementation of DG solvers
 */

#include "dgfem/solver/dg_solver.hpp"
#include "dgfem/solver/time_stepping.hpp"
#include "dgfem/core/space.hpp"
#include "dgfem/reference/mapping.hpp"
#include "dgfem/weak_forms/navier_stokes_weak_formulation.hpp"
#include <Eigen/SparseLU>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <utility>

namespace dgfem {

// DGSolverBase implementation
DGSolverBase::DGSolverBase(std::shared_ptr<DGMesh> mesh)
    : mesh_(mesh) {}

// LaplaceDGSolver implementation
LaplaceDGSolver::LaplaceDGSolver(std::shared_ptr<DGMesh> mesh, double penalty_parameter)
    : DGSolverBase(mesh), 
      weak_form_(std::make_shared<LaplaceWeakFormulation>(penalty_parameter)) {
    
    assembler_ = std::make_shared<DGAssembler>(mesh_, weak_form_);
}

Eigen::VectorXd LaplaceDGSolver::solve(std::function<double(const Eigen::Vector2d&)> source_func) {
    // Assemble system
    assembler_->assemble(source_func);

    
    std::cout << "\nSolving linear system..." << std::endl;
    
    const auto& system_matrix = assembler_->get_system_matrix();
    const auto& rhs = assembler_->get_rhs();
    
    if (system_matrix.rows() == 0) {
        throw std::runtime_error("System matrix is empty");
    }
    
    // Solve using SparseLU
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.compute(system_matrix);
    
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Matrix factorization failed");
    }
    
    Eigen::VectorXd solution = solver.solve(rhs);
    
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Linear solve failed");
    }
    
    std::cout << "Solver converged. Solution norm: " << solution.norm() << std::endl;
    
    // Distribute solution back to mesh
    assembler_->distribute_solution(solution);
    
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
    
    double L2_error_sq = 0.0;
    double H1_seminorm_sq = 0.0;
    
    auto dg_space = mesh_->get_dg_space();
    auto mapping = dg_space->get_mapping();
    auto mesh_solution = mesh_->get_solution();
    
    for (int elem_id = 0; elem_id < mesh_->get_n_elements(); ++elem_id) {
        // Get element data
        const auto& elem_data = mesh_->get_element_data(elem_id);
        Eigen::VectorXd elem_coeffs = mesh_solution->get_element_coeffs(elem_id, 0);
        
        // Get element vertices
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
            // Get physical coordinates
            Eigen::Vector2d xi_q = dg_space->get_volume_quad()->points.row(q);
            Eigen::Vector2d x_q = mapping->map_to_physical(vertices, xi_q);
            
            // Compute numerical solution and gradient
            double u_h = 0.0;
            Eigen::Vector2d grad_u_h = Eigen::Vector2d::Zero();
            
            for (int i = 0; i < n_basis; ++i) {
                u_h += elem_coeffs[i] * phi(q, i);
                grad_u_h += elem_coeffs[i] * dphi_dx.row(q * n_basis + i).transpose();
            }
            
            // Exact solution and gradient
            double u_exact = exact_solution(x_q);
            Eigen::Vector2d grad_u_exact = exact_gradient ? exact_gradient(x_q) : Eigen::Vector2d::Zero();
            
            // Integration weight
            double w_q = weights[q] * std::abs(J_det[q]);
            
            // Error contributions
            L2_error_sq += w_q * std::pow(u_h - u_exact, 2.0);
            if (exact_gradient) {
                H1_seminorm_sq += w_q * (grad_u_h - grad_u_exact).squaredNorm();
            }
        }
    }
    
    std::map<std::string, double> errors;
    errors["L2"] = std::sqrt(L2_error_sq);
    errors["H1"] = std::sqrt(H1_seminorm_sq + L2_error_sq);
    
    return errors;
}

// AdvectionDGSolver implementation
AdvectionDGSolver::AdvectionDGSolver(std::shared_ptr<DGMesh> mesh, const Eigen::Vector2d& advection_velocity)
    : DGSolverBase(mesh), 
      weak_form_(std::make_shared<AdvectionWeakFormulation>(advection_velocity)),
      advection_velocity_(advection_velocity) {
    
    assembler_ = std::make_shared<DGAssembler>(mesh_, weak_form_);
}

const Eigen::SparseMatrix<double>& AdvectionDGSolver::get_system_matrix() const {
    return assembler_->get_system_matrix();
}

std::vector<Eigen::VectorXd> AdvectionDGSolver::solve(
    std::function<double(const Eigen::Vector2d&)> initial_condition,
    double T_final,
    double dt,
    std::shared_ptr<BoundaryCondition> boundary_condition,
    int save_every) {
    
    std::cout << "\nSolving advection equation with time stepping..." << std::endl;
    std::cout << "T_final = " << T_final << ", dt = " << dt << std::endl;
    
    // 1. Project initial condition using proper L2 projection
    std::cout << "  Projecting initial condition..." << std::endl;
    compute_mass_matrix_inverse_blocks();
    Eigen::VectorXd u = project_initial_condition(initial_condition);
    
    // Store in mesh solution
    auto mesh_solution = mesh_->get_solution();
    mesh_solution->set_global_coeffs(u);
    
    // 2. Assemble spatial operator L and boundary forcing F_bc
    std::cout << "  Assembling spatial operator..." << std::endl;
    auto bc_func = [boundary_condition](const Eigen::Vector2d& x) -> double {
        if (boundary_condition) {
            return boundary_condition->evaluate(x);
        }
        return 0.0;
    };
    
    assembler_->assemble(nullptr, bc_func);
    L_operator_ = assembler_->get_system_matrix();
    F_boundary_ = assembler_->get_rhs();

    
    // 3. Time stepping loop
    std::cout << "  Starting time stepping (T_final=" << T_final << ", dt=" << dt << ")..." << std::endl;
    
    std::vector<Eigen::VectorXd> solution_frames;
    double t = 0.0;
    int step = 0;
    int n_steps = static_cast<int>(std::ceil(T_final / dt));
    
    // Save initial condition
    solution_frames.push_back(u);
    
    while (t < T_final) {
        if (step % save_every == 0 && step > 0) {
            solution_frames.push_back(u);
            if (step % (10 * save_every) == 0) {
                std::cout << "    t = " << t << " / " << T_final 
                          << "  (u_min=" << u.minCoeff() << ", u_max=" << u.maxCoeff() << ")" << std::endl;
            }
        }
        
        // SSP-RK3 time step
        u = time_step_ssp_rk3(u, dt);
        t += dt;
        step++;
    }
    
    // Save final solution
    if (solution_frames.back() != u) {
        solution_frames.push_back(u);
    }
    
    // Set final solution back to mesh
    mesh_solution->set_global_coeffs(u);
    
    std::cout << "Time stepping completed. Saved " << solution_frames.size() << " frames." << std::endl;
    
    return solution_frames;
}

void AdvectionDGSolver::compute_mass_matrix_inverse_blocks() {
    // Assemble global mass matrix
    Eigen::SparseMatrix<double> M = assembler_->assemble_mass_matrix();
    
    int n_basis = mesh_->get_dg_space()->get_basis()->get_n_basis();
    M_inv_blocks_.clear();
    M_inv_blocks_.reserve(mesh_->get_n_elements());
    
    // Extract and invert each element block
    for (int elem_id = 0; elem_id < mesh_->get_n_elements(); ++elem_id) {
        int start = elem_id * n_basis;
        int end = (elem_id + 1) * n_basis;
        
        // Extract block as dense matrix
        Eigen::MatrixXd M_block(n_basis, n_basis);
        for (int i = 0; i < n_basis; ++i) {
            for (int j = 0; j < n_basis; ++j) {
                M_block(i, j) = M.coeff(start + i, start + j);
            }
        }
        
        // Invert and store
        M_inv_blocks_.push_back(M_block.inverse());
    }
}

Eigen::VectorXd AdvectionDGSolver::apply_mass_inv(const Eigen::VectorXd& vec) const {
    int n_basis = mesh_->get_dg_space()->get_basis()->get_n_basis();
    Eigen::VectorXd result = Eigen::VectorXd::Zero(vec.size());
    
    for (int elem_id = 0; elem_id < mesh_->get_n_elements(); ++elem_id) {
        int start = elem_id * n_basis;
        int end = (elem_id + 1) * n_basis;
        
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
    
    // Assemble projection RHS: F[i] = ∫ u0(x) * phi_i dx
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
    
    // Solve M * u0 = F_proj for u0
    return apply_mass_inv(F_proj);
}

Eigen::VectorXd AdvectionDGSolver::time_step_ssp_rk3(const Eigen::VectorXd& u_n, double dt) const {
    // Use refactored SSP-RK3 from time_stepping utilities
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> rhs_func = 
        [this](const Eigen::VectorXd& u) { return compute_rhs(u); };
    return SSP_RK::step_rk3<Eigen::VectorXd>(u_n, dt, rhs_func);
}

Eigen::VectorXd AdvectionDGSolver::compute_rhs(const Eigen::VectorXd& u) const {
    // Compute RHS of M * du/dt = L*u + F_bc
    // Returns du/dt = M^{-1} * (L*u + F_bc)
    Eigen::VectorXd rhs = L_operator_ * u + F_boundary_;
    return apply_mass_inv(rhs);
}

// EulerDGSolver implementation
EulerDGSolver::EulerDGSolver(std::shared_ptr<DGMesh> mesh, double gamma)
        : EulerDGSolver(mesh, std::make_shared<EulerWeakFormulation>(gamma)) {}

EulerDGSolver::EulerDGSolver(std::shared_ptr<DGMesh> mesh, std::shared_ptr<EulerWeakFormulation> weak_form)
        : DGSolverBase(mesh),
            weak_form_(std::move(weak_form)),
            gamma_(weak_form_->get_gamma()) {
        assembler_ = std::make_shared<DGAssembler>(mesh_, weak_form_);
}

const Eigen::SparseMatrix<double>& EulerDGSolver::get_system_matrix() const {
    return assembler_->get_system_matrix();
}

void EulerDGSolver::compute_mass_matrix_inverse_blocks() {
    std::cout << "    Computing mass matrix inverse blocks..." << std::endl;
    
    int n_elem = mesh_->get_n_elements();
    int n_basis = mesh_->get_dg_space()->get_basis()->get_n_basis();
    
    M_inv_blocks_.clear();
    M_inv_blocks_.reserve(n_elem);
    
    for (int elem_id = 0; elem_id < n_elem; ++elem_id) {
        const auto& elem_data = mesh_->get_element_data(elem_id);
        Eigen::MatrixXd M_local = weak_form_->compute_mass_integral(elem_data, mesh_->get_dg_space());
        
        // Invert the mass matrix
        Eigen::MatrixXd M_inv = M_local.inverse();
        M_inv_blocks_.push_back(M_inv);
    }
}

std::vector<Eigen::MatrixXd> EulerDGSolver::apply_mass_inv(
    const std::vector<Eigen::MatrixXd>& vec) const {
    
    int n_elem = vec.size();
    std::vector<Eigen::MatrixXd> result(n_elem);
    
    for (int elem_id = 0; elem_id < n_elem; ++elem_id) {
        result[elem_id] = M_inv_blocks_[elem_id] * vec[elem_id];
    }
    
    return result;
}

std::vector<Eigen::MatrixXd> EulerDGSolver::project_initial_condition(
    std::function<Eigen::Vector4d(const Eigen::Vector2d&)> u0_func) {
    
    std::cout << "    Projecting initial condition..." << std::endl;
    
    int n_elem = mesh_->get_n_elements();
    int n_basis = mesh_->get_dg_space()->get_basis()->get_n_basis();
    int n_vars = 4;
    
    auto dg_space = mesh_->get_dg_space();
    auto mapping = dg_space->get_mapping();
    
    std::vector<Eigen::MatrixXd> F_proj(n_elem, Eigen::MatrixXd::Zero(n_basis, n_vars));
    
    for (int elem_id = 0; elem_id < n_elem; ++elem_id) {
        const auto& elem_data = mesh_->get_element_data(elem_id);
        
        // Get element vertices
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

std::vector<Eigen::MatrixXd> EulerDGSolver::assemble_euler_residual(
    const std::vector<Eigen::MatrixXd>& u_coeffs) const {
    
    return assembler_->assemble_euler_residual(u_coeffs);
}

std::vector<Eigen::MatrixXd> EulerDGSolver::time_step_ssp_rk3(
    const std::vector<Eigen::MatrixXd>& u_n,
    double dt) const {
    
    // Use refactored SSP-RK3 from time_stepping utilities
    std::function<std::vector<Eigen::MatrixXd>(const std::vector<Eigen::MatrixXd>&)> rhs_func = 
        [this](const std::vector<Eigen::MatrixXd>& u) {
            auto R = assemble_euler_residual(u);
            return apply_mass_inv(R);
        };
    return SSP_RK::step_rk3<std::vector<Eigen::MatrixXd>>(u_n, dt, rhs_func);
}

std::vector<Eigen::MatrixXd> EulerDGSolver::solve(
    std::function<Eigen::Vector4d(const Eigen::Vector2d&)> initial_condition,
    double T_final,
    double dt,
    int save_every) {
    
    std::cout << "\n=== Solving Euler equations with time stepping ===" << std::endl;
    std::cout << "T_final = " << T_final << ", dt = " << dt << std::endl;
    
    // 1. Compute mass matrix inverse blocks
    auto mass_start = std::chrono::high_resolution_clock::now();
    compute_mass_matrix_inverse_blocks();
    auto mass_end = std::chrono::high_resolution_clock::now();
    std::cout << "[TIMER] Mass matrix computation: " << std::fixed << std::setprecision(4) 
              << std::chrono::duration<double>(mass_end - mass_start).count() << " s" << std::endl;
    
    // 2. Project initial condition
    auto proj_start = std::chrono::high_resolution_clock::now();
    std::vector<Eigen::MatrixXd> u_current = project_initial_condition(initial_condition);
    auto proj_end = std::chrono::high_resolution_clock::now();
    std::cout << "[TIMER] Initial condition projection: " << std::fixed << std::setprecision(4) 
              << std::chrono::duration<double>(proj_end - proj_start).count() << " s" << std::endl;
    
    // 3. Time stepping loop
    std::cout << "  Starting time stepping (SSP-RK3, T_final=" << T_final << ", dt=" << dt << ")..." << std::endl;
    
    std::vector<std::vector<Eigen::MatrixXd>> solution_frames;
    double t = 0.0;
    int step = 0;
    
    // Save initial condition
    solution_frames.push_back(u_current);
    
    // Compute and display initial CFL number
    double initial_cfl = compute_max_cfl(u_current, dt);
    std::cout << "  Initial CFL number: " << initial_cfl << std::endl;
    if (initial_cfl > 1.0) {
        std::cout << "  WARNING: CFL > 1.0, solution may be unstable!" << std::endl;
    }
    
    double max_cfl_encountered = initial_cfl;
    
    // Timing accumulators
    double total_residual_time = 0.0;
    double total_mass_inv_time = 0.0;
    double total_timestep_time = 0.0;
    
    while (t < T_final) {
        if (step % save_every == 0 && step > 0) {
            solution_frames.push_back(u_current);
        }
        
        // Print progress every 10 steps (more frequent to show it's actually running)
        if (step % 10 == 0) {
            // Monitor solution quality
            double rho_min = std::numeric_limits<double>::max();
            double rho_max = std::numeric_limits<double>::lowest();
            
            for (const auto& elem_coeffs : u_current) {
                for (int i = 0; i < elem_coeffs.rows(); ++i) {
                    double rho = elem_coeffs(i, 0);
                    rho_min = std::min(rho_min, rho);
                    rho_max = std::max(rho_max, rho);
                }
            }
            
            // Compute current CFL number
            double current_cfl = compute_max_cfl(u_current, dt);
            max_cfl_encountered = std::max(max_cfl_encountered, current_cfl);
            
            std::cout << "    Step " << step << ", t = " << t << " / " << T_final 
                      << "  (rho: [" << rho_min << ", " << rho_max << "], CFL: " 
                      << std::fixed << std::setprecision(3) << current_cfl << ")" << std::endl;
            
            // Warn if CFL is getting large
            if (current_cfl > 1.0 && step % 50 == 0) {
                std::cout << "    WARNING: CFL = " << current_cfl << " > 1.0 at step " << step << std::endl;
            }
        }
        
        // SSP-RK3 time step
        auto step_start = std::chrono::high_resolution_clock::now();
        u_current = time_step_ssp_rk3(u_current, dt);
        auto step_end = std::chrono::high_resolution_clock::now();
        total_timestep_time += std::chrono::duration<double>(step_end - step_start).count();
        
        t += dt;
        step++;
        
        // Safety check for instabilities
        bool has_nan = false;
        for (const auto& elem_coeffs : u_current) {
            if (elem_coeffs.hasNaN()) {
                has_nan = true;
                break;
            }
        }
        
        if (has_nan) {
            std::cerr << "ERROR: Solution contains NaN values at t=" << t << std::endl;
            break;
        }
    }
    
    // Save final solution
    if (step % save_every != 0) {
        solution_frames.push_back(u_current);
    }
    
    std::cout << "\n=== Time Stepping Performance Summary ===" << std::endl;
    std::cout << "Euler solve completed with " << solution_frames.size() << " solution frames." << std::endl;
    std::cout << "  Total time steps: " << step << std::endl;
    std::cout << "  Total time stepping: " << std::fixed << std::setprecision(4) << total_timestep_time << " s" << std::endl;
    std::cout << "  Avg time per step: " << std::fixed << std::setprecision(6) << (total_timestep_time / step) << " s" << std::endl;
    std::cout << "  Maximum CFL encountered: " << std::fixed << std::setprecision(4) << max_cfl_encountered << std::endl;
    if (max_cfl_encountered > 1.0) {
        std::cout << "  WARNING: Maximum CFL exceeded 1.0 during simulation!" << std::endl;
    }
    
    // Convert to flat format for compatibility
    std::vector<Eigen::MatrixXd> flat_frames;
    for (const auto& frame : solution_frames) {
        // Concatenate all element coefficients into a single matrix
        int n_elem = frame.size();
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

double EulerDGSolver::compute_max_cfl(const std::vector<Eigen::MatrixXd>& u_coeffs, double dt) const {
    int n_elem = mesh_->get_n_elements();
    int n_basis = mesh_->get_dg_space()->get_basis()->get_n_basis();
    double max_cfl = 0.0;
    
    for (int elem_id = 0; elem_id < n_elem; ++elem_id) {
        // Get element vertices to compute characteristic size
        Eigen::MatrixXd vertices(mesh_->get_elements().cols(), 2);
        for (int i = 0; i < mesh_->get_elements().cols(); ++i) {
            vertices.row(i) = mesh_->get_vertices().row(mesh_->get_elements()(elem_id, i));
        }
        
        // Compute characteristic element size (minimum edge length or diameter)
        double h_elem = std::numeric_limits<double>::max();
        for (int i = 0; i < vertices.rows(); ++i) {
            for (int j = i + 1; j < vertices.rows(); ++j) {
                double dist = (vertices.row(i) - vertices.row(j)).norm();
                h_elem = std::min(h_elem, dist);
            }
        }
        
        // Compute maximum wave speed in this element
        double max_wave_speed = 0.0;
        for (int i = 0; i < n_basis; ++i) {
            // Extract conserved variables
            double rho = u_coeffs[elem_id](i, 0);
            double rho_u = u_coeffs[elem_id](i, 1);
            double rho_v = u_coeffs[elem_id](i, 2);
            double E = u_coeffs[elem_id](i, 3);
            
            // Compute primitive variables
            double u = rho_u / rho;
            double v = rho_v / rho;
            double p = (gamma_ - 1.0) * (E - 0.5 * rho * (u * u + v * v));
            
            // Speed of sound
            double c = std::sqrt(gamma_ * p / rho);
            
            // Maximum wave speed (velocity magnitude + speed of sound)
            double vel_mag = std::sqrt(u * u + v * v);
            double wave_speed = vel_mag + c;
            
            max_wave_speed = std::max(max_wave_speed, wave_speed);
        }
        
        // CFL number for this element
        double cfl_elem = dt * max_wave_speed / h_elem;
        max_cfl = std::max(max_cfl, cfl_elem);
    }
    
    return max_cfl;
}

NavierStokesDGSolver::NavierStokesDGSolver(std::shared_ptr<DGMesh> mesh,
                                           double gamma,
                                           double dynamic_viscosity,
                                           double prandtl,
                                           double penalty_prefactor)
    : EulerDGSolver(mesh, std::make_shared<NavierStokesWeakFormulation>(gamma, dynamic_viscosity, prandtl, penalty_prefactor)) {}

} // namespace dgfem