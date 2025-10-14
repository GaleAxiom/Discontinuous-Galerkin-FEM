/**
 * @file dg_solver.hpp
 * @brief Main DG solver classes
 */

#pragma once

#include "dgfem/core/mesh.hpp"

#include <Eigen/Dense>

#include <functional>
#include <map>
#include <memory>
#include <vector>

#include "assembler.hpp"
#include "time_stepping.hpp"
#include "weak_form.hpp"

namespace dgfem {

/**
 * @brief Base DG solver class
 */
class DGSolverBase {
public:
    explicit DGSolverBase(std::shared_ptr<DGMesh> mesh);
    virtual ~DGSolverBase() = default;

    // Delete copy, default move
    DGSolverBase(const DGSolverBase&) = delete;
    DGSolverBase& operator=(const DGSolverBase&) = delete;
    DGSolverBase(DGSolverBase&&) noexcept = default;
    DGSolverBase& operator=(DGSolverBase&&) noexcept = default;

    [[nodiscard]] virtual const Eigen::SparseMatrix<double>& get_system_matrix() const = 0;

protected:
    std::shared_ptr<DGMesh> mesh_;
    std::shared_ptr<DGAssembler> assembler_;
};

/**
 * @brief Laplace equation DG solver
 */
class LaplaceDGSolver : public DGSolverBase {
public:
    LaplaceDGSolver(std::shared_ptr<DGMesh> mesh, double penalty_parameter = 10.0);

    // Delete copy, default move
    LaplaceDGSolver(const LaplaceDGSolver&) = delete;
    LaplaceDGSolver& operator=(const LaplaceDGSolver&) = delete;
    LaplaceDGSolver(LaplaceDGSolver&&) noexcept = default;
    LaplaceDGSolver& operator=(LaplaceDGSolver&&) noexcept = default;

    /**
     * @brief Solve Laplace equation with optional source term
     */
    [[nodiscard]] Eigen::VectorXd
    solve(std::function<double(const Eigen::Vector2d&)> source_func = nullptr);

    [[nodiscard]] const Eigen::SparseMatrix<double>& get_system_matrix() const override;

    /**
     * @brief Get assembled RHS vector
     */
    [[nodiscard]] const Eigen::VectorXd& get_rhs() const noexcept;

    /**
     * @brief Compute L2 and H1 errors against exact solution
     */
    [[nodiscard]] std::map<std::string, double> compute_error(
        std::function<double(const Eigen::Vector2d&)> exact_solution,
        std::function<Eigen::Vector2d(const Eigen::Vector2d&)> exact_gradient = nullptr) const;

private:
    std::shared_ptr<LaplaceWeakFormulation> weak_form_;
};

/**
 * @brief Advection equation DG solver
 */
class AdvectionDGSolver : public DGSolverBase {
public:
    AdvectionDGSolver(std::shared_ptr<DGMesh> mesh, const Eigen::Vector2d& advection_velocity);

    // Delete copy, default move
    AdvectionDGSolver(const AdvectionDGSolver&) = delete;
    AdvectionDGSolver& operator=(const AdvectionDGSolver&) = delete;
    AdvectionDGSolver(AdvectionDGSolver&&) noexcept = default;
    AdvectionDGSolver& operator=(AdvectionDGSolver&&) noexcept = default;

    /**
     * @brief Solve advection equation with time stepping
     */
    [[nodiscard]] std::vector<Eigen::VectorXd>
    solve(std::function<double(const Eigen::Vector2d&)> initial_condition, double T_final,
          double dt, std::shared_ptr<BoundaryCondition> boundary_condition = nullptr,
          int save_every = 1);

    [[nodiscard]] const Eigen::SparseMatrix<double>& get_system_matrix() const override;

private:
    std::shared_ptr<AdvectionWeakFormulation> weak_form_;
    Eigen::Vector2d advection_velocity_;

    // Precomputed operators for time stepping
    Eigen::SparseMatrix<double> L_operator_;     // Spatial operator
    Eigen::VectorXd F_boundary_;                 // Boundary forcing
    std::vector<Eigen::MatrixXd> M_inv_blocks_;  // Mass matrix inverse blocks per element

    /**
     * @brief Precompute mass matrix inverse blocks
     */
    void compute_mass_matrix_inverse_blocks();

    /**
     * @brief Apply inverse mass matrix M^{-1} * vec
     */
    [[nodiscard]] Eigen::VectorXd apply_mass_inv(const Eigen::VectorXd& vec) const;

    /**
     * @brief Project initial condition using L2 projection
     */
    [[nodiscard]] Eigen::VectorXd
    project_initial_condition(std::function<double(const Eigen::Vector2d&)> u0_func);

    /**
     * @brief Time stepping using SSP-RK3 (Strong Stability Preserving Runge-Kutta 3rd order)
     */
    [[nodiscard]] Eigen::VectorXd time_step_ssp_rk3(const Eigen::VectorXd& u_n, double dt) const;

    /**
     * @brief Compute RHS for time stepping: M^{-1} * (L*u + F_bc)
     */
    [[nodiscard]] Eigen::VectorXd compute_rhs(const Eigen::VectorXd& u) const;
};

/**
 * @brief Euler equations DG solver
 */
class EulerDGSolver : public DGSolverBase {
public:
    EulerDGSolver(std::shared_ptr<DGMesh> mesh, double gamma = 1.4);
    EulerDGSolver(std::shared_ptr<DGMesh> mesh, std::shared_ptr<EulerWeakFormulation> weak_form);

    // Delete copy, default move
    EulerDGSolver(const EulerDGSolver&) = delete;
    EulerDGSolver& operator=(const EulerDGSolver&) = delete;
    EulerDGSolver(EulerDGSolver&&) noexcept = default;
    EulerDGSolver& operator=(EulerDGSolver&&) noexcept = default;

    /**
     * @brief Solve Euler equations with time stepping
     */
    [[nodiscard]] std::vector<Eigen::MatrixXd>
    solve(std::function<Eigen::Vector4d(const Eigen::Vector2d&)> initial_condition, double T_final,
          double dt, int save_every = 1);

    [[nodiscard]] const Eigen::SparseMatrix<double>& get_system_matrix() const override;

private:
    std::shared_ptr<EulerWeakFormulation> weak_form_;
    std::shared_ptr<DGAssembler> assembler_;
    double gamma_;

    // Precomputed operators for time stepping
    std::vector<Eigen::MatrixXd> M_inv_blocks_;  // Mass matrix inverse blocks per element

    /**
     * @brief Compute mass matrix inverse blocks
     */
    void compute_mass_matrix_inverse_blocks();

    /**
     * @brief Apply inverse mass matrix M^{-1} * vec
     * @param vec Input vector of shape (n_elem, n_basis, n_vars)
     * @return Result of shape (n_elem, n_basis, n_vars)
     */
    [[nodiscard]] std::vector<Eigen::MatrixXd>
    apply_mass_inv(const std::vector<Eigen::MatrixXd>& vec) const;

    /**
     * @brief Project initial condition using L2 projection
     */
    [[nodiscard]] std::vector<Eigen::MatrixXd>
    project_initial_condition(std::function<Eigen::Vector4d(const Eigen::Vector2d&)> u0_func);

    /**
     * @brief Time stepping using SSP-RK3 (Strong Stability Preserving Runge-Kutta 3rd order)
     */
    [[nodiscard]] std::vector<Eigen::MatrixXd>
    time_step_ssp_rk3(const std::vector<Eigen::MatrixXd>& u_n, double dt) const;

    /**
     * @brief Assemble Euler residual for all elements
     */
    [[nodiscard]] std::vector<Eigen::MatrixXd>
    assemble_euler_residual(const std::vector<Eigen::MatrixXd>& u_coeffs) const;

    /**
     * @brief Compute maximum CFL number across all elements
     * @param u_coeffs Current solution coefficients
     * @param dt Time step size
     * @return Maximum CFL number
     */
    [[nodiscard]] double compute_max_cfl(const std::vector<Eigen::MatrixXd>& u_coeffs,
                                         double dt) const;
};

/**
 * @brief Navier-Stokes solver leveraging Euler infrastructure with viscous weak formulation
 */
class NavierStokesDGSolver : public EulerDGSolver {
public:
    NavierStokesDGSolver(std::shared_ptr<DGMesh> mesh, double gamma = 1.4,
                         double dynamic_viscosity = 1.0e-3, double prandtl = 0.72,
                         double penalty_prefactor = 5.0);
    ~NavierStokesDGSolver() override = default;
};

}  // namespace dgfem