/**
 * @file dg_solver.hpp
 * @brief Main DG solver classes
 */

#pragma once

#include "dgfem/core/mesh.hpp"

#include <cstdint>
#include <filesystem>

#include <array>
#include <chrono>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "assembler.hpp"
#include "trilinos_types.hpp"
#include "weak_form.hpp"

namespace dgfem {

/**
 * @brief Base DG solver class
 */
class DGSolverBase {
public:
    enum class Stage : std::uint8_t {
        Setup = 0,
        Assembly,
        Solve,
        Projection,
        TimeStep,
        Output,
        Postprocess,
        Count
    };

    struct StageStats {
        double total_time{0.0};
        int count{0};
    };

    struct SolverTelemetry {
        std::array<StageStats, static_cast<std::size_t>(Stage::Count)> stages{};
        double wall_time{0.0};
        int steps_completed{0};
        int rhs_evaluations{0};
        int output_frames{0};
    };

    using StageObserver = std::function<void(Stage, const SolverTelemetry&)>;

    explicit DGSolverBase(std::shared_ptr<DGMesh> mesh, std::string solver_name = "DG Solver");
    virtual ~DGSolverBase() = default;

    // Delete copy, default move
    DGSolverBase(const DGSolverBase&) = delete;
    DGSolverBase& operator=(const DGSolverBase&) = delete;
    DGSolverBase(DGSolverBase&&) noexcept = default;
    DGSolverBase& operator=(DGSolverBase&&) noexcept = default;

    void set_solver_name(std::string name);
    [[nodiscard]] const std::string& solver_name() const noexcept;

    void set_verbose(bool verbose) noexcept;
    [[nodiscard]] bool verbose() const noexcept;

    void set_output_directory(std::filesystem::path directory);
    [[nodiscard]] const std::filesystem::path& output_directory() const noexcept;

    void add_stage_observer(StageObserver observer);

    void reset_telemetry();
    [[nodiscard]] const SolverTelemetry& telemetry() const noexcept;

    [[nodiscard]] std::shared_ptr<DGMesh> mesh() const noexcept { return mesh_; }
    [[nodiscard]] std::shared_ptr<DGSpace> space() const;
    [[nodiscard]] int total_dofs() const;

    [[nodiscard]] virtual Teuchos::RCP<const TpetraCrsMatrix> get_system_matrix() const = 0;

protected:
    void begin_stage(Stage stage) const;
    void end_stage(Stage stage) const;

    void increment_steps(int steps = 1) noexcept;
    void increment_rhs_evaluations(int rhs = 1) const noexcept;
    void increment_output_frames(int frames = 1) noexcept;

    void log(Stage stage, const std::string& message) const;
    void log(const std::string& message) const;

    [[nodiscard]] std::filesystem::path make_output_path(const std::string& stem,
                                                         const std::string& extension = "") const;

    static std::string stage_to_string(Stage stage);

    std::shared_ptr<DGMesh> mesh_;
    std::shared_ptr<DGAssembler> assembler_;

private:
    using Clock = std::chrono::steady_clock;

    void notify_observers(Stage stage) const;

    std::string solver_name_;
    std::filesystem::path output_directory_;
    bool verbose_{true};

    mutable SolverTelemetry telemetry_;
    mutable std::array<bool, static_cast<std::size_t>(Stage::Count)> active_stage_flags_{};
    mutable std::array<Clock::time_point, static_cast<std::size_t>(Stage::Count)>
        stage_start_time_{};
    std::vector<StageObserver> observers_;
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
    [[nodiscard]] DView1 solve(std::function<double(const Vec2&)> source_func = nullptr);

    [[nodiscard]] Teuchos::RCP<const TpetraCrsMatrix> get_system_matrix() const override;

    /**
     * @brief Get assembled RHS vector
     */
    [[nodiscard]] Teuchos::RCP<const TpetraMultiVector> get_rhs() const noexcept;

    /**
     * @brief Compute L2 and H1 errors against exact solution
     */
    [[nodiscard]] std::map<std::string, double>
    compute_error(std::function<double(const Vec2&)> exact_solution,
                  std::function<Vec2(const Vec2&)> exact_gradient = nullptr) const;

private:
    std::shared_ptr<LaplaceWeakFormulation> weak_form_;
};

/**
 * @brief Advection equation DG solver
 */
class AdvectionDGSolver : public DGSolverBase {
public:
    AdvectionDGSolver(std::shared_ptr<DGMesh> mesh, const Vec2& advection_velocity);

    // Delete copy, default move
    AdvectionDGSolver(const AdvectionDGSolver&) = delete;
    AdvectionDGSolver& operator=(const AdvectionDGSolver&) = delete;
    AdvectionDGSolver(AdvectionDGSolver&&) noexcept = default;
    AdvectionDGSolver& operator=(AdvectionDGSolver&&) noexcept = default;

    /**
     * @brief Solve advection equation with time stepping
     */
    [[nodiscard]] std::vector<Teuchos::RCP<TpetraMultiVector>>
    solve(std::function<double(const Vec2&)> initial_condition, double T_final, double dt,
          std::shared_ptr<BoundaryCondition> boundary_condition = nullptr, int save_every = 1);

    [[nodiscard]] Teuchos::RCP<const TpetraCrsMatrix> get_system_matrix() const override;

private:
    std::shared_ptr<AdvectionWeakFormulation> weak_form_;
    Vec2 advection_velocity_;

    // Precomputed operators for time stepping
    Teuchos::RCP<const TpetraCrsMatrix> L_operator_;    // Spatial operator
    Teuchos::RCP<const TpetraMultiVector> F_boundary_;  // Boundary forcing
    std::vector<DView2> M_inv_blocks_;                  // Mass matrix inverse blocks per element

    /**
     * @brief Precompute mass matrix inverse blocks
     */
    void compute_mass_matrix_inverse_blocks();

    /**
     * @brief Apply inverse mass matrix M^{-1} * vec (global vector, one Tpetra apply per block)
     */
    [[nodiscard]] Teuchos::RCP<TpetraMultiVector>
    apply_mass_inv(const Teuchos::RCP<const TpetraMultiVector>& vec) const;

    /**
     * @brief Project initial condition using L2 projection
     */
    [[nodiscard]] Teuchos::RCP<TpetraMultiVector>
    project_initial_condition(std::function<double(const Vec2&)> u0_func);

    /**
     * @brief Time stepping using SSP-RK3 (Strong Stability Preserving Runge-Kutta 3rd order)
     */
    [[nodiscard]] Teuchos::RCP<TpetraMultiVector>
    time_step_ssp_rk3(const Teuchos::RCP<TpetraMultiVector>& u_n, double dt) const;

    /**
     * @brief Compute RHS for time stepping: M^{-1} * (L*u + F_bc)
     */
    [[nodiscard]] Teuchos::RCP<TpetraMultiVector>
    compute_rhs(const Teuchos::RCP<TpetraMultiVector>& u) const;

    /**
     * @brief CFL number for the given dt: |advection_velocity_| * dt / h_min. Unlike the
     * compressible solvers, the advection velocity is spatially and temporally constant, so
     * this only needs the mesh's minimum element size, computed once, rather than a per-step
     * quadrature-point loop.
     */
    [[nodiscard]] double compute_cfl(double dt) const;
};

/**
 * @brief Base class for compressible flow solvers (Euler, Navier-Stokes)
 */
class CompressibleDGSolverBase : public DGSolverBase {
public:
    using StateVector = std::vector<DView2>;

    [[nodiscard]] Teuchos::RCP<const TpetraCrsMatrix> get_system_matrix() const override;

    /**
     * @brief True if the most recent solve() detected a non-finite (NaN/Inf) state and
     * aborted early. The returned frame history is then a partial run, not a completed one --
     * callers that care about correctness (as opposed to just inspecting the frames up to
     * failure) should check this rather than infer failure from frame/step counts.
     */
    [[nodiscard]] bool has_diverged() const noexcept { return diverged_; }

protected:
    CompressibleDGSolverBase(std::shared_ptr<DGMesh> mesh,
                             std::shared_ptr<EulerWeakFormulation> weak_form,
                             std::string solver_label);

    [[nodiscard]] StateVector assemble_residual(const StateVector& u_coeffs) const;
    void compute_mass_matrix_inverse_blocks();
    [[nodiscard]] StateVector apply_mass_inv(const StateVector& vec) const;
    [[nodiscard]] StateVector project_initial_condition(std::function<Vec4(const Vec2&)> u0_func);
    [[nodiscard]] StateVector time_step_ssp_rk3(const StateVector& u_n, double dt) const;
    [[nodiscard]] double compute_max_cfl(const StateVector& u_coeffs, double dt) const;
    [[nodiscard]] std::vector<DView2>
    run_time_integration(std::function<Vec4(const Vec2&)> initial_condition, double T_final,
                         double dt, int save_every);
    [[nodiscard]] std::pair<double, double>
    compute_density_range(const StateVector& u_coeffs) const;

    std::shared_ptr<EulerWeakFormulation> weak_form_;
    double gamma_;

private:
    std::vector<DView2> M_inv_blocks_;
    std::string solver_label_;
    mutable StateVector residual_buffer_;
    bool diverged_{false};

    [[nodiscard]] std::vector<DView2> flatten_frames(const std::vector<StateVector>& frames) const;
};

/**
 * @brief Euler equations DG solver
 */
class EulerDGSolver : public CompressibleDGSolverBase {
public:
    EulerDGSolver(std::shared_ptr<DGMesh> mesh, double gamma = 1.4);
    EulerDGSolver(std::shared_ptr<DGMesh> mesh, std::shared_ptr<EulerWeakFormulation> weak_form);

    // Delete copy, default move
    EulerDGSolver(const EulerDGSolver&) = delete;
    EulerDGSolver& operator=(const EulerDGSolver&) = delete;
    EulerDGSolver(EulerDGSolver&&) noexcept = default;
    EulerDGSolver& operator=(EulerDGSolver&&) noexcept = default;

    [[nodiscard]] std::vector<DView2> solve(std::function<Vec4(const Vec2&)> initial_condition,
                                            double T_final, double dt, int save_every = 1);
};

/**
 * @brief Navier-Stokes solver with independent implementation
 */
class NavierStokesDGSolver : public CompressibleDGSolverBase {
public:
    NavierStokesDGSolver(std::shared_ptr<DGMesh> mesh, double gamma = 1.4,
                         double dynamic_viscosity = 1.0e-3, double prandtl = 0.72,
                         double penalty_prefactor = 5.0);

    // Delete copy, default move
    NavierStokesDGSolver(const NavierStokesDGSolver&) = delete;
    NavierStokesDGSolver& operator=(const NavierStokesDGSolver&) = delete;
    NavierStokesDGSolver(NavierStokesDGSolver&&) noexcept = default;
    NavierStokesDGSolver& operator=(NavierStokesDGSolver&&) noexcept = default;

    [[nodiscard]] std::vector<DView2> solve(std::function<Vec4(const Vec2&)> initial_condition,
                                            double T_final, double dt, int save_every = 1);
};

}  // namespace dgfem