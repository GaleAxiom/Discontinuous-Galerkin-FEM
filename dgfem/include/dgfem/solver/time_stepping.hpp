/**
 * @file time_stepping.hpp
 * @brief Time-stepping schemes for temporal discretization
 *
 * This module provides common time-stepping algorithms used in DG solvers,
 * reducing code duplication across different solver types.
 */

#pragma once

#include <Eigen/Dense>

#include <functional>
#include <vector>

namespace dgfem {

/**
 * @brief Strong Stability Preserving Runge-Kutta schemes
 *
 * SSP-RK methods preserve stability properties (like positivity or TVD)
 * of forward Euler under a CFL condition.
 */
class SSP_RK {
public:
    /**
     * @brief Third-order SSP-RK3 scheme (SSP(3,3))
     *
     * Optimal SSP coefficient is 1. Time evolution:
     *   u^(1) = u^n + dt*L(u^n)
     *   u^(2) = 3/4*u^n + 1/4*u^(1) + 1/4*dt*L(u^(1))
     *   u^(n+1) = 1/3*u^n + 2/3*u^(2) + 2/3*dt*L(u^(2))
     *
     * @param u_n Current solution
     * @param dt Time step size
     * @param rhs_func Function computing du/dt = L(u)
     * @return Solution at next time step
     */
    template <typename SolutionType>
    static SolutionType step_rk3(const SolutionType& u_n, double dt,
                                 std::function<SolutionType(const SolutionType&)> rhs_func) {
        // Stage 1
        SolutionType k1 = rhs_func(u_n);
        SolutionType u1 = add_scaled(u_n, dt, k1);

        // Stage 2
        SolutionType k2 = rhs_func(u1);
        SolutionType u2 = add_scaled(scale(0.75, u_n), 0.25, add_scaled(u1, dt, k2));

        // Stage 3
        SolutionType k3 = rhs_func(u2);
        SolutionType u_np1 = add_scaled(scale(1.0 / 3.0, u_n), 2.0 / 3.0, add_scaled(u2, dt, k3));

        return u_np1;
    }

private:
    // Helper for scalar vectors: result = a + alpha*b
    static Eigen::VectorXd add_scaled(const Eigen::VectorXd& a, double alpha,
                                      const Eigen::VectorXd& b) {
        return a + alpha * b;
    }

    // Helper for scalar vectors: result = alpha*a
    static Eigen::VectorXd scale(double alpha, const Eigen::VectorXd& a) { return alpha * a; }

    // Helper for vector of matrices: result = a + alpha*b
    static std::vector<Eigen::MatrixXd> add_scaled(const std::vector<Eigen::MatrixXd>& a,
                                                   double alpha,
                                                   const std::vector<Eigen::MatrixXd>& b) {
        std::vector<Eigen::MatrixXd> result(a.size());
        for (size_t i = 0; i < a.size(); ++i) {
            result[i] = a[i] + alpha * b[i];
        }
        return result;
    }

    // Helper for vector of matrices: result = alpha*a
    static std::vector<Eigen::MatrixXd> scale(double alpha, const std::vector<Eigen::MatrixXd>& a) {
        std::vector<Eigen::MatrixXd> result(a.size());
        for (size_t i = 0; i < a.size(); ++i) {
            result[i] = alpha * a[i];
        }
        return result;
    }
};

/**
 * @brief Block-diagonal mass matrix operations
 *
 * DG discretizations have block-diagonal mass matrices (one block per element).
 * This class precomputes and applies the inverse efficiently.
 */
class BlockMassMatrix {
public:
    /**
     * @brief Compute and store inverse of mass matrix blocks
     * @param mass_blocks Mass matrix for each element
     */
    explicit BlockMassMatrix(const std::vector<Eigen::MatrixXd>& mass_blocks);

    /**
     * @brief Apply M^{-1} to a vector (scalar variable)
     * @param vec Input vector of size (n_elem * n_basis)
     * @return Result of M^{-1} * vec
     */
    Eigen::VectorXd apply_inverse(const Eigen::VectorXd& vec) const;

    /**
     * @brief Apply M^{-1} to element-wise matrices (system of variables)
     * @param vec Input (n_elem matrices of size n_basis x n_vars)
     * @return Result of M^{-1} applied to each element
     */
    std::vector<Eigen::MatrixXd> apply_inverse(const std::vector<Eigen::MatrixXd>& vec) const;

    /**
     * @brief Get number of elements
     */
    [[nodiscard]] int get_n_elements() const noexcept { return n_elem_; }

    /**
     * @brief Get number of basis functions per element
     */
    [[nodiscard]] int get_n_basis() const noexcept { return n_basis_; }

private:
    int n_elem_;
    int n_basis_;
    std::vector<Eigen::MatrixXd> M_inv_blocks_;
};

}  // namespace dgfem
