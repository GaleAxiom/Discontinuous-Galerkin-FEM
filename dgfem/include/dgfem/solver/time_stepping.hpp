/**
 * @file time_stepping.hpp
 * @brief Time-stepping schemes for temporal discretization
 *
 * This module provides common time-stepping algorithms used in DG solvers,
 * reducing code duplication across different solver types.
 */

#pragma once

#include "dgfem/kokkos_math.hpp"
#include "dgfem/solver/trilinos_types.hpp"

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
    // Helper for a global Tpetra vector: result = a + alpha*b
    static Teuchos::RCP<TpetraMultiVector> add_scaled(const Teuchos::RCP<TpetraMultiVector>& a,
                                                      double alpha,
                                                      const Teuchos::RCP<TpetraMultiVector>& b) {
        auto result = Teuchos::rcp(new TpetraMultiVector(*a, Teuchos::Copy));
        result->update(alpha, *b, 1.0);
        return result;
    }

    // Helper for a global Tpetra vector: result = alpha*a
    static Teuchos::RCP<TpetraMultiVector> scale(double alpha,
                                                 const Teuchos::RCP<TpetraMultiVector>& a) {
        auto result = Teuchos::rcp(new TpetraMultiVector(*a, Teuchos::Copy));
        result->scale(alpha);
        return result;
    }

    // Helper for per-element local blocks (compressible StateVector): result = a + alpha*b
    static std::vector<DView2> add_scaled(const std::vector<DView2>& a, double alpha,
                                          const std::vector<DView2>& b) {
        std::vector<DView2> result(a.size());
        for (size_t e = 0; e < a.size(); ++e) {
            result[e] = DView2("rk_tmp", a[e].extent(0), a[e].extent(1));
            for (int i = 0; i < static_cast<int>(a[e].extent(0)); ++i) {
                for (int j = 0; j < static_cast<int>(a[e].extent(1)); ++j) {
                    result[e](i, j) = a[e](i, j) + alpha * b[e](i, j);
                }
            }
        }
        return result;
    }

    // Helper for per-element local blocks: result = alpha*a
    static std::vector<DView2> scale(double alpha, const std::vector<DView2>& a) {
        std::vector<DView2> result(a.size());
        for (size_t e = 0; e < a.size(); ++e) {
            result[e] = DView2("rk_tmp", a[e].extent(0), a[e].extent(1));
            for (int i = 0; i < static_cast<int>(a[e].extent(0)); ++i) {
                for (int j = 0; j < static_cast<int>(a[e].extent(1)); ++j) {
                    result[e](i, j) = alpha * a[e](i, j);
                }
            }
        }
        return result;
    }
};

}  // namespace dgfem
