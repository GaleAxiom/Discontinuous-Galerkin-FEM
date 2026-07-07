/**
 * @file euler_eigensystem.hpp
 * @brief Concrete x-direction 1D Euler flux-Jacobian eigensystem provider.
 */

#pragma once

#include "dgfem/kokkos_math.hpp"
#include "dgfem/solver/flux_eigensystem_base.hpp"

namespace dgfem {

// Right/left eigenvector matrices of the 1D Euler flux Jacobian (x-direction), acting on the
// 3-field subsystem (rho, rho*u, E) -- other state indices ride along as passive scalars (see
// coupled_field_config()). Standard closed form, e.g. Toro, "Riemann Solvers and Numerical
// Methods for Fluid Dynamics", eq. 3.79-3.82. Verified numerically, not just algebraically,
// before use: max|L*R-I| ~1e-16 for representative states (see euler_eigensystem_test.cpp).
class EulerXDirectionEigensystem : public FluxEigensystemProvider {
public:
    // n_vars: total state-vector width (4 for plain Euler/NS). passive_indices is derived as
    // every index in [0, n_vars) other than {0, 1, 3} -- for n_vars=4 that's exactly {2}
    // (rho*v); a future k-omega solver constructing this with n_vars=6 gets
    // passive_indices={2,4,5} for free, with no code change here.
    explicit EulerXDirectionEigensystem(int n_vars);

    [[nodiscard]] std::string get_type() const override { return "EulerXDirection"; }

    // Only supports direction == {1.0, 0.0} this pass (throws std::invalid_argument
    // otherwise) -- concrete callers this pass (PerssonPeraireIndicator, MinmodReconstruction,
    // WenoReconstruction) always pass exactly that, so this validates the contract rather than
    // silently producing wrong physics if some future caller passes something else before a
    // genuine multi-direction provider exists.
    [[nodiscard]] LocalEigensystem
    build(const std::vector<double>& direction, const std::vector<double>& cell_avg_primitives,
         double gamma) const override;

    [[nodiscard]] const CoupledFieldConfig& coupled_field_config() const override {
        return config_;
    }

private:
    CoupledFieldConfig config_;
};

// Builds the full n_vars-width primitive-state vector EulerXDirectionEigensystem's coupled
// triple (indices 0,1,3 = rho, rho*u, E) needs conserved_to_primitive for, with any additional
// (passive) index simply divided by density -- a generic "primitive concentration" harmless
// for any index this pass never actually populates beyond n_vars==4's single passive rho*v
// (handled directly as a raw coefficient by callers, never read through this vector).
// Shared by PerssonPeraireIndicator, MinmodReconstruction, and WenoReconstruction so this
// Euler-specific conversion exists in exactly one place.
[[nodiscard]] std::vector<double> cell_average_primitives(const DView2& u_elem, int n_vars,
                                                          double gamma);

}  // namespace dgfem
