/**
 * @file flux_eigensystem_base.hpp
 * @brief Strategy interface for a local (frozen-state) hyperbolic flux-Jacobian eigensystem.
 *
 * Used by the troubled-cell indicator / reconstruction technique hierarchy so those classes
 * never hardcode "1D Euler in x, exactly 3 coupled fields" the way the original
 * build_eigensystem_3() free function did -- a k-omega RANS state (more passively-advected
 * scalars riding on the same mean-flow triple) and a future 3D solver (eigensystem built from
 * a face normal instead of implicitly-x) both fit this interface without it changing.
 */

#pragma once

#include <string>
#include <vector>

namespace dgfem {

// A local (frozen-state) eigensystem of a hyperbolic flux Jacobian, restricted to the subset
// of conserved variables that are genuinely coupled through wave propagation in a given
// direction. n_coupled is carried explicitly rather than assumed -- every case in the current
// roadmap (x-direction Euler, k-omega's mean-flow block, a 3D face-normal Euler block) happens
// to have n_coupled == 3, but nothing here hardcodes that; a provider is free to report a
// different count.
struct LocalEigensystem {
    int n_coupled = 0;
    std::vector<std::vector<double>> R;  // n_coupled x n_coupled, R[conserved_local][field]
    std::vector<std::vector<double>> L;  // n_coupled x n_coupled, L[field][conserved_local]
};

// Which global state-vector indices participate in the coupled eigensystem (in the order
// consumed/produced by a provider's R/L), and which are merely passive scalars advected
// alongside it at the local characteristic speed. This is *configuration data*, not something
// a provider or its callers hardcode as literals: today's 4-variable Euler/NS state has
// coupled_indices={0,1,3} (rho, rho*u, E) and passive_indices={2} (rho*v); a future k-omega
// RANS state adds passive_indices={2,4,5} (rho*v, rho*k, rho*omega) -- same coupled math,
// strictly more passive entries, zero change to any caller of this struct.
struct CoupledFieldConfig {
    std::vector<int> coupled_indices;
    std::vector<int> passive_indices;
};

// direction/state vectors are std::vector<double>, not a fixed-size Vec2/Vec4 (this
// codebase's existing Kokkos::Array<double,2/4> aliases): a fixed-size type would itself be
// exactly the kind of "bakes in the dimensionality" assumption this abstraction exists to
// avoid (there is no Vec3 in kokkos_math.hpp, and there shouldn't need to be one just to
// satisfy this interface).
class FluxEigensystemProvider {
public:
    virtual ~FluxEigensystemProvider() = default;
    [[nodiscard]] virtual std::string get_type() const = 0;

    // direction: unit normal the eigensystem is built along (size 2 in today's 2D code, size
    // 3 for a future 3D face normal -- the provider decides what sizes it accepts).
    // cell_avg_primitives: this element's own cell-average primitive state, full length
    // (n_vars), so a provider can read whichever raw scalars its flux Jacobian needs (rho, u,
    // v, p, ...) regardless of how many passive fields are interleaved around them.
    [[nodiscard]] virtual LocalEigensystem
    build(const std::vector<double>& direction, const std::vector<double>& cell_avg_primitives,
         double gamma) const = 0;

    [[nodiscard]] virtual const CoupledFieldConfig& coupled_field_config() const = 0;
};

}  // namespace dgfem
