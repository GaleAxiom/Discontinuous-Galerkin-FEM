/**
 * @file weno_reconstruction.hpp
 * @brief WENO-style reconstruction technique (Qiu & Shu, 2005, reduced to order-1 DG).
 *
 * Targets the root cause diagnosed for plain/TVB minmod on the Sod shock tube: minmod's hard
 * "always shrink toward the smallest same-sign candidate" rule discards genuine large-slope
 * information near a dispersive precursor the same as near a real discontinuity. WENO's
 * smoothness-weighted blend instead keeps most of the own-cell slope unless it's measurably
 * rougher than its neighbors -- a fundamentally different, literature-preferred decision rule.
 */

#pragma once

#include "dgfem/solver/basis_mode_map.hpp"
#include "dgfem/solver/flux_eigensystem_base.hpp"
#include "dgfem/solver/reconstruction_technique_base.hpp"

#include <memory>

namespace dgfem {

class WenoReconstruction : public ReconstructionTechnique {
public:
    explicit WenoReconstruction(
        std::shared_ptr<const FluxEigensystemProvider> eigensystem_provider,
        std::shared_ptr<const BasisModeMap> mode_map = std::make_shared<Order1QuadBasisModeMap>(),
        double gamma_own = 0.998, double epsilon = 1e-6, double weno_power = 2.0);

    [[nodiscard]] std::string get_type() const override { return "Weno"; }

    [[nodiscard]] std::vector<DView2>
    apply(const std::vector<DView2>& u, const std::vector<std::uint8_t>& troubled_cells,
         const NeighborConnectivity& neighbor_connectivity,
         const std::vector<double>& element_length, int n_vars, double gamma) const override;

    // Pure WENO nonlinear-weight slope blend (own/left/right candidate slopes), independently
    // testable without any mesh/eigensystem/mode-map machinery. Linear weights are
    // (gamma_own, (1-gamma_own)/2, (1-gamma_own)/2); nonlinear weights
    // omega_i = linear_weight_i / (epsilon + dx^2*slope_i^2)^weno_power, normalized to sum 1.
    [[nodiscard]] static double blend_slopes(double s_own, double s_left, double s_right,
                                             double dx, double gamma_own, double epsilon,
                                             double weno_power);

private:
    std::shared_ptr<const FluxEigensystemProvider> eigensystem_provider_;
    std::shared_ptr<const BasisModeMap> mode_map_;
    double gamma_own_;
    double epsilon_;
    double weno_power_;
};

}  // namespace dgfem
