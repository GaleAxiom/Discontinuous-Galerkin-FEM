/**
 * @file minmod_reconstruction.hpp
 * @brief Characteristic-variable TVB minmod reconstruction technique.
 *
 * Migrated from the original apply_minmod_limiter_x() (now removed from
 * CompressibleDGSolverBase): the coupled Euler triple is limited in characteristic space via
 * an injected FluxEigensystemProvider, and any other (passive) state index is limited
 * componentwise. Kept as a technique in its own right (rather than deleted outright) so the
 * Sod shock tube e2e test can pin the pre-refactor behavior via
 * {AlwaysTroubledIndicator, MinmodReconstruction} and compare it against the new default.
 */

#pragma once

#include "dgfem/solver/basis_mode_map.hpp"
#include "dgfem/solver/flux_eigensystem_base.hpp"
#include "dgfem/solver/reconstruction_technique_base.hpp"

#include <memory>

namespace dgfem {

class MinmodReconstruction : public ReconstructionTechnique {
public:
    explicit MinmodReconstruction(
        std::shared_ptr<const FluxEigensystemProvider> eigensystem_provider,
        std::shared_ptr<const BasisModeMap> mode_map = std::make_shared<Order1QuadBasisModeMap>(),
        double tvb_m = 50.0);

    [[nodiscard]] std::string get_type() const override { return "Minmod"; }

    [[nodiscard]] std::vector<DView2>
    apply(const std::vector<DView2>& u, const std::vector<std::uint8_t>& troubled_cells,
         const NeighborConnectivity& neighbor_connectivity,
         const std::vector<double>& element_length, int n_vars, double gamma) const override;

private:
    std::shared_ptr<const FluxEigensystemProvider> eigensystem_provider_;
    std::shared_ptr<const BasisModeMap> mode_map_;
    double tvb_m_;
};

}  // namespace dgfem
