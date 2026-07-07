/**
 * @file persson_peraire_indicator.hpp
 * @brief Modal-decay troubled-cell indicator (Persson & Peraire, 2006).
 *
 * Distinguishes "smooth but steep" from "genuinely discontinuous" by how fast a cell's own
 * modal coefficients decay -- a scale-invariant ratio, not a fixed magnitude threshold like
 * the TVB minmod limiter's kTvbM. This is exactly the fix for the diagnosed failure mode of
 * plain/TVB minmod: it cannot tell a genuine incoming discontinuity from the small-amplitude
 * dispersive "precursor" oscillation unlimited high-order DG produces ahead of a true
 * wavefront, because both simply exceed a fixed magnitude threshold. The Persson-Peraire
 * ratio only flags genuine non-decaying (discontinuous) modal content.
 */

#pragma once

#include "dgfem/solver/basis_mode_map.hpp"
#include "dgfem/solver/flux_eigensystem_base.hpp"
#include "dgfem/solver/troubled_cell_indicator_base.hpp"

#include <memory>

namespace dgfem {

class PerssonPeraireIndicator : public TroubledCellIndicator {
public:
    // eigensystem_provider: no parameterless default -- it must be sized to the solver's
    // actual n_vars, which isn't known at class-definition time (see the wiring note in
    // CompressibleDGSolverBase::set_limiter_enabled()).
    // kappa: the one tunable knob (analogous to the old kTvbM), calibrated empirically.
    explicit PerssonPeraireIndicator(
        std::shared_ptr<const FluxEigensystemProvider> eigensystem_provider,
        std::shared_ptr<const BasisModeMap> mode_map = std::make_shared<Order1QuadBasisModeMap>(),
        double kappa = 0.0);

    [[nodiscard]] std::string get_type() const override { return "PerssonPeraire"; }

    [[nodiscard]] std::vector<std::uint8_t>
    detect_troubled_cells(const std::vector<DView2>& u,
                         const NeighborConnectivity& neighbor_connectivity,
                         const std::vector<double>& element_length, int n_vars,
                         double gamma) const override;

private:
    std::shared_ptr<const FluxEigensystemProvider> eigensystem_provider_;
    std::shared_ptr<const BasisModeMap> mode_map_;
    double kappa_;
};

}  // namespace dgfem
