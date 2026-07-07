/**
 * @file troubled_cell_indicator_base.hpp
 * @brief Strategy interface for deciding WHICH elements need their reconstructed slope
 * replaced (independent of HOW they get fixed -- that's ReconstructionTechnique's job).
 */

#pragma once

#include "dgfem/kokkos_math.hpp"
#include "dgfem/solver/neighbor_connectivity.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace dgfem {

class TroubledCellIndicator {
public:
    virtual ~TroubledCellIndicator() = default;
    [[nodiscard]] virtual std::string get_type() const = 0;

    // neighbor_connectivity: full per-element classified neighbor list -- a concrete
    // indicator reads out of it only the directions it cares about (this pass: XMinus/XPlus
    // via get_axis_neighbors()); a future y/z-capable indicator would read more of the same
    // structure without this signature changing at all.
    // element_length: characteristic element length along the direction the concrete
    // indicator actually limits (this pass: x-extent).
    // Returns one flag per element (u.size() long); boundary elements (missing an interior
    // neighbor on the relevant axis) are always 0/untroubled.
    [[nodiscard]] virtual std::vector<std::uint8_t>
    detect_troubled_cells(const std::vector<DView2>& u,
                         const NeighborConnectivity& neighbor_connectivity,
                         const std::vector<double>& element_length, int n_vars,
                         double gamma) const = 0;
};

// Trivial indicator: flags every element that has both x-neighbors present, regardless of
// smoothness -- reproduces the pre-refactor minmod limiter's coverage exactly (that code
// touched every such element unconditionally), used for regression pinning when paired with
// MinmodReconstruction.
class AlwaysTroubledIndicator : public TroubledCellIndicator {
public:
    [[nodiscard]] std::string get_type() const override { return "AlwaysTroubled"; }
    [[nodiscard]] std::vector<std::uint8_t>
    detect_troubled_cells(const std::vector<DView2>& u,
                         const NeighborConnectivity& neighbor_connectivity,
                         const std::vector<double>& element_length, int n_vars,
                         double gamma) const override;
};

}  // namespace dgfem
