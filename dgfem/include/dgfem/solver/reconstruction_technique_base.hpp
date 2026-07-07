/**
 * @file reconstruction_technique_base.hpp
 * @brief Strategy interface for HOW a troubled cell's reconstructed slope gets replaced
 * (independent of WHICH cells are troubled -- that's TroubledCellIndicator's job).
 */

#pragma once

#include "dgfem/kokkos_math.hpp"
#include "dgfem/solver/neighbor_connectivity.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace dgfem {

class ReconstructionTechnique {
public:
    virtual ~ReconstructionTechnique() = default;
    [[nodiscard]] virtual std::string get_type() const = 0;

    // Elements with troubled_cells[e] == 0 pass through byte-identical to the input.
    [[nodiscard]] virtual std::vector<DView2>
    apply(const std::vector<DView2>& u, const std::vector<std::uint8_t>& troubled_cells,
         const NeighborConnectivity& neighbor_connectivity,
         const std::vector<double>& element_length, int n_vars, double gamma) const = 0;
};

}  // namespace dgfem
