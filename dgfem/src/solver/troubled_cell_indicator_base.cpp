/**
 * @file troubled_cell_indicator_base.cpp
 * @brief Implementation of AlwaysTroubledIndicator.
 */

#include "dgfem/solver/troubled_cell_indicator_base.hpp"

namespace dgfem {

std::vector<std::uint8_t>
AlwaysTroubledIndicator::detect_troubled_cells(const std::vector<DView2>& u,
                                               const NeighborConnectivity& neighbor_connectivity,
                                               const std::vector<double>& element_length,
                                               int n_vars, double gamma) const {
    (void)element_length;
    (void)n_vars;
    (void)gamma;
    int n_elem = static_cast<int>(u.size());
    std::vector<std::uint8_t> flags(n_elem, 0);
    for (int e = 0; e < n_elem; ++e) {
        NeighborPair pair = get_axis_neighbors(neighbor_connectivity[e], FaceDirection::XMinus,
                                               FaceDirection::XPlus);
        if (pair.minus_id >= 0 && pair.plus_id >= 0) {
            flags[e] = 1;
        }
    }
    return flags;
}

}  // namespace dgfem
