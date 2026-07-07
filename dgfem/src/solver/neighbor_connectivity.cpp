/**
 * @file neighbor_connectivity.cpp
 * @brief Implementation of neighbor-connectivity classification.
 */

#include "dgfem/solver/neighbor_connectivity.hpp"

#include "dgfem/core/mesh.hpp"
#include "dgfem/kokkos_math.hpp"

#include <algorithm>
#include <limits>

namespace dgfem {

NeighborConnectivityResult build_neighbor_connectivity(const std::shared_ptr<DGMesh>& mesh) {
    int n_elem = mesh->get_n_elements();

    NeighborConnectivityResult result;
    result.connectivity.assign(n_elem, {});
    result.element_dx.assign(n_elem, 0.0);

    std::vector<Vec2> centroids(n_elem);
    for (int e = 0; e < n_elem; ++e) {
        DView2 verts = mesh->get_element_vertices(e);
        int n_verts = static_cast<int>(verts.extent(0));
        double sx = 0.0, sy = 0.0;
        double x_min = std::numeric_limits<double>::max();
        double x_max = std::numeric_limits<double>::lowest();
        for (int i = 0; i < n_verts; ++i) {
            sx += verts(i, 0);
            sy += verts(i, 1);
            x_min = std::min(x_min, verts(i, 0));
            x_max = std::max(x_max, verts(i, 0));
        }
        centroids[e] = Vec2{sx / n_verts, sy / n_verts};
        result.element_dx[e] = x_max - x_min;
    }

    for (int e = 0; e < n_elem; ++e) {
        for (const auto& [nbr, nbr_face] : mesh->get_element_neighbors(e)) {
            if (nbr < 0 || nbr == e) {
                continue;
            }
            double dx = centroids[nbr][0] - centroids[e][0];
            double dy = centroids[nbr][1] - centroids[e][1];
            if (std::abs(dx) > std::abs(dy)) {
                result.connectivity[e].push_back(
                    {nbr, dx < 0.0 ? FaceDirection::XMinus : FaceDirection::XPlus});
            } else {
                result.connectivity[e].push_back(
                    {nbr, dy < 0.0 ? FaceDirection::YMinus : FaceDirection::YPlus});
            }
        }
    }

    return result;
}

}  // namespace dgfem
