/**
 * @file neighbor_connectivity.hpp
 * @brief General, direction-classified element neighbor connectivity.
 *
 * Used by the troubled-cell indicator / reconstruction technique hierarchy so those base
 * interfaces don't bake in "exactly one direction, two neighbors" the way an x-only
 * left/right pair would -- a future y- or z-direction (or 3D hex, up to 6 faces) consumer
 * reads more of the same structure without any base-class signature changing.
 */

#pragma once

#include <memory>
#include <vector>

namespace dgfem {

class DGMesh;

// Which geometric direction a neighbor relationship sits in, as seen from a given element.
// Six values so a 3D hex element (6 faces) has a natural tag for each; a 2D quad element (4
// faces) only ever produces the first four. This is a *classified geometric direction*, not
// DGMesh's raw local face index -- face indices are mesh-topology-defined and, for a
// non-axis-aligned element, would not line up with any coordinate axis at all. Every element
// this codebase supports (order-1 quad now, order-1 hex later) is axis-aligned, so the
// classification (comparing neighbor/self centroid deltas per axis) is well-defined; a future
// curved/unstructured mesh would need a different scheme entirely (out of scope).
enum class FaceDirection { XMinus, XPlus, YMinus, YPlus, ZMinus, ZPlus };

// One neighbor relationship as seen from a given element.
struct ElementNeighbor {
    int neighbor_id;  // -1 if this face is a boundary (no interior neighbor)
    FaceDirection direction;
};

// Per-element list of classified neighbor relationships. connectivity[e] holds one entry per
// mesh face of element e, each carrying the geometric direction the classification step
// assigned it -- mirroring what DGMesh::get_element_neighbors already returns
// (std::vector<std::pair<int,int>> of (neighbor_elem, neighbor_face)) but with the raw local
// face id replaced by a direction a strategy class can actually reason about.
using NeighborConnectivity = std::vector<std::vector<ElementNeighbor>>;

// Convenience extraction: given one element's neighbor list and the two FaceDirection values
// bounding one axis (e.g. XMinus/XPlus), pull out the minus/plus neighbor ids. Any
// direction-scoped indicator/reconstruction (this pass: x-only; later: a y- or z-only
// sibling) uses this same helper with a different direction pair -- it is "the one-axis
// helper", not "the x helper", so adding a y-direction concrete class later requires no new
// plumbing here, only a new call site.
struct NeighborPair {
    int minus_id = -1;
    int plus_id = -1;
};

[[nodiscard]] inline NeighborPair
get_axis_neighbors(const std::vector<ElementNeighbor>& elem_neighbors,
                  FaceDirection minus_direction, FaceDirection plus_direction) {
    NeighborPair result;
    for (const auto& n : elem_neighbors) {
        if (n.direction == minus_direction) {
            result.minus_id = n.neighbor_id;
        } else if (n.direction == plus_direction) {
            result.plus_id = n.neighbor_id;
        }
    }
    return result;
}

// Result of classifying a mesh's neighbor connectivity: the per-element classified neighbor
// lists, plus each element's x-extent (this pass's only length scale, computed in the same
// centroid pass as the classification itself).
struct NeighborConnectivityResult {
    NeighborConnectivity connectivity;
    std::vector<double> element_dx;
};

// Classifies every element-neighbor relationship in `mesh` by comparing element centroid
// deltas along each coordinate axis (X then Y; Z is reserved for a future 3D mesh and never
// produced today), and computes each element's x-extent alongside it. Requires
// mesh->initialize_dg_space() to already have been called (uses get_element_neighbors()/
// get_element_vertices()), and an axis-aligned element mesh (quad now, hex later).
[[nodiscard]] NeighborConnectivityResult
build_neighbor_connectivity(const std::shared_ptr<DGMesh>& mesh);

}  // namespace dgfem
