/**
 * @file neighbor_connectivity_test.cpp
 * @brief Tests for the general (direction-classified) element neighbor connectivity used by
 * the troubled-cell indicator / reconstruction technique hierarchy.
 */

#include <dgfem/core/mesh.hpp>
#include <dgfem/core/space.hpp>
#include <dgfem/solver/neighbor_connectivity.hpp>
#include <dgfem/utils/mesh_creation.hpp>

#include <gmsh.h>
#include <gtest/gtest.h>

using namespace dgfem;
using namespace testing;

namespace {

class NeighborConnectivityTest : public Test {
protected:
    void SetUp() override {
        try {
            gmsh::clear();
            gmsh::finalize();
        } catch (...) {
        }
        gmsh::initialize();
        // 3x3 unit quads: domain [0,3]x[0,3], dx=1 -> 9 elements. The center cell (centroid
        // (1.5,1.5)) has a real interior neighbor on all 4 sides; a 2-row domain wouldn't give
        // any cell both a below AND an above neighbor.
        mesh_ = MeshCreator::create_rectangular_mesh(1.0, /*use_triangles=*/false, 0.0, 3.0, 0.0,
                                                     3.0);
        auto space = std::make_shared<DGSpace>("quad", 1);
        mesh_->initialize_dg_space(space, 4);
    }

    void TearDown() override {
        try {
            gmsh::clear();
            gmsh::finalize();
        } catch (...) {
        }
    }

    std::shared_ptr<DGMesh> mesh_;
};

// Finds the element whose centroid is closest to (x, y) -- avoids depending on GMSH's
// internal element numbering order.
int find_element_near(const std::shared_ptr<DGMesh>& mesh, double x, double y) {
    int best = -1;
    double best_dist2 = std::numeric_limits<double>::max();
    for (int e = 0; e < mesh->get_n_elements(); ++e) {
        DView2 verts = mesh->get_element_vertices(e);
        double cx = 0.0, cy = 0.0;
        int n_verts = static_cast<int>(verts.extent(0));
        for (int i = 0; i < n_verts; ++i) {
            cx += verts(i, 0);
            cy += verts(i, 1);
        }
        cx /= n_verts;
        cy /= n_verts;
        double dist2 = (cx - x) * (cx - x) + (cy - y) * (cy - y);
        if (dist2 < best_dist2) {
            best_dist2 = dist2;
            best = e;
        }
    }
    return best;
}

}  // namespace

TEST_F(NeighborConnectivityTest, InteriorElementHasAllFourNeighbors) {
    auto result = build_neighbor_connectivity(mesh_);
    int center = find_element_near(mesh_, 1.5, 1.5);  // the true center cell of the 3x3 grid
    ASSERT_GE(center, 0);

    NeighborPair x_pair = get_axis_neighbors(result.connectivity[center], FaceDirection::XMinus,
                                             FaceDirection::XPlus);
    NeighborPair y_pair = get_axis_neighbors(result.connectivity[center], FaceDirection::YMinus,
                                             FaceDirection::YPlus);

    int left = find_element_near(mesh_, 0.5, 1.5);
    int right = find_element_near(mesh_, 2.5, 1.5);
    int below = find_element_near(mesh_, 1.5, 0.5);
    int above = find_element_near(mesh_, 1.5, 2.5);

    EXPECT_EQ(x_pair.minus_id, left);
    EXPECT_EQ(x_pair.plus_id, right);
    EXPECT_EQ(y_pair.minus_id, below);
    EXPECT_EQ(y_pair.plus_id, above);
}

TEST_F(NeighborConnectivityTest, CornerElementHasNoLeftOrBottomNeighbor) {
    auto result = build_neighbor_connectivity(mesh_);
    int corner = find_element_near(mesh_, 0.5, 0.5);  // bottom-left cell
    ASSERT_GE(corner, 0);

    NeighborPair x_pair = get_axis_neighbors(result.connectivity[corner], FaceDirection::XMinus,
                                             FaceDirection::XPlus);
    NeighborPair y_pair = get_axis_neighbors(result.connectivity[corner], FaceDirection::YMinus,
                                             FaceDirection::YPlus);

    EXPECT_EQ(x_pair.minus_id, -1);  // no interior neighbor to the left: domain boundary
    EXPECT_GE(x_pair.plus_id, 0);    // has a neighbor to the right
    EXPECT_EQ(y_pair.minus_id, -1);  // no interior neighbor below: domain boundary
    EXPECT_GE(y_pair.plus_id, 0);    // has a neighbor above
}

TEST_F(NeighborConnectivityTest, ElementDxMatchesUniformGridSpacing) {
    auto result = build_neighbor_connectivity(mesh_);
    ASSERT_EQ(result.element_dx.size(), static_cast<size_t>(mesh_->get_n_elements()));
    for (double dx : result.element_dx) {
        EXPECT_NEAR(dx, 1.0, 1e-9);
    }
}

TEST_F(NeighborConnectivityTest, GetAxisNeighborsIgnoresOtherDirections) {
    // A hand-built neighbor list mixing all 6 FaceDirection values: get_axis_neighbors must
    // only read the two directions it's asked for, regardless of what else is present --
    // this is the property a future y- or z-direction consumer of the same structure relies
    // on without needing get_axis_neighbors itself to change.
    std::vector<ElementNeighbor> neighbors = {
        {10, FaceDirection::XMinus}, {11, FaceDirection::XPlus}, {20, FaceDirection::YMinus},
        {21, FaceDirection::YPlus},  {30, FaceDirection::ZMinus}, {31, FaceDirection::ZPlus}};

    NeighborPair x_pair = get_axis_neighbors(neighbors, FaceDirection::XMinus, FaceDirection::XPlus);
    NeighborPair y_pair = get_axis_neighbors(neighbors, FaceDirection::YMinus, FaceDirection::YPlus);
    NeighborPair z_pair = get_axis_neighbors(neighbors, FaceDirection::ZMinus, FaceDirection::ZPlus);

    EXPECT_EQ(x_pair.minus_id, 10);
    EXPECT_EQ(x_pair.plus_id, 11);
    EXPECT_EQ(y_pair.minus_id, 20);
    EXPECT_EQ(y_pair.plus_id, 21);
    EXPECT_EQ(z_pair.minus_id, 30);
    EXPECT_EQ(z_pair.plus_id, 31);
}

TEST_F(NeighborConnectivityTest, MissingDirectionYieldsSentinel) {
    std::vector<ElementNeighbor> neighbors = {{10, FaceDirection::XMinus}};
    NeighborPair x_pair = get_axis_neighbors(neighbors, FaceDirection::XMinus, FaceDirection::XPlus);
    EXPECT_EQ(x_pair.minus_id, 10);
    EXPECT_EQ(x_pair.plus_id, -1);
}
