#include <dgfem/kokkos_math.hpp>
#include <dgfem/reference/elements.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace dgfem;
using namespace testing;

namespace {
double sum(const DView1& v) {
    double s = 0.0;
    for (int i = 0; i < static_cast<int>(v.extent(0)); ++i)
        s += v(i);
    return s;
}
double col_sum(const DView2& m, int col) {
    double s = 0.0;
    for (int i = 0; i < static_cast<int>(m.extent(0)); ++i)
        s += m(i, col);
    return s;
}
}  // namespace

TEST(ReferenceTriangleTest, VertexCount) {
    ReferenceTriangle tri;
    EXPECT_EQ(tri.get_n_vertices(), 3);
    EXPECT_EQ(tri.get_n_edges(), 3);
    EXPECT_EQ(tri.get_dimension(), 2);
}

TEST(ReferenceTriangleTest, VertexLocations) {
    ReferenceTriangle tri;
    auto vertices = tri.get_vertices();

    // Check vertices are at correct locations
    EXPECT_NEAR(vertices(0, 0), 0.0, 1e-12);  // v0.x
    EXPECT_NEAR(vertices(0, 1), 0.0, 1e-12);  // v0.y
    EXPECT_NEAR(vertices(1, 0), 1.0, 1e-12);  // v1.x
    EXPECT_NEAR(vertices(1, 1), 0.0, 1e-12);  // v1.y
    EXPECT_NEAR(vertices(2, 0), 0.0, 1e-12);  // v2.x
    EXPECT_NEAR(vertices(2, 1), 1.0, 1e-12);  // v2.y
}

TEST(ReferenceTriangleTest, ContainsPoint) {
    ReferenceTriangle tri;

    // Test points inside
    EXPECT_TRUE(tri.contains_point(Vec2{0.0, 0.0}));    // vertex
    EXPECT_TRUE(tri.contains_point(Vec2{1.0, 0.0}));    // vertex
    EXPECT_TRUE(tri.contains_point(Vec2{0.0, 1.0}));    // vertex
    EXPECT_TRUE(tri.contains_point(Vec2{0.25, 0.25}));  // interior

    // Test points outside
    EXPECT_FALSE(tri.contains_point(Vec2{-0.1, 0.0}));
    EXPECT_FALSE(tri.contains_point(Vec2{0.0, -0.1}));
    EXPECT_FALSE(tri.contains_point(Vec2{0.5, 0.6}));  // sum > 1
}

TEST(ReferenceTriangleTest, EdgeVertices) {
    ReferenceTriangle tri;

    // Check each edge has correct vertices
    auto edge0 = tri.edge_vertices(0);
    EXPECT_EQ(edge0.first, 0);
    EXPECT_EQ(edge0.second, 1);

    auto edge1 = tri.edge_vertices(1);
    EXPECT_EQ(edge1.first, 1);
    EXPECT_EQ(edge1.second, 2);

    auto edge2 = tri.edge_vertices(2);
    EXPECT_EQ(edge2.first, 2);
    EXPECT_EQ(edge2.second, 0);
}

// Test shape functions for triangles
TEST(ReferenceTriangleTest, ShapeFunctions) {
    ReferenceTriangle tri;

    // Test at vertices - nodal property (shape function i = 1 at vertex i, 0 elsewhere)
    std::vector<Vec2> vertices = {
        Vec2{0.0, 0.0},  // vertex 0
        Vec2{1.0, 0.0},  // vertex 1
        Vec2{0.0, 1.0}   // vertex 2
    };

    for (size_t v = 0; v < vertices.size(); ++v) {
        DView1 N;
        DView2 dN_dxi;
        tri.compute_shape_functions(vertices[v], N, dN_dxi);

        EXPECT_EQ(N.size(), 3) << "At vertex " << v;

        // Check nodal property
        for (int i = 0; i < 3; ++i) {
            if (i == static_cast<int>(v)) {
                EXPECT_NEAR(N(i), 1.0, 1e-12) << "Shape function " << i << " at vertex " << v;
            } else {
                EXPECT_NEAR(N(i), 0.0, 1e-12) << "Shape function " << i << " at vertex " << v;
            }
        }
    }

    // Test partition of unity: sum of shape functions = 1
    Vec2 test_point{0.3, 0.4};
    DView1 N;
    DView2 dN_dxi;
    tri.compute_shape_functions(test_point, N, dN_dxi);

    EXPECT_NEAR(sum(N), 1.0, 1e-12) << "Partition of unity failed at interior point";
}

// Test shape function gradients for triangles
TEST(ReferenceTriangleTest, ShapeFunctionGradients) {
    ReferenceTriangle tri;

    Vec2 test_point{0.25, 0.35};
    DView1 N;
    DView2 dN_dxi;
    tri.compute_shape_functions(test_point, N, dN_dxi);

    EXPECT_EQ(dN_dxi.extent(0), 3);
    EXPECT_EQ(dN_dxi.extent(1), 2);

    // For linear triangle: N = [1-x-y, x, y]
    // Gradients should be constant
    EXPECT_NEAR(dN_dxi(0, 0), -1.0, 1e-12);  // dN0/dx
    EXPECT_NEAR(dN_dxi(0, 1), -1.0, 1e-12);  // dN0/dy
    EXPECT_NEAR(dN_dxi(1, 0), 1.0, 1e-12);   // dN1/dx
    EXPECT_NEAR(dN_dxi(1, 1), 0.0, 1e-12);   // dN1/dy
    EXPECT_NEAR(dN_dxi(2, 0), 0.0, 1e-12);   // dN2/dx
    EXPECT_NEAR(dN_dxi(2, 1), 1.0, 1e-12);   // dN2/dy

    // Verify gradients sum to zero (constant field has zero gradient)
    EXPECT_NEAR(col_sum(dN_dxi, 0), 0.0, 1e-12);
    EXPECT_NEAR(col_sum(dN_dxi, 1), 0.0, 1e-12);
}

// Test project_to_bounds for triangles
TEST(ReferenceTriangleTest, ProjectToBounds) {
    ReferenceTriangle tri;

    // Point inside - should not change
    Vec2 inside{0.3, 0.4};
    Vec2 inside_copy = inside;
    tri.project_to_bounds(inside);
    EXPECT_NEAR(norm(inside - inside_copy), 0.0, 1e-12);

    // Point with x < 0
    Vec2 neg_x{-0.1, 0.5};
    tri.project_to_bounds(neg_x);
    EXPECT_GE(neg_x[0], 0.0);
    EXPECT_LE(neg_x[0] + neg_x[1], 1.0);

    // Point with y < 0
    Vec2 neg_y{0.5, -0.1};
    tri.project_to_bounds(neg_y);
    EXPECT_GE(neg_y[1], 0.0);
    EXPECT_LE(neg_y[0] + neg_y[1], 1.0);

    // Point with x+y > 1
    Vec2 outside{0.7, 0.7};
    tri.project_to_bounds(outside);
    EXPECT_LE(outside[0] + outside[1], 1.0 + 1e-10);  // Small tolerance for rounding
}

TEST(ReferenceQuadTest, VertexCount) {
    ReferenceQuad quad;
    EXPECT_EQ(quad.get_n_vertices(), 4);
    EXPECT_EQ(quad.get_n_edges(), 4);
    EXPECT_EQ(quad.get_dimension(), 2);
}

TEST(ReferenceQuadTest, VertexLocations) {
    ReferenceQuad quad;
    auto vertices = quad.get_vertices();

    // Check vertices are at correct locations
    EXPECT_NEAR(vertices(0, 0), -1.0, 1e-12);  // v0.x
    EXPECT_NEAR(vertices(0, 1), -1.0, 1e-12);  // v0.y
    EXPECT_NEAR(vertices(1, 0), 1.0, 1e-12);   // v1.x
    EXPECT_NEAR(vertices(1, 1), -1.0, 1e-12);  // v1.y
    EXPECT_NEAR(vertices(2, 0), 1.0, 1e-12);   // v2.x
    EXPECT_NEAR(vertices(2, 1), 1.0, 1e-12);   // v2.y
    EXPECT_NEAR(vertices(3, 0), -1.0, 1e-12);  // v3.x
    EXPECT_NEAR(vertices(3, 1), 1.0, 1e-12);   // v3.y
}

TEST(ReferenceQuadTest, ContainsPoint) {
    ReferenceQuad quad;

    // Test points inside
    EXPECT_TRUE(quad.contains_point(Vec2{-1.0, -1.0}));  // vertex
    EXPECT_TRUE(quad.contains_point(Vec2{1.0, -1.0}));   // vertex
    EXPECT_TRUE(quad.contains_point(Vec2{1.0, 1.0}));    // vertex
    EXPECT_TRUE(quad.contains_point(Vec2{-1.0, 1.0}));   // vertex
    EXPECT_TRUE(quad.contains_point(Vec2{0.0, 0.0}));    // center

    // Test points outside
    EXPECT_FALSE(quad.contains_point(Vec2{-1.1, 0.0}));
    EXPECT_FALSE(quad.contains_point(Vec2{0.0, 1.1}));
}

TEST(ReferenceQuadTest, EdgeVertices) {
    ReferenceQuad quad;

    // Check each edge has correct vertices
    auto edge0 = quad.edge_vertices(0);
    EXPECT_EQ(edge0.first, 0);
    EXPECT_EQ(edge0.second, 1);

    auto edge1 = quad.edge_vertices(1);
    EXPECT_EQ(edge1.first, 1);
    EXPECT_EQ(edge1.second, 2);

    auto edge2 = quad.edge_vertices(2);
    EXPECT_EQ(edge2.first, 2);
    EXPECT_EQ(edge2.second, 3);

    auto edge3 = quad.edge_vertices(3);
    EXPECT_EQ(edge3.first, 3);
    EXPECT_EQ(edge3.second, 0);
}

// Test shape functions for quads
TEST(ReferenceQuadTest, ShapeFunctions) {
    ReferenceQuad quad;

    // Test at vertices - nodal property
    std::vector<Vec2> vertices = {
        Vec2{-1.0, -1.0},  // vertex 0
        Vec2{1.0, -1.0},   // vertex 1
        Vec2{1.0, 1.0},    // vertex 2
        Vec2{-1.0, 1.0}    // vertex 3
    };

    for (size_t v = 0; v < vertices.size(); ++v) {
        DView1 N;
        DView2 dN_dxi;
        quad.compute_shape_functions(vertices[v], N, dN_dxi);

        EXPECT_EQ(N.size(), 4) << "At vertex " << v;

        // Check nodal property
        for (int i = 0; i < 4; ++i) {
            if (i == static_cast<int>(v)) {
                EXPECT_NEAR(N(i), 1.0, 1e-12) << "Shape function " << i << " at vertex " << v;
            } else {
                EXPECT_NEAR(N(i), 0.0, 1e-12) << "Shape function " << i << " at vertex " << v;
            }
        }
    }

    // Test partition of unity at center
    Vec2 center{0.0, 0.0};
    DView1 N_center;
    DView2 dN_center;
    quad.compute_shape_functions(center, N_center, dN_center);

    EXPECT_NEAR(sum(N_center), 1.0, 1e-12) << "Partition of unity failed at center";

    // At center, all shape functions should be equal (by symmetry)
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(N_center(i), 0.25, 1e-12) << "Shape function " << i << " at center";
    }
}

// Test shape function gradients for quads
TEST(ReferenceQuadTest, ShapeFunctionGradients) {
    ReferenceQuad quad;

    // Test at center (0, 0)
    Vec2 center{0.0, 0.0};
    DView1 N;
    DView2 dN_dxi;
    quad.compute_shape_functions(center, N, dN_dxi);

    EXPECT_EQ(dN_dxi.extent(0), 4);
    EXPECT_EQ(dN_dxi.extent(1), 2);

    // At center, bilinear shape functions have specific gradients
    // N0 = 0.25(1-x)(1-y) => dN0/dx = -0.25(1-y) = -0.25 at (0,0)
    EXPECT_NEAR(dN_dxi(0, 0), -0.25, 1e-12);
    EXPECT_NEAR(dN_dxi(0, 1), -0.25, 1e-12);
    EXPECT_NEAR(dN_dxi(1, 0), 0.25, 1e-12);
    EXPECT_NEAR(dN_dxi(1, 1), -0.25, 1e-12);
    EXPECT_NEAR(dN_dxi(2, 0), 0.25, 1e-12);
    EXPECT_NEAR(dN_dxi(2, 1), 0.25, 1e-12);
    EXPECT_NEAR(dN_dxi(3, 0), -0.25, 1e-12);
    EXPECT_NEAR(dN_dxi(3, 1), 0.25, 1e-12);

    // Gradients should sum to zero
    EXPECT_NEAR(col_sum(dN_dxi, 0), 0.0, 1e-12);
    EXPECT_NEAR(col_sum(dN_dxi, 1), 0.0, 1e-12);
}

// Test project_to_bounds for quads
TEST(ReferenceQuadTest, ProjectToBounds) {
    ReferenceQuad quad;

    // Point inside - should not change
    Vec2 inside{0.3, -0.4};
    Vec2 inside_copy = inside;
    quad.project_to_bounds(inside);
    EXPECT_NEAR(norm(inside - inside_copy), 0.0, 1e-12);

    // Point outside x > 1
    Vec2 large_x{1.5, 0.0};
    quad.project_to_bounds(large_x);
    EXPECT_LE(large_x[0], 1.0);
    EXPECT_GE(large_x[0], -1.0);

    // Point outside x < -1
    Vec2 small_x{-1.5, 0.0};
    quad.project_to_bounds(small_x);
    EXPECT_LE(small_x[0], 1.0);
    EXPECT_GE(small_x[0], -1.0);

    // Point outside y bounds
    Vec2 outside_y{0.0, 2.0};
    quad.project_to_bounds(outside_y);
    EXPECT_LE(outside_y[1], 1.0);
    EXPECT_GE(outside_y[1], -1.0);
}

// Test polymorphic behavior through base class
TEST(ReferenceElementTest, PolymorphicShapeFunctions) {
    // Test triangle through base class pointer
    std::shared_ptr<ReferenceElement> tri = std::make_shared<ReferenceTriangle>();
    Vec2 pt_tri{0.3, 0.4};
    DView1 N_tri;
    DView2 dN_tri;

    tri->compute_shape_functions(pt_tri, N_tri, dN_tri);
    EXPECT_EQ(N_tri.size(), 3);
    EXPECT_NEAR(sum(N_tri), 1.0, 1e-12);

    // Test quad through base class pointer
    std::shared_ptr<ReferenceElement> quad = std::make_shared<ReferenceQuad>();
    Vec2 pt_quad{0.2, -0.3};
    DView1 N_quad;
    DView2 dN_quad;

    quad->compute_shape_functions(pt_quad, N_quad, dN_quad);
    EXPECT_EQ(N_quad.size(), 4);
    EXPECT_NEAR(sum(N_quad), 1.0, 1e-12);
}
