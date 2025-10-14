#include <Eigen/Dense>
#include <dgfem/reference/elements.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace dgfem;
using namespace testing;

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
    EXPECT_TRUE(tri.contains_point(Eigen::Vector2d(0.0, 0.0)));    // vertex
    EXPECT_TRUE(tri.contains_point(Eigen::Vector2d(1.0, 0.0)));    // vertex
    EXPECT_TRUE(tri.contains_point(Eigen::Vector2d(0.0, 1.0)));    // vertex
    EXPECT_TRUE(tri.contains_point(Eigen::Vector2d(0.25, 0.25)));  // interior

    // Test points outside
    EXPECT_FALSE(tri.contains_point(Eigen::Vector2d(-0.1, 0.0)));
    EXPECT_FALSE(tri.contains_point(Eigen::Vector2d(0.0, -0.1)));
    EXPECT_FALSE(tri.contains_point(Eigen::Vector2d(0.5, 0.6)));  // sum > 1
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
    std::vector<Eigen::Vector2d> vertices = {
        Eigen::Vector2d(0.0, 0.0),  // vertex 0
        Eigen::Vector2d(1.0, 0.0),  // vertex 1
        Eigen::Vector2d(0.0, 1.0)   // vertex 2
    };

    for (size_t v = 0; v < vertices.size(); ++v) {
        Eigen::VectorXd N;
        Eigen::MatrixXd dN_dxi;
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
    Eigen::Vector2d test_point(0.3, 0.4);
    Eigen::VectorXd N;
    Eigen::MatrixXd dN_dxi;
    tri.compute_shape_functions(test_point, N, dN_dxi);

    double sum = N.sum();
    EXPECT_NEAR(sum, 1.0, 1e-12) << "Partition of unity failed at interior point";
}

// Test shape function gradients for triangles
TEST(ReferenceTriangleTest, ShapeFunctionGradients) {
    ReferenceTriangle tri;

    Eigen::Vector2d test_point(0.25, 0.35);
    Eigen::VectorXd N;
    Eigen::MatrixXd dN_dxi;
    tri.compute_shape_functions(test_point, N, dN_dxi);

    EXPECT_EQ(dN_dxi.rows(), 3);
    EXPECT_EQ(dN_dxi.cols(), 2);

    // For linear triangle: N = [1-x-y, x, y]
    // Gradients should be constant
    EXPECT_NEAR(dN_dxi(0, 0), -1.0, 1e-12);  // dN0/dx
    EXPECT_NEAR(dN_dxi(0, 1), -1.0, 1e-12);  // dN0/dy
    EXPECT_NEAR(dN_dxi(1, 0), 1.0, 1e-12);   // dN1/dx
    EXPECT_NEAR(dN_dxi(1, 1), 0.0, 1e-12);   // dN1/dy
    EXPECT_NEAR(dN_dxi(2, 0), 0.0, 1e-12);   // dN2/dx
    EXPECT_NEAR(dN_dxi(2, 1), 1.0, 1e-12);   // dN2/dy

    // Verify gradients sum to zero (constant field has zero gradient)
    Eigen::VectorXd grad_sum_x = dN_dxi.col(0);
    Eigen::VectorXd grad_sum_y = dN_dxi.col(1);
    EXPECT_NEAR(grad_sum_x.sum(), 0.0, 1e-12);
    EXPECT_NEAR(grad_sum_y.sum(), 0.0, 1e-12);
}

// Test project_to_bounds for triangles
TEST(ReferenceTriangleTest, ProjectToBounds) {
    ReferenceTriangle tri;

    // Point inside - should not change
    Eigen::Vector2d inside(0.3, 0.4);
    Eigen::Vector2d inside_copy = inside;
    tri.project_to_bounds(inside);
    EXPECT_NEAR((inside - inside_copy).norm(), 0.0, 1e-12);

    // Point with x < 0
    Eigen::Vector2d neg_x(-0.1, 0.5);
    tri.project_to_bounds(neg_x);
    EXPECT_GE(neg_x(0), 0.0);
    EXPECT_LE(neg_x(0) + neg_x(1), 1.0);

    // Point with y < 0
    Eigen::Vector2d neg_y(0.5, -0.1);
    tri.project_to_bounds(neg_y);
    EXPECT_GE(neg_y(1), 0.0);
    EXPECT_LE(neg_y(0) + neg_y(1), 1.0);

    // Point with x+y > 1
    Eigen::Vector2d outside(0.7, 0.7);
    tri.project_to_bounds(outside);
    EXPECT_LE(outside(0) + outside(1), 1.0 + 1e-10);  // Small tolerance for rounding
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
    EXPECT_TRUE(quad.contains_point(Eigen::Vector2d(-1.0, -1.0)));  // vertex
    EXPECT_TRUE(quad.contains_point(Eigen::Vector2d(1.0, -1.0)));   // vertex
    EXPECT_TRUE(quad.contains_point(Eigen::Vector2d(1.0, 1.0)));    // vertex
    EXPECT_TRUE(quad.contains_point(Eigen::Vector2d(-1.0, 1.0)));   // vertex
    EXPECT_TRUE(quad.contains_point(Eigen::Vector2d(0.0, 0.0)));    // center

    // Test points outside
    EXPECT_FALSE(quad.contains_point(Eigen::Vector2d(-1.1, 0.0)));
    EXPECT_FALSE(quad.contains_point(Eigen::Vector2d(0.0, 1.1)));
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
    std::vector<Eigen::Vector2d> vertices = {
        Eigen::Vector2d(-1.0, -1.0),  // vertex 0
        Eigen::Vector2d(1.0, -1.0),   // vertex 1
        Eigen::Vector2d(1.0, 1.0),    // vertex 2
        Eigen::Vector2d(-1.0, 1.0)    // vertex 3
    };

    for (size_t v = 0; v < vertices.size(); ++v) {
        Eigen::VectorXd N;
        Eigen::MatrixXd dN_dxi;
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
    Eigen::Vector2d center(0.0, 0.0);
    Eigen::VectorXd N_center;
    Eigen::MatrixXd dN_center;
    quad.compute_shape_functions(center, N_center, dN_center);

    double sum = N_center.sum();
    EXPECT_NEAR(sum, 1.0, 1e-12) << "Partition of unity failed at center";

    // At center, all shape functions should be equal (by symmetry)
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(N_center(i), 0.25, 1e-12) << "Shape function " << i << " at center";
    }
}

// Test shape function gradients for quads
TEST(ReferenceQuadTest, ShapeFunctionGradients) {
    ReferenceQuad quad;

    // Test at center (0, 0)
    Eigen::Vector2d center(0.0, 0.0);
    Eigen::VectorXd N;
    Eigen::MatrixXd dN_dxi;
    quad.compute_shape_functions(center, N, dN_dxi);

    EXPECT_EQ(dN_dxi.rows(), 4);
    EXPECT_EQ(dN_dxi.cols(), 2);

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
    EXPECT_NEAR(dN_dxi.col(0).sum(), 0.0, 1e-12);
    EXPECT_NEAR(dN_dxi.col(1).sum(), 0.0, 1e-12);
}

// Test project_to_bounds for quads
TEST(ReferenceQuadTest, ProjectToBounds) {
    ReferenceQuad quad;

    // Point inside - should not change
    Eigen::Vector2d inside(0.3, -0.4);
    Eigen::Vector2d inside_copy = inside;
    quad.project_to_bounds(inside);
    EXPECT_NEAR((inside - inside_copy).norm(), 0.0, 1e-12);

    // Point outside x > 1
    Eigen::Vector2d large_x(1.5, 0.0);
    quad.project_to_bounds(large_x);
    EXPECT_LE(large_x(0), 1.0);
    EXPECT_GE(large_x(0), -1.0);

    // Point outside x < -1
    Eigen::Vector2d small_x(-1.5, 0.0);
    quad.project_to_bounds(small_x);
    EXPECT_LE(small_x(0), 1.0);
    EXPECT_GE(small_x(0), -1.0);

    // Point outside y bounds
    Eigen::Vector2d outside_y(0.0, 2.0);
    quad.project_to_bounds(outside_y);
    EXPECT_LE(outside_y(1), 1.0);
    EXPECT_GE(outside_y(1), -1.0);
}

// Test polymorphic behavior through base class
TEST(ReferenceElementTest, PolymorphicShapeFunctions) {
    // Test triangle through base class pointer
    std::shared_ptr<ReferenceElement> tri = std::make_shared<ReferenceTriangle>();
    Eigen::Vector2d pt_tri(0.3, 0.4);
    Eigen::VectorXd N_tri;
    Eigen::MatrixXd dN_tri;

    tri->compute_shape_functions(pt_tri, N_tri, dN_tri);
    EXPECT_EQ(N_tri.size(), 3);
    EXPECT_NEAR(N_tri.sum(), 1.0, 1e-12);

    // Test quad through base class pointer
    std::shared_ptr<ReferenceElement> quad = std::make_shared<ReferenceQuad>();
    Eigen::Vector2d pt_quad(0.2, -0.3);
    Eigen::VectorXd N_quad;
    Eigen::MatrixXd dN_quad;

    quad->compute_shape_functions(pt_quad, N_quad, dN_quad);
    EXPECT_EQ(N_quad.size(), 4);
    EXPECT_NEAR(N_quad.sum(), 1.0, 1e-12);
}
