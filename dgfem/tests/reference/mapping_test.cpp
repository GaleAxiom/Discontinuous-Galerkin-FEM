#include <Eigen/Dense>
#include <cmath>
#include <dgfem/reference/elements.hpp>
#include <dgfem/reference/mapping.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace dgfem;
using namespace testing;

TEST(GeometricMappingTest, TriangleShapeFunctions) {
    std::shared_ptr<ReferenceTriangle> tri = std::make_shared<ReferenceTriangle>();
    GeometricMapping mapping(tri);

    // Test at center
    Eigen::Vector2d center(1.0 / 3.0, 1.0 / 3.0);
    Eigen::VectorXd N;
    Eigen::MatrixXd dN;
    mapping.compute_shape_functions(center, N, dN);

    ASSERT_EQ(N.size(), 3);
    EXPECT_NEAR(N(0), 1.0 / 3.0, 1e-12);
    EXPECT_NEAR(N(1), 1.0 / 3.0, 1e-12);
    EXPECT_NEAR(N(2), 1.0 / 3.0, 1e-12);

    // Test partition of unity
    EXPECT_NEAR(N.sum(), 1.0, 1e-12);
}

TEST(GeometricMappingTest, QuadShapeFunctions) {
    std::shared_ptr<ReferenceQuad> quad = std::make_shared<ReferenceQuad>();
    GeometricMapping mapping(quad);

    // Test at center
    Eigen::Vector2d center(0.0, 0.0);
    Eigen::VectorXd N;
    Eigen::MatrixXd dN;
    mapping.compute_shape_functions(center, N, dN);

    ASSERT_EQ(N.size(), 4);
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(N(i), 0.25, 1e-12);
    }

    // Test partition of unity
    EXPECT_NEAR(N.sum(), 1.0, 1e-12);
}

TEST(GeometricMappingTest, TriangleShapeFunctionDerivatives) {
    std::shared_ptr<ReferenceTriangle> tri = std::make_shared<ReferenceTriangle>();
    GeometricMapping mapping(tri);

    // Test at any point (derivatives are constant for linear triangle)
    Eigen::Vector2d point(0.25, 0.25);
    Eigen::VectorXd N;
    Eigen::MatrixXd dN;
    mapping.compute_shape_functions(point, N, dN);

    ASSERT_EQ(dN.rows(), 3);
    ASSERT_EQ(dN.cols(), 2);

    // Known derivatives for triangle
    Eigen::MatrixXd expected(3, 2);
    expected << -1, -1, 1, 0, 0, 1;

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 2; ++j) {
            EXPECT_NEAR(dN(i, j), expected(i, j), 1e-12);
        }
    }
}

TEST(GeometricMappingTest, QuadShapeFunctionDerivatives) {
    std::shared_ptr<ReferenceQuad> quad = std::make_shared<ReferenceQuad>();
    GeometricMapping mapping(quad);

    // Test at center
    Eigen::Vector2d center(0.0, 0.0);
    Eigen::VectorXd N;
    Eigen::MatrixXd dN;
    mapping.compute_shape_functions(center, N, dN);

    ASSERT_EQ(dN.rows(), 4);
    ASSERT_EQ(dN.cols(), 2);

    // At center, derivatives should match known values
    Eigen::MatrixXd expected(4, 2);
    expected << -0.25, -0.25, 0.25, -0.25, 0.25, 0.25, -0.25, 0.25;

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 2; ++j) {
            EXPECT_NEAR(dN(i, j), expected(i, j), 1e-12);
        }
    }
}

TEST(GeometricMappingTest, ComputeMapping) {
    std::shared_ptr<ReferenceTriangle> tri = std::make_shared<ReferenceTriangle>();
    GeometricMapping mapping(tri);

    // Create a physical triangle
    Eigen::MatrixXd vertices(3, 2);
    vertices << 0.0, 0.0, 1.0, 0.0, 0.0, 1.0;

    // Test at center of reference triangle
    Eigen::Vector2d xi(1.0 / 3.0, 1.0 / 3.0);
    auto result = mapping.compute_mapping(vertices, xi);

    // Physical coordinates should be at centroid
    EXPECT_NEAR(result.x_phys(0), 1.0 / 3.0, 1e-12);
    EXPECT_NEAR(result.x_phys(1), 1.0 / 3.0, 1e-12);

    // Jacobian determinant should be area * 2
    EXPECT_NEAR(result.J_T_det, 1.0, 1e-12);

    // Test grad_transform (dxi_dx)
    Eigen::Matrix2d expected_transform;
    expected_transform << 1.0, 0.0, 0.0, 1.0;

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            EXPECT_NEAR(result.dxi_dx(i, j), expected_transform(i, j), 1e-12);
        }
    }
}

TEST(GeometricMappingTest, SingularJacobian) {
    std::shared_ptr<ReferenceTriangle> tri = std::make_shared<ReferenceTriangle>();
    GeometricMapping mapping(tri);

    // Create a degenerate triangle
    Eigen::MatrixXd vertices(3, 2);
    vertices << 0.0, 0.0, 0.0, 0.0,  // Coincident vertex
        0.0, 1.0;

    Eigen::Vector2d xi(1.0 / 3.0, 1.0 / 3.0);

    // Expect the function to throw a runtime_error for a singular Jacobian
    EXPECT_THROW(mapping.compute_mapping(vertices, xi), std::runtime_error);
}

// Test affine mapping preservation
TEST(GeometricMappingTest, AffineMappingTriangle) {
    std::shared_ptr<ReferenceTriangle> tri = std::make_shared<ReferenceTriangle>();
    GeometricMapping mapping(tri);

    // Create an affine-mapped triangle (translation + rotation + scaling)
    Eigen::MatrixXd vertices(3, 2);
    vertices << 1.0, 2.0, 3.0, 2.0, 1.0, 4.0;

    // Test at multiple reference points
    std::vector<Eigen::Vector2d> ref_points = {Eigen::Vector2d(0.0, 0.0), Eigen::Vector2d(1.0, 0.0),
                                               Eigen::Vector2d(0.0, 1.0),
                                               Eigen::Vector2d(0.5, 0.25)};

    for (const auto& xi : ref_points) {
        auto result = mapping.compute_mapping(vertices, xi);

        // Verify mapping is consistent: x_phys = sum(N_i * vertex_i)
        Eigen::VectorXd N;
        Eigen::MatrixXd dN_dxi;
        mapping.compute_shape_functions(xi, N, dN_dxi);

        Eigen::Vector2d expected_x = Eigen::Vector2d::Zero();
        for (int i = 0; i < 3; ++i) {
            expected_x += N(i) * vertices.row(i).transpose();
        }

        EXPECT_NEAR((result.x_phys - expected_x).norm(), 0.0, 1e-12)
            << "Mapping inconsistent at xi = (" << xi(0) << ", " << xi(1) << ")";
    }
}

// Test quad mapping with distortion
TEST(GeometricMappingTest, BilinearQuadMapping) {
    std::shared_ptr<ReferenceQuad> quad = std::make_shared<ReferenceQuad>();
    GeometricMapping mapping(quad);

    // Create a distorted quad (not rectangular)
    Eigen::MatrixXd vertices(4, 2);
    vertices << 0.0, 0.0, 2.0, 0.2, 2.1, 2.0, 0.1, 1.9;

    Eigen::Vector2d center(0.0, 0.0);
    auto result = mapping.compute_mapping(vertices, center);

    // Physical center should be near average of vertices
    Eigen::Vector2d expected_center = vertices.colwise().mean();
    EXPECT_NEAR((result.x_phys - expected_center).norm(), 0.0,
                0.1)  // Larger tolerance for distorted element
        << "Quad center mapping inaccurate";

    // Jacobian should be positive
    EXPECT_GT(result.J_T_det, 0.0) << "Jacobian should be positive for valid element";
}

// Test inverse mapping (find_reference_coords)
TEST(GeometricMappingTest, InverseMappingTriangle) {
    std::shared_ptr<ReferenceTriangle> tri = std::make_shared<ReferenceTriangle>();
    GeometricMapping mapping(tri);

    Eigen::MatrixXd vertices(3, 2);
    vertices << 1.0, 1.0, 4.0, 1.0, 1.0, 4.0;

    // Test various reference points
    std::vector<Eigen::Vector2d> ref_points = {Eigen::Vector2d(0.0, 0.0), Eigen::Vector2d(1.0, 0.0),
                                               Eigen::Vector2d(0.0, 1.0), Eigen::Vector2d(0.3, 0.4),
                                               Eigen::Vector2d(0.25, 0.25)};

    for (const auto& xi_ref : ref_points) {
        // Map to physical
        auto map_data = mapping.compute_mapping(vertices, xi_ref);
        Eigen::Vector2d x_phys = map_data.x_phys;

        // Find reference coordinates
        auto xi_found = mapping.find_reference_coords(vertices, x_phys);

        ASSERT_TRUE(xi_found.has_value()) << "Failed to find reference coords for physical point";

        EXPECT_NEAR((xi_found.value() - xi_ref).norm(), 0.0, 1e-8)
            << "Inverse mapping inaccurate for xi = (" << xi_ref(0) << ", " << xi_ref(1) << ")";
    }
}

// Test inverse mapping for quad
TEST(GeometricMappingTest, InverseMappingQuad) {
    std::shared_ptr<ReferenceQuad> quad = std::make_shared<ReferenceQuad>();
    GeometricMapping mapping(quad);

    Eigen::MatrixXd vertices(4, 2);
    vertices << 0.0, 0.0, 3.0, 0.0, 3.0, 2.0, 0.0, 2.0;

    // Test various reference points
    std::vector<Eigen::Vector2d> ref_points = {
        Eigen::Vector2d(0.0, 0.0), Eigen::Vector2d(0.5, -0.5), Eigen::Vector2d(-0.7, 0.3)};

    for (const auto& xi_ref : ref_points) {
        auto map_data = mapping.compute_mapping(vertices, xi_ref);
        Eigen::Vector2d x_phys = map_data.x_phys;

        auto xi_found = mapping.find_reference_coords(vertices, x_phys);

        ASSERT_TRUE(xi_found.has_value());
        EXPECT_NEAR((xi_found.value() - xi_ref).norm(), 0.0, 1e-8)
            << "Inverse quad mapping inaccurate";
    }
}

// Test inverse mapping failure for point outside element
TEST(GeometricMappingTest, InverseMappingOutside) {
    std::shared_ptr<ReferenceTriangle> tri = std::make_shared<ReferenceTriangle>();
    GeometricMapping mapping(tri);

    Eigen::MatrixXd vertices(3, 2);
    vertices << 0.0, 0.0, 1.0, 0.0, 0.0, 1.0;

    // Point clearly outside the triangle
    Eigen::Vector2d outside_point(5.0, 5.0);

    auto result = mapping.find_reference_coords(vertices, outside_point, 1e-10, 50);

    // Should either return nullopt or a point outside reference domain
    if (result.has_value()) {
        // If it returns something, it should not be in the reference element
        EXPECT_FALSE(tri->contains_point(result.value()))
            << "Incorrectly found reference coords for outside point";
    }
}

// Test Jacobian consistency
TEST(GeometricMappingTest, JacobianConsistency) {
    std::shared_ptr<ReferenceQuad> quad = std::make_shared<ReferenceQuad>();
    GeometricMapping mapping(quad);

    Eigen::MatrixXd vertices(4, 2);
    vertices << 0.0, 0.0, 2.0, 0.0, 2.0, 3.0, 0.0, 3.0;

    Eigen::Vector2d xi(0.25, -0.35);
    auto result = mapping.compute_mapping(vertices, xi);

    // J.transpose() * dxi_dx should be identity
    // Since J = dx/dxi and dxi_dx = inv((dx/dxi)^T)
    Eigen::Matrix2d product = result.J.transpose() * result.dxi_dx;
    Eigen::Matrix2d identity = Eigen::Matrix2d::Identity();

    EXPECT_NEAR((product - identity).norm(), 0.0, 1e-10)
        << "J.transpose() * dxi_dx should equal identity";

    // Also check: dxi_dx * J.transpose() should be identity
    Eigen::Matrix2d product2 = result.dxi_dx * result.J.transpose();
    EXPECT_NEAR((product2 - identity).norm(), 0.0, 1e-10)
        << "dxi_dx * J.transpose() should equal identity";
}

// Test map_to_physical convenience method
TEST(GeometricMappingTest, MapToPhysical) {
    std::shared_ptr<ReferenceTriangle> tri = std::make_shared<ReferenceTriangle>();
    GeometricMapping mapping(tri);

    Eigen::MatrixXd vertices(3, 2);
    vertices << 2.0, 3.0, 5.0, 3.0, 2.0, 6.0;

    Eigen::Vector2d xi(0.5, 0.25);

    // Use convenience method
    Eigen::Vector2d x_phys = mapping.map_to_physical(vertices, xi);

    // Compare with full mapping
    auto map_data = mapping.compute_mapping(vertices, xi);

    EXPECT_NEAR((x_phys - map_data.x_phys).norm(), 0.0, 1e-12)
        << "map_to_physical should match compute_mapping";
}
