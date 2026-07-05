#include <cmath>
#include <dgfem/kokkos_math.hpp>
#include <dgfem/reference/elements.hpp>
#include <dgfem/reference/mapping.hpp>

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
}  // namespace

TEST(GeometricMappingTest, TriangleShapeFunctions) {
    std::shared_ptr<ReferenceTriangle> tri = std::make_shared<ReferenceTriangle>();
    GeometricMapping mapping(tri);

    // Test at center
    Vec2 center{1.0 / 3.0, 1.0 / 3.0};
    DView1 N;
    DView2 dN;
    mapping.compute_shape_functions(center, N, dN);

    ASSERT_EQ(N.size(), 3);
    EXPECT_NEAR(N(0), 1.0 / 3.0, 1e-12);
    EXPECT_NEAR(N(1), 1.0 / 3.0, 1e-12);
    EXPECT_NEAR(N(2), 1.0 / 3.0, 1e-12);

    // Test partition of unity
    EXPECT_NEAR(sum(N), 1.0, 1e-12);
}

TEST(GeometricMappingTest, QuadShapeFunctions) {
    std::shared_ptr<ReferenceQuad> quad = std::make_shared<ReferenceQuad>();
    GeometricMapping mapping(quad);

    // Test at center
    Vec2 center{0.0, 0.0};
    DView1 N;
    DView2 dN;
    mapping.compute_shape_functions(center, N, dN);

    ASSERT_EQ(N.size(), 4);
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(N(i), 0.25, 1e-12);
    }

    // Test partition of unity
    EXPECT_NEAR(sum(N), 1.0, 1e-12);
}

TEST(GeometricMappingTest, TriangleShapeFunctionDerivatives) {
    std::shared_ptr<ReferenceTriangle> tri = std::make_shared<ReferenceTriangle>();
    GeometricMapping mapping(tri);

    // Test at any point (derivatives are constant for linear triangle)
    Vec2 point{0.25, 0.25};
    DView1 N;
    DView2 dN;
    mapping.compute_shape_functions(point, N, dN);

    ASSERT_EQ(dN.extent(0), 3);
    ASSERT_EQ(dN.extent(1), 2);

    // Known derivatives for triangle
    double expected[3][2] = {{-1, -1}, {1, 0}, {0, 1}};

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 2; ++j) {
            EXPECT_NEAR(dN(i, j), expected[i][j], 1e-12);
        }
    }
}

TEST(GeometricMappingTest, QuadShapeFunctionDerivatives) {
    std::shared_ptr<ReferenceQuad> quad = std::make_shared<ReferenceQuad>();
    GeometricMapping mapping(quad);

    // Test at center
    Vec2 center{0.0, 0.0};
    DView1 N;
    DView2 dN;
    mapping.compute_shape_functions(center, N, dN);

    ASSERT_EQ(dN.extent(0), 4);
    ASSERT_EQ(dN.extent(1), 2);

    // At center, derivatives should match known values
    double expected[4][2] = {{-0.25, -0.25}, {0.25, -0.25}, {0.25, 0.25}, {-0.25, 0.25}};

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 2; ++j) {
            EXPECT_NEAR(dN(i, j), expected[i][j], 1e-12);
        }
    }
}

TEST(GeometricMappingTest, ComputeMapping) {
    std::shared_ptr<ReferenceTriangle> tri = std::make_shared<ReferenceTriangle>();
    GeometricMapping mapping(tri);

    // Create a physical triangle
    DView2 vertices("vertices", 3, 2);
    set_row2(vertices, 0, Vec2{0.0, 0.0});
    set_row2(vertices, 1, Vec2{1.0, 0.0});
    set_row2(vertices, 2, Vec2{0.0, 1.0});

    // Test at center of reference triangle
    Vec2 xi{1.0 / 3.0, 1.0 / 3.0};
    auto result = mapping.compute_mapping(vertices, xi);

    // Physical coordinates should be at centroid
    EXPECT_NEAR(result.x_phys[0], 1.0 / 3.0, 1e-12);
    EXPECT_NEAR(result.x_phys[1], 1.0 / 3.0, 1e-12);

    // Jacobian determinant should be area * 2
    EXPECT_NEAR(result.J_T_det, 1.0, 1e-12);

    // Test grad_transform (dxi_dx)
    Mat2 expected_transform{1.0, 0.0, 0.0, 1.0};

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
    DView2 vertices("vertices", 3, 2);
    set_row2(vertices, 0, Vec2{0.0, 0.0});
    set_row2(vertices, 1, Vec2{0.0, 0.0});  // Coincident vertex
    set_row2(vertices, 2, Vec2{0.0, 1.0});

    Vec2 xi{1.0 / 3.0, 1.0 / 3.0};

    // Expect the function to throw a runtime_error for a singular Jacobian
    EXPECT_THROW(mapping.compute_mapping(vertices, xi), std::runtime_error);
}

// Test affine mapping preservation
TEST(GeometricMappingTest, AffineMappingTriangle) {
    std::shared_ptr<ReferenceTriangle> tri = std::make_shared<ReferenceTriangle>();
    GeometricMapping mapping(tri);

    // Create an affine-mapped triangle (translation + rotation + scaling)
    DView2 vertices("vertices", 3, 2);
    set_row2(vertices, 0, Vec2{1.0, 2.0});
    set_row2(vertices, 1, Vec2{3.0, 2.0});
    set_row2(vertices, 2, Vec2{1.0, 4.0});

    // Test at multiple reference points
    std::vector<Vec2> ref_points = {Vec2{0.0, 0.0}, Vec2{1.0, 0.0}, Vec2{0.0, 1.0},
                                    Vec2{0.5, 0.25}};

    for (const auto& xi : ref_points) {
        auto result = mapping.compute_mapping(vertices, xi);

        // Verify mapping is consistent: x_phys = sum(N_i * vertex_i)
        DView1 N;
        DView2 dN_dxi;
        mapping.compute_shape_functions(xi, N, dN_dxi);

        Vec2 expected_x{0.0, 0.0};
        for (int i = 0; i < 3; ++i) {
            expected_x = expected_x + N(i) * row2(vertices, i);
        }

        EXPECT_NEAR(norm(result.x_phys - expected_x), 0.0, 1e-12)
            << "Mapping inconsistent at xi = (" << xi[0] << ", " << xi[1] << ")";
    }
}

// Test quad mapping with distortion
TEST(GeometricMappingTest, BilinearQuadMapping) {
    std::shared_ptr<ReferenceQuad> quad = std::make_shared<ReferenceQuad>();
    GeometricMapping mapping(quad);

    // Create a distorted quad (not rectangular)
    DView2 vertices("vertices", 4, 2);
    set_row2(vertices, 0, Vec2{0.0, 0.0});
    set_row2(vertices, 1, Vec2{2.0, 0.2});
    set_row2(vertices, 2, Vec2{2.1, 2.0});
    set_row2(vertices, 3, Vec2{0.1, 1.9});

    Vec2 center{0.0, 0.0};
    auto result = mapping.compute_mapping(vertices, center);

    // Physical center should be near average of vertices
    Vec2 expected_center{0.0, 0.0};
    for (int i = 0; i < 4; ++i) {
        expected_center = expected_center + row2(vertices, i);
    }
    expected_center = expected_center * 0.25;
    EXPECT_NEAR(norm(result.x_phys - expected_center), 0.0,
                0.1)  // Larger tolerance for distorted element
        << "Quad center mapping inaccurate";

    // Jacobian should be positive
    EXPECT_GT(result.J_T_det, 0.0) << "Jacobian should be positive for valid element";
}

// Test inverse mapping (find_reference_coords)
TEST(GeometricMappingTest, InverseMappingTriangle) {
    std::shared_ptr<ReferenceTriangle> tri = std::make_shared<ReferenceTriangle>();
    GeometricMapping mapping(tri);

    DView2 vertices("vertices", 3, 2);
    set_row2(vertices, 0, Vec2{1.0, 1.0});
    set_row2(vertices, 1, Vec2{4.0, 1.0});
    set_row2(vertices, 2, Vec2{1.0, 4.0});

    // Test various reference points
    std::vector<Vec2> ref_points = {Vec2{0.0, 0.0}, Vec2{1.0, 0.0}, Vec2{0.0, 1.0}, Vec2{0.3, 0.4},
                                    Vec2{0.25, 0.25}};

    for (const auto& xi_ref : ref_points) {
        // Map to physical
        auto map_data = mapping.compute_mapping(vertices, xi_ref);
        Vec2 x_phys = map_data.x_phys;

        // Find reference coordinates
        auto xi_found = mapping.find_reference_coords(vertices, x_phys);

        ASSERT_TRUE(xi_found.has_value()) << "Failed to find reference coords for physical point";

        EXPECT_NEAR(norm(xi_found.value() - xi_ref), 0.0, 1e-8)
            << "Inverse mapping inaccurate for xi = (" << xi_ref[0] << ", " << xi_ref[1] << ")";
    }
}

// Test inverse mapping for quad
TEST(GeometricMappingTest, InverseMappingQuad) {
    std::shared_ptr<ReferenceQuad> quad = std::make_shared<ReferenceQuad>();
    GeometricMapping mapping(quad);

    DView2 vertices("vertices", 4, 2);
    set_row2(vertices, 0, Vec2{0.0, 0.0});
    set_row2(vertices, 1, Vec2{3.0, 0.0});
    set_row2(vertices, 2, Vec2{3.0, 2.0});
    set_row2(vertices, 3, Vec2{0.0, 2.0});

    // Test various reference points
    std::vector<Vec2> ref_points = {Vec2{0.0, 0.0}, Vec2{0.5, -0.5}, Vec2{-0.7, 0.3}};

    for (const auto& xi_ref : ref_points) {
        auto map_data = mapping.compute_mapping(vertices, xi_ref);
        Vec2 x_phys = map_data.x_phys;

        auto xi_found = mapping.find_reference_coords(vertices, x_phys);

        ASSERT_TRUE(xi_found.has_value());
        EXPECT_NEAR(norm(xi_found.value() - xi_ref), 0.0, 1e-8)
            << "Inverse quad mapping inaccurate";
    }
}

// Test inverse mapping failure for point outside element
TEST(GeometricMappingTest, InverseMappingOutside) {
    std::shared_ptr<ReferenceTriangle> tri = std::make_shared<ReferenceTriangle>();
    GeometricMapping mapping(tri);

    DView2 vertices("vertices", 3, 2);
    set_row2(vertices, 0, Vec2{0.0, 0.0});
    set_row2(vertices, 1, Vec2{1.0, 0.0});
    set_row2(vertices, 2, Vec2{0.0, 1.0});

    // Point clearly outside the triangle
    Vec2 outside_point{5.0, 5.0};

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

    DView2 vertices("vertices", 4, 2);
    set_row2(vertices, 0, Vec2{0.0, 0.0});
    set_row2(vertices, 1, Vec2{2.0, 0.0});
    set_row2(vertices, 2, Vec2{2.0, 3.0});
    set_row2(vertices, 3, Vec2{0.0, 3.0});

    Vec2 xi{0.25, -0.35};
    auto result = mapping.compute_mapping(vertices, xi);

    // J.transpose() * dxi_dx should be identity
    // Since J = dx/dxi and dxi_dx = inv((dx/dxi)^T)
    Mat2 product = result.J.transpose() * result.dxi_dx;
    Mat2 identity = Mat2::identity();

    EXPECT_NEAR(norm(product - identity), 0.0, 1e-10)
        << "J.transpose() * dxi_dx should equal identity";

    // Also check: dxi_dx * J.transpose() should be identity
    Mat2 product2 = result.dxi_dx * result.J.transpose();
    EXPECT_NEAR(norm(product2 - identity), 0.0, 1e-10)
        << "dxi_dx * J.transpose() should equal identity";
}

// Test map_to_physical convenience method
TEST(GeometricMappingTest, MapToPhysical) {
    std::shared_ptr<ReferenceTriangle> tri = std::make_shared<ReferenceTriangle>();
    GeometricMapping mapping(tri);

    DView2 vertices("vertices", 3, 2);
    set_row2(vertices, 0, Vec2{2.0, 3.0});
    set_row2(vertices, 1, Vec2{5.0, 3.0});
    set_row2(vertices, 2, Vec2{2.0, 6.0});

    Vec2 xi{0.5, 0.25};

    // Use convenience method
    Vec2 x_phys = mapping.map_to_physical(vertices, xi);

    // Compare with full mapping
    auto map_data = mapping.compute_mapping(vertices, xi);

    EXPECT_NEAR(norm(x_phys - map_data.x_phys), 0.0, 1e-12)
        << "map_to_physical should match compute_mapping";
}
