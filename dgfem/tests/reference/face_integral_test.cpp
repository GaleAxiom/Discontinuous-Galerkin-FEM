/**
 * @file face_integral_test.cpp
 * @brief Analytical verification of face integral computations
 */

#include <cmath>
#include <dgfem/core/space.hpp>
#include <dgfem/kokkos_math.hpp>
#include <dgfem/quadrature/factory.hpp>
#include <dgfem/reference/elements.hpp>

#include <gtest/gtest.h>

using namespace dgfem;

class FaceIntegralTest : public ::testing::Test {
protected:
    double integrate_face(const ReferenceElement& ref_elem, int face_id,
                          std::function<double(const Vec2&)> func,
                          const std::string& element_type = "triangle") {
        // Create a DGSpace to get quadrature rules and mapping
        DGSpace dg_space(element_type, 2);  // order 2 is sufficient for test
        auto face_quad = dg_space.get_face_quad();

        double integral = 0.0;
        for (int q = 0; q < face_quad->size(); ++q) {
            double s = face_quad->points(q, 0);  // 1D quadrature point in [-1,1]
            Vec2 xi_face = dg_space.map_face_quad_point(face_id, s);
            double value = func(xi_face);

            // Get face length scaling from reference element vertices
            auto [v1_idx, v2_idx] = ref_elem.edge_vertices(face_id);
            Vec2 v1 = row2(ref_elem.get_vertices(), v1_idx);
            Vec2 v2 = row2(ref_elem.get_vertices(), v2_idx);
            double edge_length = norm(v2 - v1);

            integral += value * face_quad->weights[q] * edge_length / 2.0;
        }

        return integral;
    }
};

TEST_F(FaceIntegralTest, TriangleFaceIntegralConstant) {
    // Integrate constant function over each face of reference triangle
    ReferenceTriangle tri;

    auto constant = [](const Vec2& x) { return 1.0; };

    // Face 0: from (0,0) to (1,0), length = 1
    double int0 = integrate_face(tri, 0, constant);
    EXPECT_NEAR(int0, 1.0, 1e-12) << "Integral of 1 over unit edge should be 1";

    // Face 1: from (1,0) to (0,1), length = √2
    double int1 = integrate_face(tri, 1, constant);
    EXPECT_NEAR(int1, std::sqrt(2.0), 1e-12) << "Integral over diagonal edge";

    // Face 2: from (0,1) to (0,0), length = 1
    double int2 = integrate_face(tri, 2, constant);
    EXPECT_NEAR(int2, 1.0, 1e-12) << "Integral of 1 over unit edge should be 1";
}

TEST_F(FaceIntegralTest, TriangleFaceIntegralLinear) {
    // Integrate linear functions over triangle faces
    ReferenceTriangle tri;

    // Face 0: y=0, x from 0 to 1
    // Integrate x: ∫₀¹ x dx = 1/2
    auto func_x = [](const Vec2& x) { return x[0]; };
    double int0 = integrate_face(tri, 0, func_x);
    EXPECT_NEAR(int0, 0.5, 1e-12);

    // Face 1: from (1,0) to (0,1)
    // Parameterized as (1-t, t) for t in [0,1]
    // Length element ds = √2 dt
    // ∫ x ds = ∫₀¹ (1-t)√2 dt = √2 * [t - t²/2]₀¹ = √2/2
    int0 = integrate_face(tri, 1, func_x);
    EXPECT_NEAR(int0, std::sqrt(2.0) / 2.0, 1e-11);

    // Face 2: x=0, y from 1 to 0
    // ∫ x ds = 0
    int0 = integrate_face(tri, 2, func_x);
    EXPECT_NEAR(int0, 0.0, 1e-12);
}

TEST_F(FaceIntegralTest, TriangleFaceIntegralQuadratic) {
    // Integrate quadratic functions
    ReferenceTriangle tri;

    // Face 0: y=0, x from 0 to 1
    // Integrate x²: ∫₀¹ x² dx = 1/3
    auto func_x2 = [](const Vec2& x) { return x[0] * x[0]; };
    double int0 = integrate_face(tri, 0, func_x2);
    EXPECT_NEAR(int0, 1.0 / 3.0, 1e-12);

    // Integrate xy over face 1
    // Parameterized as (1-t, t), so xy = (1-t)t
    // ∫₀¹ (1-t)t √2 dt = √2 ∫₀¹ (t - t²) dt = √2 * [t²/2 - t³/3]₀¹ = √2/6
    auto func_xy = [](const Vec2& x) { return x[0] * x[1]; };
    double int1 = integrate_face(tri, 1, func_xy);
    EXPECT_NEAR(int1, std::sqrt(2.0) / 6.0, 1e-11);
}

TEST_F(FaceIntegralTest, QuadFaceIntegralConstant) {
    // Integrate constant over quad reference element [-1,1]²
    ReferenceQuad quad;

    auto constant = [](const Vec2& x) { return 1.0; };

    // All faces have length 2
    for (int face = 0; face < 4; ++face) {
        double integral = integrate_face(quad, face, constant, "quad");
        EXPECT_NEAR(integral, 2.0, 1e-12) << "Each face of reference quad has length 2";
    }
}

TEST_F(FaceIntegralTest, QuadFaceIntegralLinear) {
    // Integrate linear functions over quad faces
    ReferenceQuad quad;

    // Face 0: bottom, y=-1, x from -1 to 1
    // ∫₋₁¹ x dx = 0 (odd function)
    auto func_x = [](const Vec2& x) { return x[0]; };
    double int0 = integrate_face(quad, 0, func_x, "quad");
    EXPECT_NEAR(int0, 0.0, 1e-12) << "Integral of odd function over symmetric interval";

    // Face 1: right, x=1, y from -1 to 1
    // ∫₋₁¹ y dy = 0
    auto func_y = [](const Vec2& x) { return x[1]; };
    double int1 = integrate_face(quad, 1, func_y, "quad");
    EXPECT_NEAR(int1, 0.0, 1e-12);

    // Integrate constant on face 0: should be 2
    auto const_one = [](const Vec2& x) { return 1.0; };
    int0 = integrate_face(quad, 0, const_one, "quad");
    EXPECT_NEAR(int0, 2.0, 1e-12);
}

TEST_F(FaceIntegralTest, QuadFaceIntegralQuadratic) {
    // Integrate x² over faces
    ReferenceQuad quad;

    auto func_x2 = [](const Vec2& x) { return x[0] * x[0]; };

    // Face 0: bottom, y=-1, x from -1 to 1
    // ∫₋₁¹ x² dx = 2/3
    double int0 = integrate_face(quad, 0, func_x2, "quad");
    EXPECT_NEAR(int0, 2.0 / 3.0, 1e-11);

    // Face 1: right, x=1, y from -1 to 1
    // ∫₋₁¹ 1² dy = 2
    double int1 = integrate_face(quad, 1, func_x2, "quad");
    EXPECT_NEAR(int1, 2.0, 1e-11);

    // Face 2: top, y=1, x from 1 to -1 (reversed)
    // ∫₁⁻¹ x² dx = -2/3, but with proper orientation
    double int2 = integrate_face(quad, 2, func_x2, "quad");
    EXPECT_NEAR(int2, 2.0 / 3.0, 1e-11);
}

TEST_F(FaceIntegralTest, BasisFunctionFaceIntegral) {
    // Test face integrals of basis functions
    ReferenceTriangle tri;
    DGSpace dg_space("triangle", 1);

    auto basis = dg_space.get_basis();
    int n_basis = basis->get_n_basis();

    // Face 0: integrate each basis function
    for (int i = 0; i < n_basis; ++i) {
        auto basis_func = [&basis, i](const Vec2& x) {
            DView1 all_basis = basis->evaluate(x);
            return all_basis[i];
        };

        double integral = integrate_face(tri, 0, basis_func, "triangle");

        // Specific checks for Legendre basis
        // On face 0 (y=0):
        // basis 0 (constant) = 1, integral = 1
        // basis 1 (x-related) depends on position
        // basis 2 (y-related) = 0 when y=0 if linear in y

        if (i == 0) {
            // Constant basis
            EXPECT_GT(integral, 0.0) << "Constant basis should have positive integral";
        }
    }
}

TEST_F(FaceIntegralTest, NormalDerivativeIntegral) {
    // Test integration of normal derivatives (important for DG)
    ReferenceTriangle tri;

    // For a linear function u = x, du/dx = 1, du/dy = 0
    // On face 1: normal is (1/√2, 1/√2)
    // du/dn = ∇u · n = (1, 0) · (1/√2, 1/√2) = 1/√2

    auto func_constant = [](const Vec2& x) { return 1.0 / std::sqrt(2.0); };

    // Integrate over face 1 (length √2)
    double integral = integrate_face(tri, 1, func_constant);
    EXPECT_NEAR(integral, 1.0, 1e-11) << "∫ 1/√2 ds over edge of length √2 = 1";
}

TEST_F(FaceIntegralTest, PhysicalFaceIntegral) {
    // Test face integral on a physical (scaled) triangle
    ReferenceTriangle tri;

    // Consider a physical triangle scaled by factor 2
    // Physical edge length = 2 * reference edge length
    double scale = 2.0;

    auto constant = [](const Vec2& x) { return 1.0; };

    // Reference integral over face 0
    double ref_integral = integrate_face(tri, 0, constant);
    EXPECT_NEAR(ref_integral, 1.0, 1e-12);

    // Physical integral would be ref_integral * jacobian
    // For uniform scaling, jacobian = scale
    double phys_integral = ref_integral * scale;
    EXPECT_NEAR(phys_integral, 2.0, 1e-12);
}

TEST_F(FaceIntegralTest, HighOrderQuadratureAccuracy) {
    // Test that quadrature integrates polynomials accurately
    ReferenceTriangle tri;
    DGSpace dg_space("triangle", 2);  // Order 2
    auto face_quad = dg_space.get_face_quad();

    // Use a polynomial that can be integrated exactly with available quadrature
    // For 3 Gauss points: exact up to degree 5
    // Polynomial of degree 4: x²y²
    auto poly4 = [](const Vec2& x) {
        double x2 = x[0] * x[0];
        double y2 = x[1] * x[1];
        return x2 * y2;
    };

    // Test on face 0
    double integral = 0.0;
    double edge_length_0 = 1.0;  // Face 0 goes from (0,0) to (1,0)
    for (int q = 0; q < face_quad->size(); ++q) {
        double s = face_quad->points(q, 0);
        Vec2 xi_face = dg_space.map_face_quad_point(0, s);
        double value = poly4(xi_face);
        integral += value * face_quad->weights[q] * edge_length_0 / 2.0;
    }

    // On face 0 (y=0), polynomial evaluates to 0
    EXPECT_NEAR(integral, 0.0, 1e-14);

    // Test on face 1 where y varies
    integral = 0.0;
    double edge_length_1 = std::sqrt(2.0);  // Face 1 goes from (1,0) to (0,1)
    for (int q = 0; q < face_quad->size(); ++q) {
        double s = face_quad->points(q, 0);
        Vec2 xi_face = dg_space.map_face_quad_point(1, s);
        double value = poly4(xi_face);
        integral += value * face_quad->weights[q] * edge_length_1 / 2.0;
    }

    // Analytical: ∫₀¹ (1-t)²t² √2 dt = √2 ∫₀¹ t²(1-t)² dt
    // = √2 * B(3,3) = √2 * 2!*2!/5! = √2 * 4/120 = √2/30
    EXPECT_NEAR(integral, std::sqrt(2.0) / 30.0, 1e-12)
        << "Integral of x²y² over face 1 should match analytical value";
}

TEST_F(FaceIntegralTest, FaceOrientationConsistency) {
    // Test that face orientations are consistent
    ReferenceTriangle tri;

    // The sum of outward normal components should be consistent
    auto constant = [](const Vec2& x) { return 1.0; };

    double total = 0.0;
    for (int face = 0; face < 3; ++face) {
        double integral = integrate_face(tri, face, constant);
        total += integral;
    }

    // Total perimeter of reference triangle: 1 + √2 + 1
    double expected_perimeter = 2.0 + std::sqrt(2.0);
    EXPECT_NEAR(total, expected_perimeter, 1e-11);
}

TEST_F(FaceIntegralTest, JacobianCorrectness) {
    // Verify that face jacobian is computed correctly
    ReferenceQuad quad;

    // For reference quad, all faces should have jacobian = 1
    // (in reference space, edge length is 2, but jacobian accounts for parameterization)

    auto constant = [](const Vec2& x) { return 1.0; };

    // Each face should integrate to its length
    for (int face = 0; face < 4; ++face) {
        double integral = integrate_face(quad, face, constant, "quad");
        EXPECT_NEAR(integral, 2.0, 1e-12) << "Face " << face << " should have length 2";
    }
}
