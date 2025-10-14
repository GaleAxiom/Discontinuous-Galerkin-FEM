/**
 * @file quad_test.cpp
 * @brief Tests for quadrilateral element implementation
 */

#include "dgfem/basis/legendre.hpp"
#include "dgfem/boundary/conditions.hpp"
#include "dgfem/core/mesh.hpp"
#include "dgfem/core/space.hpp"
#include "dgfem/reference/elements.hpp"
#include "dgfem/reference/mapping.hpp"
#include "dgfem/solver/dg_solver.hpp"
#include "dgfem/solver/weak_form.hpp"
#include "dgfem/utils/mesh_creation.hpp"

#include <Eigen/Dense>
#include <cmath>

#include <functional>

#include <gtest/gtest.h>

using namespace dgfem;

class QuadElementTest : public ::testing::Test {
protected:
    void SetUp() override {
        ref_quad = std::make_shared<ReferenceQuad>();
        mapping = std::make_shared<GeometricMapping>(ref_quad);
    }

    std::shared_ptr<ReferenceQuad> ref_quad;
    std::shared_ptr<GeometricMapping> mapping;
};

TEST_F(QuadElementTest, ReferenceVertices) {
    const auto& vertices = ref_quad->get_vertices();

    EXPECT_EQ(vertices.rows(), 4);
    EXPECT_EQ(vertices.cols(), 2);

    // Check vertices are at corners of [-1,1]^2
    EXPECT_NEAR(vertices(0, 0), -1.0, 1e-10);
    EXPECT_NEAR(vertices(0, 1), -1.0, 1e-10);
    EXPECT_NEAR(vertices(1, 0), 1.0, 1e-10);
    EXPECT_NEAR(vertices(1, 1), -1.0, 1e-10);
    EXPECT_NEAR(vertices(2, 0), 1.0, 1e-10);
    EXPECT_NEAR(vertices(2, 1), 1.0, 1e-10);
    EXPECT_NEAR(vertices(3, 0), -1.0, 1e-10);
    EXPECT_NEAR(vertices(3, 1), 1.0, 1e-10);
}

TEST_F(QuadElementTest, EdgeVertices) {
    auto [v0, v1] = ref_quad->edge_vertices(0);
    EXPECT_EQ(v0, 0);
    EXPECT_EQ(v1, 1);

    auto [v2, v3] = ref_quad->edge_vertices(1);
    EXPECT_EQ(v2, 1);
    EXPECT_EQ(v3, 2);

    auto [v4, v5] = ref_quad->edge_vertices(2);
    EXPECT_EQ(v4, 2);
    EXPECT_EQ(v5, 3);

    auto [v6, v7] = ref_quad->edge_vertices(3);
    EXPECT_EQ(v6, 3);
    EXPECT_EQ(v7, 0);
}

TEST_F(QuadElementTest, ContainsPoint) {
    EXPECT_TRUE(ref_quad->contains_point(Eigen::Vector2d(0.0, 0.0)));
    EXPECT_TRUE(ref_quad->contains_point(Eigen::Vector2d(1.0, 1.0)));
    EXPECT_TRUE(ref_quad->contains_point(Eigen::Vector2d(-1.0, -1.0)));
    EXPECT_TRUE(ref_quad->contains_point(Eigen::Vector2d(0.5, -0.5)));

    EXPECT_FALSE(ref_quad->contains_point(Eigen::Vector2d(1.1, 0.0)));
    EXPECT_FALSE(ref_quad->contains_point(Eigen::Vector2d(0.0, -1.1)));
}

TEST_F(QuadElementTest, BilinearMapping) {
    // Create a simple unit square [0,1]x[0,1]
    Eigen::MatrixXd vertices(4, 2);
    vertices << 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0;

    // Test mapping at center of reference element
    Eigen::Vector2d xi(0.0, 0.0);
    auto data = mapping->compute_mapping(vertices, xi);

    // Physical point should be at (0.5, 0.5)
    EXPECT_NEAR(data.x_phys[0], 0.5, 1e-10);
    EXPECT_NEAR(data.x_phys[1], 0.5, 1e-10);

    // Jacobian determinant should be 0.25
    EXPECT_NEAR(data.J_T_det, 0.25, 1e-10);

    // dxi/dx should be diagonal(2, 2)
    EXPECT_NEAR(data.dxi_dx(0, 0), 2.0, 1e-10);
    EXPECT_NEAR(data.dxi_dx(1, 1), 2.0, 1e-10);
    EXPECT_NEAR(std::abs(data.dxi_dx(0, 1)), 0.0, 1e-10);
    EXPECT_NEAR(std::abs(data.dxi_dx(1, 0)), 0.0, 1e-10);
}

TEST_F(QuadElementTest, MappingAtCorners) {
    // Create a unit square [0,1]x[0,1]
    Eigen::MatrixXd vertices(4, 2);
    vertices << 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0;

    // Test all four corners
    std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> corner_tests = {
        {Eigen::Vector2d(-1.0, -1.0), Eigen::Vector2d(0.0, 0.0)},
        {Eigen::Vector2d(1.0, -1.0), Eigen::Vector2d(1.0, 0.0)},
        {Eigen::Vector2d(1.0, 1.0), Eigen::Vector2d(1.0, 1.0)},
        {Eigen::Vector2d(-1.0, 1.0), Eigen::Vector2d(0.0, 1.0)}};

    for (const auto& [xi, expected] : corner_tests) {
        auto data = mapping->compute_mapping(vertices, xi);
        EXPECT_NEAR(data.x_phys[0], expected[0], 1e-10);
        EXPECT_NEAR(data.x_phys[1], expected[1], 1e-10);
    }
}

class QuadLegendreBasisTest : public ::testing::Test {
protected:
    void SetUp() override {
        basis_p1 = std::make_shared<LegendreBasis>(1);
        basis_p2 = std::make_shared<LegendreBasis>(2);
    }

    std::shared_ptr<LegendreBasis> basis_p1;
    std::shared_ptr<LegendreBasis> basis_p2;
};

TEST_F(QuadLegendreBasisTest, NumberOfBasisFunctions) {
    EXPECT_EQ(basis_p1->get_n_basis(), 4);  // (1+1)^2
    EXPECT_EQ(basis_p2->get_n_basis(), 9);  // (2+1)^2
}

TEST_F(QuadLegendreBasisTest, EvaluateAtCenter) {
    Eigen::Vector2d xi(0.0, 0.0);
    Eigen::VectorXd phi = basis_p2->evaluate(xi);

    EXPECT_EQ(phi.size(), 9);

    // P_0(0) = 1, P_1(0) = 0, P_2(0) = -0.5
    // phi[idx] = P_i(0) * P_j(0) where idx = j*3 + i
    EXPECT_NEAR(phi[0], 1.0, 1e-10);   // P_0(0) * P_0(0)
    EXPECT_NEAR(phi[1], 0.0, 1e-10);   // P_1(0) * P_0(0)
    EXPECT_NEAR(phi[2], -0.5, 1e-10);  // P_2(0) * P_0(0)
    EXPECT_NEAR(phi[3], 0.0, 1e-10);   // P_0(0) * P_1(0)
    EXPECT_NEAR(phi[4], 0.0, 1e-10);   // P_1(0) * P_1(0)
    EXPECT_NEAR(phi[5], 0.0, 1e-10);   // P_2(0) * P_1(0)
    EXPECT_NEAR(phi[6], -0.5, 1e-10);  // P_0(0) * P_2(0)
    EXPECT_NEAR(phi[7], 0.0, 1e-10);   // P_1(0) * P_2(0)
    EXPECT_NEAR(phi[8], 0.25, 1e-10);  // P_2(0) * P_2(0)
}

TEST_F(QuadLegendreBasisTest, GradientAtCenter) {
    Eigen::Vector2d xi(0.0, 0.0);
    Eigen::MatrixXd grad = basis_p2->evaluate_gradient(xi);

    EXPECT_EQ(grad.rows(), 9);
    EXPECT_EQ(grad.cols(), 2);

    // P_0'(0) = 0, P_1'(0) = 1, P_2'(0) = 0
    // grad[idx] = (dP_i/dxi * P_j, P_i * dP_j/deta)

    // phi[0] = P_0(xi) * P_0(eta), grad = (0, 0)
    EXPECT_NEAR(grad(0, 0), 0.0, 1e-10);
    EXPECT_NEAR(grad(0, 1), 0.0, 1e-10);

    // phi[1] = P_1(xi) * P_0(eta), grad = (1*1, 0*0) = (1, 0)
    EXPECT_NEAR(grad(1, 0), 1.0, 1e-10);
    EXPECT_NEAR(grad(1, 1), 0.0, 1e-10);

    // phi[3] = P_0(xi) * P_1(eta), grad = (0*0, 1*1) = (0, 1)
    EXPECT_NEAR(grad(3, 0), 0.0, 1e-10);
    EXPECT_NEAR(grad(3, 1), 1.0, 1e-10);
}

class QuadDGSpaceTest : public ::testing::Test {
protected:
    void SetUp() override { dg_space = std::make_shared<DGSpace>("quad", 2); }

    std::shared_ptr<DGSpace> dg_space;
};

TEST_F(QuadDGSpaceTest, Initialization) {
    EXPECT_EQ(dg_space->get_order(), 2);
    EXPECT_EQ(dg_space->get_basis()->get_n_basis(), 9);
    EXPECT_EQ(dg_space->get_ref_element()->get_n_edges(), 4);
}

TEST_F(QuadDGSpaceTest, FaceQuadratureMapping) {
    const auto& face_quad = dg_space->get_face_quad();

    // For each face, check that quadrature points map to the correct edge
    std::vector<std::pair<int, std::function<bool(const Eigen::Vector2d&)>>> face_checks = {
        {0,
         [](const Eigen::Vector2d& xi) {
             return std::abs(xi[1] + 1.0) < 1e-10;
         }},  // Bottom: y = -1
        {1,
         [](const Eigen::Vector2d& xi) { return std::abs(xi[0] - 1.0) < 1e-10; }},  // Right: x = 1
        {2, [](const Eigen::Vector2d& xi) { return std::abs(xi[1] - 1.0) < 1e-10; }},  // Top: y = 1
        {3,
         [](const Eigen::Vector2d& xi) { return std::abs(xi[0] + 1.0) < 1e-10; }}  // Left: x = -1
    };

    for (const auto& [face_id, check_func] : face_checks) {
        for (int q = 0; q < face_quad->size(); ++q) {
            double s = face_quad->points(q, 0);
            Eigen::Vector2d xi = dg_space->map_face_quad_point(face_id, s);
            EXPECT_TRUE(check_func(xi)) << "Face " << face_id << ", quad point " << q
                                        << " mapped to (" << xi[0] << ", " << xi[1] << ")";
        }
    }
}

TEST_F(QuadDGSpaceTest, FaceNormals) {
    MeshCreator::initialize_gmsh();

    // Create a simple mesh with quads
    auto mesh = MeshCreator::create_rectangular_mesh(0.5, false, 0.0, 1.0, 0.0, 1.0);
    mesh->initialize_dg_space(dg_space, 1);

    // Check face normals for first element
    // Get the actual vertices to determine expected normals
    const auto& elem_verts = mesh->get_elements().row(0);
    Eigen::MatrixXd vertices(4, 2);
    for (int i = 0; i < 4; ++i) {
        vertices.row(i) = mesh->get_vertices().row(elem_verts(i));
    }

    for (int face = 0; face < 4; ++face) {
        const auto& face_data = mesh->get_face_data(0, face);
        Eigen::Vector2d normal = face_data.at("normal").head<2>();
        double length = face_data.at("length")[0];

        // Normal should be unit length
        EXPECT_NEAR(normal.norm(), 1.0, 1e-6) << "Face " << face;

        // Face length should be 0.5 for this mesh
        EXPECT_NEAR(length, 0.5, 1e-2) << "Face " << face;
    }

    MeshCreator::finalize_gmsh();
}

class QuadVolumeIntegralTest : public ::testing::Test {
protected:
    void SetUp() override {
        MeshCreator::initialize_gmsh();

        // Create a single quad element
        mesh = MeshCreator::create_rectangular_mesh(10.0, false, 0.0, 1.0, 0.0, 1.0);
        dg_space = std::make_shared<DGSpace>("quad", 2);
        mesh->initialize_dg_space(dg_space, 1);
    }

    void TearDown() override { MeshCreator::finalize_gmsh(); }

    std::shared_ptr<DGMesh> mesh;
    std::shared_ptr<DGSpace> dg_space;
};

TEST_F(QuadVolumeIntegralTest, StiffnessMatrix) {
    const auto& elem_data = mesh->get_element_data(0);

    LaplaceWeakFormulation weak_form(10.0);
    Eigen::MatrixXd K = weak_form.compute_volume_integral(elem_data, dg_space);

    EXPECT_EQ(K.rows(), 9);
    EXPECT_EQ(K.cols(), 9);

    // K[0,0] should be 0 (constant function has zero gradient)
    EXPECT_NEAR(K(0, 0), 0.0, 1e-10);

    // K[0,1] should be 0 (orthogonality)
    EXPECT_NEAR(K(0, 1), 0.0, 1e-10);

    // K[1,1] should be 4.0 for unit square with order 2
    // phi_1 = P_1(xi)*P_0(eta) = xi
    // grad_phi_1 in reference = (1, 0)
    // Transform to physical: (2, 0) since dxi/dx = 2
    // Integral of (2,0)·(2,0) * 0.25 over [-1,1]^2 = 4 * 0.25 * 2 * 2 = 4
    EXPECT_NEAR(K(1, 1), 4.0, 1e-10);

    // Matrix should be symmetric
    for (int i = 0; i < 9; ++i) {
        for (int j = 0; j < 9; ++j) {
            EXPECT_NEAR(K(i, j), K(j, i), 1e-10)
                << "K(" << i << "," << j << ") != K(" << j << "," << i << ")";
        }
    }

    // Matrix should be positive semi-definite (all eigenvalues >= 0)
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(K);
    for (int i = 0; i < 9; ++i) {
        EXPECT_GE(eigensolver.eigenvalues()[i], -1e-10) << "Eigenvalue " << i << " is negative";
    }
}

class QuadConvergenceTest : public ::testing::Test {
protected:
    void SetUp() override { MeshCreator::initialize_gmsh(); }

    void TearDown() override { MeshCreator::finalize_gmsh(); }
};

TEST_F(QuadConvergenceTest, PolynomialSolution) {
    // Test with a polynomial solution that quads should resolve exactly
    // u = x(1-x)y(1-y)
    // -Δu = 2y(1-y) + 2x(1-x)

    auto source = [](const Eigen::Vector2d& x) -> double {
        return 2.0 * x[1] * (1.0 - x[1]) + 2.0 * x[0] * (1.0 - x[0]);
    };

    auto exact = [](const Eigen::Vector2d& x) -> double {
        return x[0] * (1.0 - x[0]) * x[1] * (1.0 - x[1]);
    };

    auto exact_grad = [](const Eigen::Vector2d& x) -> Eigen::Vector2d {
        Eigen::Vector2d grad;
        grad[0] = (1.0 - 2.0 * x[0]) * x[1] * (1.0 - x[1]);
        grad[1] = x[0] * (1.0 - x[0]) * (1.0 - 2.0 * x[1]);
        return grad;
    };

    // Test with a fine mesh
    auto mesh = MeshCreator::create_rectangular_mesh(0.1, false, 0.0, 1.0, 0.0, 1.0);
    auto dg_space = std::make_shared<DGSpace>("quad", 2);
    mesh->initialize_dg_space(dg_space, 1);

    // Set boundary conditions
    auto bc_zero = dgfem::make_dirichlet_bc(0.0);
    mesh->set_boundary_condition("Bottom", bc_zero);
    mesh->set_boundary_condition("Top", bc_zero);
    mesh->set_boundary_condition("Left", bc_zero);
    mesh->set_boundary_condition("Right", bc_zero);

    // Solve
    LaplaceDGSolver solver(mesh, 10.0);
    Eigen::VectorXd solution = solver.solve(source);

    // Compute error
    auto errors = solver.compute_error(exact, exact_grad);

    // For order 2, error should be small on a fine mesh
    EXPECT_LT(errors["L2"], 0.01) << "L2 error too large for polynomial solution";
    EXPECT_LT(errors["H1"], 0.1) << "H1 error too large for polynomial solution";
}
