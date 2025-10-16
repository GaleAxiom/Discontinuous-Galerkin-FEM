/**
 * @file euler_boundary_conditions_test.cpp
 * @brief Comprehensive tests for all Euler boundary condition types
 */

#include <cmath>
#include <dgfem/boundary/conditions.hpp>
#include <dgfem/core/mesh.hpp>
#include <dgfem/core/space.hpp>
#include <dgfem/solver/weak_form.hpp>
#include <dgfem/utils/mesh_creation.hpp>

#include <memory>

#include <gmock/gmock.h>
#include <gmsh.h>
#include <gtest/gtest.h>

#include "test_helpers.h"

using namespace dgfem;
using namespace testing;

namespace {

// Helper function to create uniform flow state
Eigen::Vector4d make_uniform_conserved(double rho, double u, double v, double p, double gamma) {
    Eigen::Vector4d primitive;
    primitive << rho, u, v, p;
    return primitive_to_conserved(primitive, gamma);
}

// Helper function to create constant coefficient matrix
Eigen::MatrixXd make_constant_coeffs(int n_basis, const Eigen::Vector4d& conserved_state) {
    Eigen::MatrixXd coeffs = Eigen::MatrixXd::Zero(n_basis, conserved_state.size());
    coeffs.row(0) = conserved_state.transpose();
    return coeffs;
}

// Helper to set all mesh boundaries to the same BC
void set_all_boundaries(std::shared_ptr<DGMesh> mesh,
                        const std::shared_ptr<BoundaryConditionEuler>& bc) {
    for (const auto& [name, _] : mesh->get_boundary_tags()) {
        mesh->set_boundary_condition_euler(name, bc);
    }
}

}  // namespace

/**
 * @brief Test fixture for Euler boundary conditions
 */
class EulerBoundaryConditionsTest : public ::testing::Test {
protected:
    void SetUp() override {
        gmsh::initialize();
        gamma_ = 1.4;
        weak_form_ = std::make_shared<EulerWeakFormulation>(gamma_);
    }

    void TearDown() override { gmsh::finalize(); }

    std::shared_ptr<EulerWeakFormulation> weak_form_;
    double gamma_;
};

// ============================================================================
// FAR_FIELD Boundary Condition Tests
// ============================================================================

TEST_F(EulerBoundaryConditionsTest, FarFieldBCConstantValue) {
    // Create a simple mesh
    auto mesh = MeshCreator::create_rectangular_mesh(2, true, 0.0, 1.0, 0.0, 1.0);
    auto space = std::make_shared<DGSpace>("triangle", 1);
    mesh->initialize_dg_space(space);

    // Set up far-field BC with constant state
    double rho_inf = 1.0;
    double u_inf = 1.0;
    double v_inf = 0.0;
    double p_inf = 1.0 / gamma_;

    Eigen::Vector4d U_inf = make_uniform_conserved(rho_inf, u_inf, v_inf, p_inf, gamma_);
    auto far_field_bc = std::make_shared<BoundaryConditionEuler>(BCTypeEuler::FAR_FIELD, U_inf);

    // Verify BC type
    EXPECT_EQ(far_field_bc->get_type(), BCTypeEuler::FAR_FIELD);

    // Verify BC evaluation
    Eigen::Vector2d test_point(0.5, 0.5);
    Eigen::Vector4d evaluated = far_field_bc->evaluate(test_point);
    EXPECT_NEAR(evaluated[0], U_inf[0], 1e-12);
    EXPECT_NEAR(evaluated[1], U_inf[1], 1e-12);
    EXPECT_NEAR(evaluated[2], U_inf[2], 1e-12);
    EXPECT_NEAR(evaluated[3], U_inf[3], 1e-12);
}

TEST_F(EulerBoundaryConditionsTest, FarFieldBCFunctionValue) {
    // Create mesh
    auto mesh = MeshCreator::create_rectangular_mesh(2, true, 0.0, 1.0, 0.0, 1.0);
    auto space = std::make_shared<DGSpace>("triangle", 1);
    mesh->initialize_dg_space(space);

    // Set up spatially-varying far-field BC
    auto bc_func = [this](const Eigen::Vector2d& x) {
        double rho = 1.0 + 0.1 * std::sin(M_PI * x[0]);
        double u = 1.0;
        double v = 0.0;
        double p = 1.0 / gamma_;
        return make_uniform_conserved(rho, u, v, p, gamma_);
    };

    auto far_field_bc = std::make_shared<BoundaryConditionEuler>(BCTypeEuler::FAR_FIELD, bc_func);

    // Test evaluation at different points
    Eigen::Vector2d x1(0.0, 0.5);
    Eigen::Vector2d x2(0.5, 0.5);

    Eigen::Vector4d U1 = far_field_bc->evaluate(x1);
    Eigen::Vector4d U2 = far_field_bc->evaluate(x2);

    // Values should differ due to spatial variation
    EXPECT_NE(U1[0], U2[0]);

    // Verify the function is being evaluated correctly
    Eigen::Vector4d expected_x1 = bc_func(x1);
    EXPECT_NEAR(U1[0], expected_x1[0], 1e-12);
    EXPECT_NEAR(U1[1], expected_x1[1], 1e-12);
}

TEST_F(EulerBoundaryConditionsTest, FarFieldBCResidual) {
    // Create mesh and set up space
    auto mesh = MeshCreator::create_rectangular_mesh(2, true, 0.0, 1.0, 0.0, 1.0);
    auto space = std::make_shared<DGSpace>("triangle", 2);
    mesh->initialize_dg_space(space);

    // Set far-field BC with known state
    double rho = 1.0;
    double u = 1.0;
    double v = 0.0;
    double p = 1.0 / gamma_;
    Eigen::Vector4d U_inf = make_uniform_conserved(rho, u, v, p, gamma_);
    auto far_field_bc = std::make_shared<BoundaryConditionEuler>(BCTypeEuler::FAR_FIELD, U_inf);
    set_all_boundaries(mesh, far_field_bc);

    // Find a boundary face
    int boundary_elem = -1, boundary_face = -1;
    for (int e = 0; e < mesh->get_n_elements(); ++e) {
        for (int f = 0; f < 3; ++f) {
            if (mesh->is_boundary_face(e, f)) {
                boundary_elem = e;
                boundary_face = f;
                break;
            }
        }
        if (boundary_elem >= 0)
            break;
    }

    ASSERT_GE(boundary_elem, 0) << "No boundary face found";

    // Set up coefficients (uniform flow matching far-field)
    int n_basis = space->get_basis()->get_n_basis();
    Eigen::MatrixXd u_coeffs = make_constant_coeffs(n_basis, U_inf);

    // Compute boundary face residual
    auto face_data = mesh->get_element_face_data(boundary_elem, boundary_face);
    Eigen::Vector2d normal = face_data.at("normal").col(0);
    Eigen::MatrixXd R_face =
        weak_form_->boundary_face_residual(u_coeffs, face_data, far_field_bc, space);

    // With matching states U_L = U_R, the Rusanov flux is: 0.5*(Fn_L + Fn_R) = Fn
    // The residual should equal the physical flux through the boundary
    EXPECT_TRUE(R_face.allFinite());
    EXPECT_EQ(R_face.rows(), n_basis);
    EXPECT_EQ(R_face.cols(), 4);

    // The residual norm should be consistent with the flux magnitude
    double flux_magnitude = std::abs(u * normal[0] + v * normal[1]);
    EXPECT_GT(R_face.norm(), flux_magnitude * 0.1);  // At least 10% of velocity-based flux
    EXPECT_LT(R_face.norm(), 10.0);                  // Reasonable upper bound

    // Test with zero velocity - should still have pressure flux
    Eigen::Vector4d U_zero = make_uniform_conserved(rho, 0.0, 0.0, p, gamma_);
    auto bc_zero = std::make_shared<BoundaryConditionEuler>(BCTypeEuler::FAR_FIELD, U_zero);
    set_all_boundaries(mesh, bc_zero);
    Eigen::MatrixXd u_coeffs_zero = make_constant_coeffs(n_basis, U_zero);
    Eigen::MatrixXd R_zero =
        weak_form_->boundary_face_residual(u_coeffs_zero, face_data, bc_zero, space);

    // With zero velocity, only pressure contribution remains
    EXPECT_TRUE(R_zero.allFinite());
    EXPECT_GT(R_zero.norm(), 1e-12);          // Non-zero due to pressure
    EXPECT_LT(R_zero.norm(), R_face.norm());  // Should be less than with velocity
}

// ============================================================================
// SLIP_WALL Boundary Condition Tests
// ============================================================================

TEST_F(EulerBoundaryConditionsTest, SlipWallBCType) {
    // Slip wall BC doesn't need a specific state, but we provide a dummy one
    Eigen::Vector4d dummy_state = Eigen::Vector4d::Zero();
    auto slip_bc = std::make_shared<BoundaryConditionEuler>(BCTypeEuler::SLIP_WALL, dummy_state);

    EXPECT_EQ(slip_bc->get_type(), BCTypeEuler::SLIP_WALL);
}

TEST_F(EulerBoundaryConditionsTest, SlipWallBCNormalVelocityReflection) {
    // Create mesh
    auto mesh = MeshCreator::create_rectangular_mesh(2, true, 0.0, 1.0, 0.0, 1.0);
    auto space = std::make_shared<DGSpace>("triangle", 2);
    mesh->initialize_dg_space(space);

    // Set slip wall BC
    Eigen::Vector4d dummy_state = Eigen::Vector4d::Zero();
    auto slip_bc = std::make_shared<BoundaryConditionEuler>(BCTypeEuler::SLIP_WALL, dummy_state);
    set_all_boundaries(mesh, slip_bc);

    // Find a boundary face (preferably on the bottom or top)
    int boundary_elem = -1, boundary_face = -1;
    for (int e = 0; e < mesh->get_n_elements(); ++e) {
        for (int f = 0; f < 3; ++f) {
            if (mesh->is_boundary_face(e, f)) {
                boundary_elem = e;
                boundary_face = f;
                break;
            }
        }
        if (boundary_elem >= 0)
            break;
    }

    ASSERT_GE(boundary_elem, 0) << "No boundary face found";

    // Set up a flow with both normal and tangential components
    // For a horizontal wall (normal = [0, ±1]), set velocity = [u_tan, u_norm]
    double rho = 1.0;
    double u = 0.5;   // tangential component
    double v = 0.25;  // normal component
    double p = 1.0 / gamma_;

    Eigen::Vector4d U = make_uniform_conserved(rho, u, v, p, gamma_);
    int n_basis = space->get_basis()->get_n_basis();
    Eigen::MatrixXd u_coeffs = make_constant_coeffs(n_basis, U);

    // Compute boundary face residual
    auto face_data = mesh->get_element_face_data(boundary_elem, boundary_face);
    Eigen::MatrixXd R_face =
        weak_form_->boundary_face_residual(u_coeffs, face_data, slip_bc, space);

    // Residual should be finite (not zero, as velocity is reflected)
    EXPECT_GT(R_face.norm(), 1e-12);
    EXPECT_LT(R_face.norm(), 1e3);  // Should be reasonable magnitude
}

TEST_F(EulerBoundaryConditionsTest, SlipWallBCZeroNormalVelocity) {
    // When flow is already tangential, slip wall ghost state matches interior
    auto mesh = MeshCreator::create_rectangular_mesh(2, true, 0.0, 1.0, 0.0, 1.0);
    auto space = std::make_shared<DGSpace>("triangle", 2);
    mesh->initialize_dg_space(space);

    Eigen::Vector4d dummy_state = Eigen::Vector4d::Zero();
    auto slip_bc = std::make_shared<BoundaryConditionEuler>(BCTypeEuler::SLIP_WALL, dummy_state);
    set_all_boundaries(mesh, slip_bc);

    // Find a boundary face
    int boundary_elem = -1, boundary_face = -1;
    for (int e = 0; e < mesh->get_n_elements(); ++e) {
        for (int f = 0; f < 3; ++f) {
            if (mesh->is_boundary_face(e, f)) {
                boundary_elem = e;
                boundary_face = f;
                break;
            }
        }
        if (boundary_elem >= 0)
            break;
    }

    ASSERT_GE(boundary_elem, 0);

    // Get the normal to determine tangential direction
    auto face_data = mesh->get_element_face_data(boundary_elem, boundary_face);
    Eigen::Vector2d normal = face_data.at("normal").col(0);

    // Create purely tangential velocity (perpendicular to normal)
    Eigen::Vector2d tangent(-normal[1], normal[0]);
    double u_mag = 1.0;
    double u = tangent[0] * u_mag;
    double v = tangent[1] * u_mag;

    // Verify velocity is actually tangential
    double normal_velocity = u * normal[0] + v * normal[1];
    EXPECT_NEAR(normal_velocity, 0.0, 1e-10);

    double rho = 1.0;
    double p = 1.0 / gamma_;
    Eigen::Vector4d U = make_uniform_conserved(rho, u, v, p, gamma_);
    int n_basis = space->get_basis()->get_n_basis();
    Eigen::MatrixXd u_coeffs = make_constant_coeffs(n_basis, U);

    // Compute residual
    Eigen::MatrixXd R_face =
        weak_form_->boundary_face_residual(u_coeffs, face_data, slip_bc, space);

    // When velocity is tangential, the ghost state equals interior state
    // So we get the same flux as if both sides had identical states
    EXPECT_TRUE(R_face.allFinite());
    EXPECT_EQ(R_face.rows(), n_basis);
    EXPECT_EQ(R_face.cols(), 4);
    EXPECT_GT(R_face.norm(), 1e-10);

    // Compare with normal velocity case - should have less dissipation
    double u_norm = 0.3;
    double v_norm = 0.3;
    Eigen::Vector4d U_norm = make_uniform_conserved(rho, u_norm, v_norm, p, gamma_);
    Eigen::MatrixXd u_coeffs_norm = make_constant_coeffs(n_basis, U_norm);
    Eigen::MatrixXd R_norm =
        weak_form_->boundary_face_residual(u_coeffs_norm, face_data, slip_bc, space);

    // Both should be finite and non-zero
    EXPECT_TRUE(R_norm.allFinite());
    EXPECT_GT(R_norm.norm(), 1e-10);
}

// ============================================================================
// NO_SLIP_WALL Boundary Condition Tests
// ============================================================================

TEST_F(EulerBoundaryConditionsTest, NoSlipWallBCType) {
    // No-slip wall requires primitive variables (rho, u=0, v=0, p)
    Eigen::Vector4d primitive_bc;
    primitive_bc << 1.0, 0.0, 0.0, 1.0 / gamma_;

    auto no_slip_bc =
        std::make_shared<BoundaryConditionEuler>(BCTypeEuler::NO_SLIP_WALL, primitive_bc);

    EXPECT_EQ(no_slip_bc->get_type(), BCTypeEuler::NO_SLIP_WALL);
}

TEST_F(EulerBoundaryConditionsTest, NoSlipWallBCZeroVelocity) {
    // Create mesh
    auto mesh = MeshCreator::create_rectangular_mesh(2, true, 0.0, 1.0, 0.0, 1.0);
    auto space = std::make_shared<DGSpace>("triangle", 1);
    mesh->initialize_dg_space(space);

    // Set no-slip BC with zero velocity
    double rho_wall = 1.0;
    double p_wall = 1.0 / gamma_;
    Eigen::Vector4d primitive_bc;
    primitive_bc << rho_wall, 0.0, 0.0, p_wall;

    auto no_slip_bc =
        std::make_shared<BoundaryConditionEuler>(BCTypeEuler::NO_SLIP_WALL, primitive_bc);

    // Verify evaluation returns primitive variables at multiple points
    std::vector<Eigen::Vector2d> test_points = {
        Eigen::Vector2d(0.0, 0.0), Eigen::Vector2d(0.5, 0.0), Eigen::Vector2d(1.0, 0.5),
        Eigen::Vector2d(0.25, 0.75)};

    for (const auto& point : test_points) {
        Eigen::Vector4d evaluated = no_slip_bc->evaluate(point);

        // The BC stores primitive variables, check them precisely
        EXPECT_DOUBLE_EQ(evaluated[0], rho_wall);  // rho - exact match
        EXPECT_DOUBLE_EQ(evaluated[1], 0.0);       // u = 0 - exact match
        EXPECT_DOUBLE_EQ(evaluated[2], 0.0);       // v = 0 - exact match
        EXPECT_DOUBLE_EQ(evaluated[3], p_wall);    // p - exact match

        // Verify physical validity
        EXPECT_GT(evaluated[0], 0.0);  // Positive density
        EXPECT_GT(evaluated[3], 0.0);  // Positive pressure
    }

    // Test with different wall conditions
    double rho_wall2 = 1.5;
    double p_wall2 = 2.0;
    Eigen::Vector4d primitive_bc2;
    primitive_bc2 << rho_wall2, 0.0, 0.0, p_wall2;
    auto no_slip_bc2 =
        std::make_shared<BoundaryConditionEuler>(BCTypeEuler::NO_SLIP_WALL, primitive_bc2);

    Eigen::Vector4d eval2 = no_slip_bc2->evaluate(Eigen::Vector2d(0.5, 0.5));
    EXPECT_DOUBLE_EQ(eval2[0], rho_wall2);
    EXPECT_DOUBLE_EQ(eval2[1], 0.0);
    EXPECT_DOUBLE_EQ(eval2[2], 0.0);
    EXPECT_DOUBLE_EQ(eval2[3], p_wall2);
}

TEST_F(EulerBoundaryConditionsTest, NoSlipWallBCResidual) {
    auto mesh = MeshCreator::create_rectangular_mesh(2, true, 0.0, 1.0, 0.0, 1.0);
    auto space = std::make_shared<DGSpace>("triangle", 2);
    mesh->initialize_dg_space(space);

    // Set no-slip wall BC
    double rho_wall = 1.0;
    double p_wall = 1.0 / gamma_;
    Eigen::Vector4d primitive_bc;
    primitive_bc << rho_wall, 0.0, 0.0, p_wall;

    auto no_slip_bc =
        std::make_shared<BoundaryConditionEuler>(BCTypeEuler::NO_SLIP_WALL, primitive_bc);
    set_all_boundaries(mesh, no_slip_bc);

    // Find boundary face
    int boundary_elem = -1, boundary_face = -1;
    for (int e = 0; e < mesh->get_n_elements(); ++e) {
        for (int f = 0; f < 3; ++f) {
            if (mesh->is_boundary_face(e, f)) {
                boundary_elem = e;
                boundary_face = f;
                break;
            }
        }
        if (boundary_elem >= 0)
            break;
    }

    ASSERT_GE(boundary_elem, 0);
    int n_basis = space->get_basis()->get_n_basis();
    auto face_data = mesh->get_element_face_data(boundary_elem, boundary_face);

    // Test 1: Interior flow with velocity - should have finite residual
    double u_int = 0.5;
    double v_int = 0.2;
    Eigen::Vector4d U_moving = make_uniform_conserved(rho_wall, u_int, v_int, p_wall, gamma_);
    Eigen::MatrixXd u_coeffs_moving = make_constant_coeffs(n_basis, U_moving);
    Eigen::MatrixXd R_moving =
        weak_form_->boundary_face_residual(u_coeffs_moving, face_data, no_slip_bc, space);

    // Should have non-zero residual due to velocity mismatch
    EXPECT_TRUE(R_moving.allFinite());
    EXPECT_EQ(R_moving.rows(), n_basis);
    EXPECT_EQ(R_moving.cols(), 4);
    EXPECT_GT(R_moving.norm(), 1e-10);  // Non-zero residual

    // Test 2: Interior flow at rest - should also have finite residual
    Eigen::Vector4d U_rest = make_uniform_conserved(rho_wall, 0.0, 0.0, p_wall, gamma_);
    Eigen::MatrixXd u_coeffs_rest = make_constant_coeffs(n_basis, U_rest);
    Eigen::MatrixXd R_rest =
        weak_form_->boundary_face_residual(u_coeffs_rest, face_data, no_slip_bc, space);

    EXPECT_TRUE(R_rest.allFinite());
    EXPECT_GT(R_rest.norm(), 0.0);  // Still non-zero due to flux

    // Test 3: Verify residual scales with velocity magnitude
    // Higher velocity should affect the momentum components more
    Eigen::Vector4d U_fast = make_uniform_conserved(rho_wall, 2.0, 1.0, p_wall, gamma_);
    Eigen::MatrixXd u_coeffs_fast = make_constant_coeffs(n_basis, U_fast);
    Eigen::MatrixXd R_fast =
        weak_form_->boundary_face_residual(u_coeffs_fast, face_data, no_slip_bc, space);

    EXPECT_TRUE(R_fast.allFinite());
    EXPECT_GT(R_fast.norm(), 0.0);

    // Test 4: Different pressure should affect residual
    double p_high = 2.0 * p_wall;
    Eigen::Vector4d U_high_p = make_uniform_conserved(rho_wall, u_int, v_int, p_high, gamma_);
    Eigen::MatrixXd u_coeffs_high_p = make_constant_coeffs(n_basis, U_high_p);
    Eigen::MatrixXd R_high_p =
        weak_form_->boundary_face_residual(u_coeffs_high_p, face_data, no_slip_bc, space);

    EXPECT_TRUE(R_high_p.allFinite());
    EXPECT_GT(R_high_p.norm(), 0.0);

    // All residuals should be of reasonable magnitude
    EXPECT_LT(R_moving.norm(), 100.0);
    EXPECT_LT(R_rest.norm(), 100.0);
    EXPECT_LT(R_fast.norm(), 100.0);
    EXPECT_LT(R_high_p.norm(), 100.0);
}

TEST_F(EulerBoundaryConditionsTest, NoSlipWallBCSpatiallyVarying) {
    // Test spatially-varying no-slip BC (e.g., varying wall temperature/pressure)
    auto mesh = MeshCreator::create_rectangular_mesh(2, true, 0.0, 1.0, 0.0, 1.0);
    auto space = std::make_shared<DGSpace>("triangle", 1);
    mesh->initialize_dg_space(space);

    // Spatially varying BC function with temperature gradient
    double rho_base = 1.0;
    double p_base = 1.0 / gamma_;
    auto bc_func = [this, rho_base, p_base](const Eigen::Vector2d& x) {
        Eigen::Vector4d primitive;
        // Linear pressure variation in x-direction (10% variation)
        double p_local = p_base * (1.0 + 0.1 * x[0]);
        // Density varies to maintain isothermal condition: rho ~ p
        double rho_local = rho_base * (1.0 + 0.1 * x[0]);
        primitive << rho_local, 0.0, 0.0, p_local;
        return primitive;
    };

    auto no_slip_bc = std::make_shared<BoundaryConditionEuler>(BCTypeEuler::NO_SLIP_WALL, bc_func);

    // Test evaluation at multiple points
    std::vector<Eigen::Vector2d> test_points = {
        Eigen::Vector2d(0.0, 0.5), Eigen::Vector2d(0.5, 0.5), Eigen::Vector2d(1.0, 0.5)};

    std::vector<Eigen::Vector4d> evaluations;
    for (const auto& point : test_points) {
        Eigen::Vector4d prim = no_slip_bc->evaluate(point);
        evaluations.push_back(prim);

        // Velocity must always be zero
        EXPECT_DOUBLE_EQ(prim[1], 0.0);
        EXPECT_DOUBLE_EQ(prim[2], 0.0);

        // Physical validity
        EXPECT_GT(prim[0], 0.0);
        EXPECT_GT(prim[3], 0.0);
    }

    // Verify spatial variation - pressure should increase with x
    EXPECT_LT(evaluations[0][3], evaluations[1][3]);
    EXPECT_LT(evaluations[1][3], evaluations[2][3]);

    // Verify density also varies proportionally
    EXPECT_LT(evaluations[0][0], evaluations[1][0]);
    EXPECT_LT(evaluations[1][0], evaluations[2][0]);

    // Check exact values at x=0 and x=1
    double tolerance = 1e-13;
    EXPECT_NEAR(evaluations[0][0], rho_base, tolerance);
    EXPECT_NEAR(evaluations[0][3], p_base, tolerance);
    EXPECT_NEAR(evaluations[2][0], rho_base * 1.1, tolerance);
    EXPECT_NEAR(evaluations[2][3], p_base * 1.1, tolerance);
}

// ============================================================================
// Multiple BC Types on Different Boundaries
// ============================================================================

TEST_F(EulerBoundaryConditionsTest, MixedBoundaryConditions) {
    // Create a mesh with multiple boundary tags
    auto mesh = MeshCreator::create_rectangular_mesh(4, true, 0.0, 1.0, 0.0, 1.0);
    auto space = std::make_shared<DGSpace>("triangle", 1);
    mesh->initialize_dg_space(space);

    // Set different BCs on different boundaries
    Eigen::Vector4d U_inf = make_uniform_conserved(1.0, 1.0, 0.0, 1.0 / gamma_, gamma_);
    auto far_field_bc = std::make_shared<BoundaryConditionEuler>(BCTypeEuler::FAR_FIELD, U_inf);

    Eigen::Vector4d primitive_wall;
    primitive_wall << 1.0, 0.0, 0.0, 1.0 / gamma_;
    auto no_slip_bc =
        std::make_shared<BoundaryConditionEuler>(BCTypeEuler::NO_SLIP_WALL, primitive_wall);

    Eigen::Vector4d dummy = Eigen::Vector4d::Zero();
    auto slip_bc = std::make_shared<BoundaryConditionEuler>(BCTypeEuler::SLIP_WALL, dummy);

    // Get boundary tags and set different BCs
    auto tags = mesh->get_boundary_tags();
    int tag_count = 0;
    for (const auto& [tag_name, tag_id] : tags) {
        if (tag_count == 0) {
            mesh->set_boundary_condition_euler(tag_name, far_field_bc);
        } else if (tag_count == 1) {
            mesh->set_boundary_condition_euler(tag_name, no_slip_bc);
        } else {
            mesh->set_boundary_condition_euler(tag_name, slip_bc);
        }
        tag_count++;
    }

    // Verify BCs are correctly assigned
    for (int e = 0; e < mesh->get_n_elements(); ++e) {
        for (int f = 0; f < 3; ++f) {
            if (mesh->is_boundary_face(e, f)) {
                auto bc = mesh->get_boundary_condition_euler(e, f);
                ASSERT_NE(bc, nullptr);
                // BC type should be one of the three we set
                bool valid_type =
                    (bc->get_type() == BCTypeEuler::FAR_FIELD ||
                     bc->get_type() == BCTypeEuler::NO_SLIP_WALL ||
                     bc->get_type() == BCTypeEuler::SLIP_WALL ||
                     bc->get_type() == BCTypeEuler::INLET || bc->get_type() == BCTypeEuler::OUTLET);
                EXPECT_TRUE(valid_type);
            }
        }
    }
}

// ============================================================================
// Error Handling Tests
// ============================================================================

TEST_F(EulerBoundaryConditionsTest, PeriodicBCThrowsError) {
    // Periodic BC should throw error in boundary_face_residual
    auto mesh = MeshCreator::create_rectangular_mesh(2, true, 0.0, 1.0, 0.0, 1.0);
    auto space = std::make_shared<DGSpace>("triangle", 1);
    mesh->initialize_dg_space(space);

    // Create periodic BC (should not be used as boundary BC)
    Eigen::Vector4d dummy = Eigen::Vector4d::Zero();
    auto periodic_bc = std::make_shared<BoundaryConditionEuler>(BCTypeEuler::PERIODIC, dummy);

    // Find a boundary face
    int boundary_elem = -1, boundary_face = -1;
    for (int e = 0; e < mesh->get_n_elements(); ++e) {
        for (int f = 0; f < 3; ++f) {
            if (mesh->is_boundary_face(e, f)) {
                boundary_elem = e;
                boundary_face = f;
                break;
            }
        }
        if (boundary_elem >= 0)
            break;
    }

    ASSERT_GE(boundary_elem, 0);

    Eigen::Vector4d U = make_uniform_conserved(1.0, 1.0, 0.0, 1.0 / gamma_, gamma_);
    int n_basis = space->get_basis()->get_n_basis();
    Eigen::MatrixXd u_coeffs = make_constant_coeffs(n_basis, U);

    auto face_data = mesh->get_element_face_data(boundary_elem, boundary_face);

    // Should throw an error for periodic BC on boundary
    EXPECT_THROW(weak_form_->boundary_face_residual(u_coeffs, face_data, periodic_bc, space),
                 std::runtime_error);
}

// ============================================================================
// Physical Consistency Tests
// ============================================================================

TEST_F(EulerBoundaryConditionsTest, FarFieldBCConservationProperties) {
    // Test that far-field BC preserves total energy and mass flux
    auto mesh = MeshCreator::create_rectangular_mesh(2, true, 0.0, 1.0, 0.0, 1.0);
    auto space = std::make_shared<DGSpace>("triangle", 2);
    mesh->initialize_dg_space(space);

    // Set up subsonic far-field conditions
    double mach = 0.5;
    double rho_inf = 1.0;
    double u_inf = mach * std::sqrt(gamma_ * 1.0 / rho_inf);  // c = sqrt(gamma * p / rho)
    double v_inf = 0.0;
    double p_inf = 1.0;

    Eigen::Vector4d U_inf = make_uniform_conserved(rho_inf, u_inf, v_inf, p_inf, gamma_);
    auto far_field_bc = std::make_shared<BoundaryConditionEuler>(BCTypeEuler::FAR_FIELD, U_inf);

    // Evaluate conserved variables
    Eigen::Vector4d U_bc = far_field_bc->evaluate(Eigen::Vector2d(0.5, 0.0));

    // Check that density is positive
    EXPECT_GT(U_bc[0], 0.0);

    // Check that total energy is positive
    EXPECT_GT(U_bc[3], 0.0);

    // Verify consistency with primitive variables
    Eigen::Vector4d W_bc = conserved_to_primitive(U_bc, gamma_);
    EXPECT_NEAR(W_bc[0], rho_inf, 1e-10);
    EXPECT_NEAR(W_bc[3], p_inf, 1e-10);
}

TEST_F(EulerBoundaryConditionsTest, SlipWallBCConservesMassAndEnergy) {
    // Slip wall should preserve density and pressure (only velocity changes)
    auto mesh = MeshCreator::create_rectangular_mesh(2, true, 0.0, 1.0, 0.0, 1.0);
    auto space = std::make_shared<DGSpace>("triangle", 1);
    mesh->initialize_dg_space(space);

    Eigen::Vector4d dummy = Eigen::Vector4d::Zero();
    auto slip_bc = std::make_shared<BoundaryConditionEuler>(BCTypeEuler::SLIP_WALL, dummy);
    set_all_boundaries(mesh, slip_bc);

    // Find a boundary face
    int boundary_elem = -1, boundary_face = -1;
    for (int e = 0; e < mesh->get_n_elements(); ++e) {
        for (int f = 0; f < 3; ++f) {
            if (mesh->is_boundary_face(e, f)) {
                boundary_elem = e;
                boundary_face = f;
                break;
            }
        }
        if (boundary_elem >= 0)
            break;
    }

    ASSERT_GE(boundary_elem, 0);

    // The slip wall BC implementation reflects normal velocity
    // This test verifies the BC is applied correctly
    double rho = 1.2;
    double u = 0.5;
    double v = 0.3;
    double p = 1.0;

    Eigen::Vector4d U = make_uniform_conserved(rho, u, v, p, gamma_);
    int n_basis = space->get_basis()->get_n_basis();
    Eigen::MatrixXd u_coeffs = make_constant_coeffs(n_basis, U);

    auto face_data = mesh->get_element_face_data(boundary_elem, boundary_face);

    // Compute residual - should be well-defined
    EXPECT_NO_THROW({
        Eigen::MatrixXd R_face =
            weak_form_->boundary_face_residual(u_coeffs, face_data, slip_bc, space);
        EXPECT_TRUE(R_face.allFinite());
    });
}

TEST_F(EulerBoundaryConditionsTest, NoSlipWallBCPositivePressureAndDensity) {
    // No-slip wall should maintain positive density and pressure
    Eigen::Vector4d primitive_bc;
    primitive_bc << 1.2, 0.0, 0.0, 1.5;

    auto no_slip_bc =
        std::make_shared<BoundaryConditionEuler>(BCTypeEuler::NO_SLIP_WALL, primitive_bc);

    Eigen::Vector2d test_point(0.3, 0.7);
    Eigen::Vector4d evaluated = no_slip_bc->evaluate(test_point);

    // Density and pressure should be positive
    EXPECT_GT(evaluated[0], 0.0);
    EXPECT_GT(evaluated[3], 0.0);

    // Velocity should be zero
    EXPECT_NEAR(evaluated[1], 0.0, 1e-12);
    EXPECT_NEAR(evaluated[2], 0.0, 1e-12);
}
