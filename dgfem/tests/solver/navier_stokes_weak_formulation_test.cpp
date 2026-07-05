#include <dgfem/boundary/conditions.hpp>
#include <dgfem/core/mesh.hpp>
#include <dgfem/core/space.hpp>
#include <dgfem/kokkos_math.hpp>
#include <dgfem/solver/dg_solver.hpp>
#include <dgfem/solver/weak_form.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>
#include <limits>
#include <stdexcept>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "test_helpers.h"

using namespace dgfem;
using namespace testing;

namespace {

Vec4 make_uniform_conserved(double rho, double u, double v, double p, double gamma) {
    Vec4 primitive{rho, u, v, p};
    return primitive_to_conserved(primitive, gamma);
}

// phi0_value is the order-1 triangle basis's mode-0 (constant) value: 1.0 for
// MonomialBasisTriangle (phi_0 == 1, the default here so existing call sites are unaffected),
// 2.0 for DubinerBasis (phi_0 == 2, see dgfem/src/basis/dubiner.cpp -- confirmed by direct
// evaluation, also noted in HANDOFF.md). Dividing by it is what makes coeffs(0, v) alone
// reproduce a spatially-constant field regardless of which basis is active.
DView2 make_constant_coeffs(int n_basis, const Vec4& conserved_state, double phi0_value = 1.0) {
    DView2 coeffs("coeffs", n_basis, 4);
    for (int v = 0; v < 4; ++v) {
        coeffs(0, v) = conserved_state[v] / phi0_value;
    }
    return coeffs;
}

std::shared_ptr<BoundaryConditionEuler> make_far_field_bc(const Vec4& conserved_state) {
    return std::make_shared<BoundaryConditionEuler>(BCTypeEuler::FAR_FIELD, conserved_state);
}

std::shared_ptr<BoundaryConditionEuler> make_no_slip_bc(double rho, double p) {
    Vec4 primitive{rho, 0.0, 0.0, p};
    return std::make_shared<BoundaryConditionEuler>(BCTypeEuler::NO_SLIP_WALL, primitive);
}

void set_all_boundaries(std::shared_ptr<DGMesh> mesh,
                        const std::shared_ptr<BoundaryConditionEuler>& bc) {
    for (const auto& [name, _] : mesh->get_boundary_tags()) {
        mesh->set_boundary_condition_euler(name, bc);
    }
}

struct ShearFlowSetup {
    std::shared_ptr<DGMesh> mesh;
    std::shared_ptr<DGSpace> space;
    std::vector<DView2> u_coeffs;
};

struct AnalyticViscousFlux {
    double gamma;
    double mu;
    double prandtl;

    std::pair<Vec4, Vec4> operator()(const Vec4& U, const GradU4& grad_U) const {
        constexpr double gas_constant = 1.0;
        double rho = U[0];
        double rho_safe = std::max(rho, 1e-12);

        Vec4 W = conserved_to_primitive(U, gamma);
        double rho_p = W[0];
        double u = W[1];
        double v = W[2];
        double p = W[3];

        Vec2 grad_rho = grad_U[0];
        Vec2 grad_rhou = grad_U[1];
        Vec2 grad_rhov = grad_U[2];
        Vec2 grad_E = grad_U[3];

        double rho_inv = 1.0 / rho_safe;
        Vec2 grad_u = (grad_rhou - u * grad_rho) * rho_inv;
        Vec2 grad_v = (grad_rhov - v * grad_rho) * rho_inv;

        double kinetic_sq = u * u + v * v;
        Vec2 grad_velocity_norm = 2.0 * u * grad_u + 2.0 * v * grad_v;
        (void)grad_velocity_norm;  // Not used explicitly
        Vec2 momentum_term = rho_p * (u * grad_u + v * grad_v);
        Vec2 grad_p = (gamma - 1.0) * (grad_E - 0.5 * kinetic_sq * grad_rho - momentum_term);

        double rho_sq = std::max(rho_p * rho_p, 1e-24);
        double temperature = p / (rho_p * gas_constant);
        (void)temperature;  // Only kept for completeness
        Vec2 grad_T = (rho_p * grad_p - p * grad_rho) * (1.0 / (rho_sq * gas_constant));

        double divergence = grad_u[0] + grad_v[1];
        double lambda = -2.0 * mu / 3.0;

        double tau_xx = 2.0 * mu * grad_u[0] + lambda * divergence;
        double tau_yy = 2.0 * mu * grad_v[1] + lambda * divergence;
        double tau_xy = mu * (grad_u[1] + grad_v[0]);

        double kappa = mu * gamma / (prandtl * (gamma - 1.0));
        double q_x = -kappa * grad_T[0];
        double q_y = -kappa * grad_T[1];

        Vec4 Fv{0.0, 0.0, 0.0, 0.0};
        Vec4 Gv{0.0, 0.0, 0.0, 0.0};

        Fv[1] = tau_xx;
        Fv[2] = tau_xy;
        Fv[3] = u * tau_xx + v * tau_xy + q_x;

        Gv[1] = tau_xy;
        Gv[2] = tau_yy;
        Gv[3] = u * tau_xy + v * tau_yy + q_y;

        return {Fv, Gv};
    }
};

// basis_kind: "monomial" (default) or "dubiner" -- selects both which basis the space uses
// and the matching hardcoded nodal-values-to-modal-coefficients formula in solve_coeffs below.
ShearFlowSetup make_shear_flow_setup(double gamma, std::string_view basis_kind = "monomial") {
    ShearFlowSetup setup;
    setup.mesh = create_test_mesh();
    setup.space = std::make_shared<DGSpace>(setup.mesh->get_element_type(), 1, basis_kind);
    setup.mesh->initialize_dg_space(setup.space, 4);

    int n_basis = setup.space->get_basis()->get_n_basis();
    if (n_basis != 3) {
        throw std::runtime_error("Shear flow helper assumes P1 triangle basis");
    }

    setup.u_coeffs.reserve(setup.mesh->get_n_elements());
    for (int e = 0; e < setup.mesh->get_n_elements(); ++e) {
        setup.u_coeffs.push_back(DView2("u_coeffs", n_basis, 4));
    }

    bool use_dubiner = (basis_kind == "dubiner");
    auto solve_coeffs = [&](std::function<double(const Vec2&)> func, int elem_id) {
        std::array<double, 3> values{};
        for (int v = 0; v < 3; ++v) {
            int global_vertex = setup.mesh->get_elements()(elem_id, v);
            Vec2 x = row2(setup.mesh->get_vertices(), global_vertex);
            values[v] = func(x);
        }
        DView1 coeffs("coeffs", n_basis);
        if (use_dubiner) {
            // Order-1 DubinerBasis closed form (see dgfem/src/basis/dubiner.cpp, and the
            // derivation in AdvectionWeakFormulationTest.MassIntegralDubinerBasis in
            // weak_form_test.cpp): phi_0=2, phi_1=sqrt(6)*(4*xi+2*eta-2),
            // phi_2=2*sqrt(2)*(3*eta-1). Evaluating at the reference triangle's three vertices
            // (0,0),(1,0),(0,1) -- which correspond to values[0],values[1],values[2]
            // respectively, per get_elements()'s local vertex ordering -- and inverting the
            // resulting 3x3 linear system by hand gives this closed-form map from nodal values
            // to modal coefficients (verified by direct numerical evaluation against
            // DubinerBasis before use).
            double s6 = std::sqrt(6.0);
            double s2 = std::sqrt(2.0);
            coeffs[0] = (values[0] + values[1] + values[2]) / 6.0;
            coeffs[1] = (values[1] - values[0]) / (4.0 * s6);
            coeffs[2] = (2.0 * values[2] - values[0] - values[1]) / (12.0 * s2);
        } else {
            // Order-1 MonomialBasisTriangle: phi = {1, eta, xi}, so coefficients are just the
            // nodal values directly (Lagrange-like combination for this specific basis).
            coeffs[0] = values[0];
            coeffs[1] = values[2] - values[0];
            coeffs[2] = values[1] - values[0];
        }
        return coeffs;
    };

    auto rho_func = [](const Vec2&) { return 1.0; };
    auto rho_u_func = [](const Vec2& x) { return x[1]; };
    auto rho_v_func = [](const Vec2&) { return 0.0; };
    auto energy_func = [gamma](const Vec2& x) {
        double u = x[1];
        double v = 0.0;
        double rho = 1.0;
        double p = 1.0;
        double internal = p / (gamma - 1.0);
        double kinetic = 0.5 * rho * (u * u + v * v);
        return internal + kinetic;
    };

    for (int elem = 0; elem < setup.mesh->get_n_elements(); ++elem) {
        DView1 coeff_rho = solve_coeffs(rho_func, elem);
        DView1 coeff_rho_u = solve_coeffs(rho_u_func, elem);
        DView1 coeff_rho_v = solve_coeffs(rho_v_func, elem);
        DView1 coeff_E = solve_coeffs(energy_func, elem);
        for (int i = 0; i < n_basis; ++i) {
            setup.u_coeffs[elem](i, 0) = coeff_rho[i];
            setup.u_coeffs[elem](i, 1) = coeff_rho_u[i];
            setup.u_coeffs[elem](i, 2) = coeff_rho_v[i];
            setup.u_coeffs[elem](i, 3) = coeff_E[i];
        }
    }

    return setup;
}

}  // namespace

TEST(NavierStokesWeakFormulationTest, Construction) {
    EXPECT_NO_THROW(NavierStokesWeakFormulation weak_form(1.4, 1.0e-3, 0.72, 5.0));
}

TEST(NavierStokesWeakFormulationTest, RejectsNonPositiveViscosity) {
    EXPECT_THROW(NavierStokesWeakFormulation(1.4, 0.0, 0.72, 5.0), std::invalid_argument);
    EXPECT_THROW(NavierStokesWeakFormulation(1.4, -1.0e-3, 0.72, 5.0), std::invalid_argument);
}

TEST(NavierStokesWeakFormulationTest, RejectsNonPositivePrandtl) {
    EXPECT_THROW(NavierStokesWeakFormulation(1.4, 1.0e-3, 0.0, 5.0), std::invalid_argument);
    EXPECT_THROW(NavierStokesWeakFormulation(1.4, 1.0e-3, -0.72, 5.0), std::invalid_argument);
}

TEST(NavierStokesWeakFormulationTest, RejectsNonPositivePenaltyPrefactor) {
    EXPECT_THROW(NavierStokesWeakFormulation(1.4, 1.0e-3, 0.72, 0.0), std::invalid_argument);
    EXPECT_THROW(NavierStokesWeakFormulation(1.4, 1.0e-3, 0.72, -5.0), std::invalid_argument);
}

TEST(NavierStokesWeakFormulationTest, PenaltyParameterScaling) {
    const double gamma = 1.4;
    const double mu = 1.0e-2;
    const double sigma0 = 7.5;
    NavierStokesWeakFormulation weak_form(gamma, mu, 0.72, sigma0);

    int p = 2;
    double h = 0.125;
    double expected = sigma0 * mu * (p + 1) * (p + 1) / h;
    EXPECT_DOUBLE_EQ(weak_form.compute_penalty_parameter(p, h), expected);

    // Degenerate h should clamp and scale like (p+1)^2
    double tiny_h = 1e-14;
    double expected_tiny = sigma0 * mu * (p + 1) * (p + 1) / 1e-12;
    EXPECT_DOUBLE_EQ(weak_form.compute_penalty_parameter(p, tiny_h), expected_tiny);
}

TEST(NavierStokesWeakFormulationTest, UniformFlowHasZeroResidual) {
    const double gamma = 1.4;
    auto mesh = create_test_mesh();
    // Explicit "monomial" override: make_constant_coeffs's default phi0_value=1.0 matches
    // MonomialBasisTriangle specifically (phi_0 == 1), independent of whichever basis
    // DGSpace::create_basis picks by default for triangles.
    auto space = std::make_shared<DGSpace>(mesh->get_element_type(), 1, "monomial");
    mesh->initialize_dg_space(space, 4);

    Vec4 uniform_primitive{1.0, 0.5, 0.0, 1.0};
    Vec4 uniform_conserved = primitive_to_conserved(uniform_primitive, gamma);

    auto far_field_bc = make_far_field_bc(uniform_conserved);
    set_all_boundaries(mesh, far_field_bc);

    auto weak_form = std::make_shared<NavierStokesWeakFormulation>(gamma, 1.0e-3, 0.72, 5.0);
    int n_elements = mesh->get_n_elements();
    int n_basis = space->get_basis()->get_n_basis();
    std::vector<DView2> u_coeffs(n_elements);
    for (int elem = 0; elem < n_elements; ++elem) {
        u_coeffs[elem] = make_constant_coeffs(n_basis, uniform_conserved);
    }

    for (int elem = 0; elem < n_elements; ++elem) {
        const auto& elem_data = mesh->get_element_data(elem);
        DView2 R_vol = weak_form->viscous_volume_residual(u_coeffs[elem], elem_data, space);
        for (int i = 0; i < static_cast<int>(R_vol.extent(0)); ++i) {
            for (int j = 0; j < static_cast<int>(R_vol.extent(1)); ++j) {
                EXPECT_NEAR(R_vol(i, j), 0.0, 1e-12);
            }
        }
    }

    // Interior faces
    for (const auto& face : mesh->get_interior_faces()) {
        const auto& face_data_L = mesh->get_element_face_data(face.elem_L, face.face_L);
        const auto& face_data_R = mesh->get_element_face_data(face.elem_R, face.face_R);
        auto [R_L, R_R] = weak_form->viscous_interior_face_residual(
            u_coeffs[face.elem_L], u_coeffs[face.elem_R], face_data_L, face_data_R, space,
            face.permutation);
        for (int i = 0; i < static_cast<int>(R_L.extent(0)); ++i) {
            for (int j = 0; j < static_cast<int>(R_L.extent(1)); ++j) {
                EXPECT_NEAR(R_L(i, j), 0.0, 1e-12);
                EXPECT_NEAR(R_R(i, j), 0.0, 1e-12);
            }
        }
    }

    // Boundary faces
    for (const auto& face : mesh->get_boundary_face_data()) {
        const auto& face_data = mesh->get_element_face_data(face.elem_L, face.face_L);
        auto R_bc = weak_form->viscous_boundary_face_residual(u_coeffs[face.elem_L], face_data,
                                                              face.bc_euler, space);
        for (int i = 0; i < static_cast<int>(R_bc.extent(0)); ++i) {
            for (int j = 0; j < static_cast<int>(R_bc.extent(1)); ++j) {
                EXPECT_NEAR(R_bc(i, j), 0.0, 1e-12);
            }
        }
    }
}

TEST(NavierStokesWeakFormulationTest, UniformFlowHasZeroResidualDubinerBasis) {
    // Same as UniformFlowHasZeroResidual above, but forcing the orthogonal DubinerBasis
    // instead of MonomialBasisTriangle, so both bases get real coverage of this physical
    // invariant (uniform flow produces zero viscous residual everywhere, regardless of basis).
    const double gamma = 1.4;
    auto mesh = create_test_mesh();
    auto space = std::make_shared<DGSpace>(mesh->get_element_type(), 1, "dubiner");
    mesh->initialize_dg_space(space, 4);

    Vec4 uniform_primitive{1.0, 0.5, 0.0, 1.0};
    Vec4 uniform_conserved = primitive_to_conserved(uniform_primitive, gamma);

    auto far_field_bc = make_far_field_bc(uniform_conserved);
    set_all_boundaries(mesh, far_field_bc);

    auto weak_form = std::make_shared<NavierStokesWeakFormulation>(gamma, 1.0e-3, 0.72, 5.0);
    int n_elements = mesh->get_n_elements();
    int n_basis = space->get_basis()->get_n_basis();
    std::vector<DView2> u_coeffs(n_elements);
    for (int elem = 0; elem < n_elements; ++elem) {
        // phi0_value=2.0: DubinerBasis's mode-0 constant value (see make_constant_coeffs above).
        u_coeffs[elem] = make_constant_coeffs(n_basis, uniform_conserved, 2.0);
    }

    for (int elem = 0; elem < n_elements; ++elem) {
        const auto& elem_data = mesh->get_element_data(elem);
        DView2 R_vol = weak_form->viscous_volume_residual(u_coeffs[elem], elem_data, space);
        for (int i = 0; i < static_cast<int>(R_vol.extent(0)); ++i) {
            for (int j = 0; j < static_cast<int>(R_vol.extent(1)); ++j) {
                EXPECT_NEAR(R_vol(i, j), 0.0, 1e-10);
            }
        }
    }

    for (const auto& face : mesh->get_interior_faces()) {
        const auto& face_data_L = mesh->get_element_face_data(face.elem_L, face.face_L);
        const auto& face_data_R = mesh->get_element_face_data(face.elem_R, face.face_R);
        auto [R_L, R_R] = weak_form->viscous_interior_face_residual(
            u_coeffs[face.elem_L], u_coeffs[face.elem_R], face_data_L, face_data_R, space,
            face.permutation);
        for (int i = 0; i < static_cast<int>(R_L.extent(0)); ++i) {
            for (int j = 0; j < static_cast<int>(R_L.extent(1)); ++j) {
                EXPECT_NEAR(R_L(i, j), 0.0, 1e-10);
                EXPECT_NEAR(R_R(i, j), 0.0, 1e-10);
            }
        }
    }

    for (const auto& face : mesh->get_boundary_face_data()) {
        const auto& face_data = mesh->get_element_face_data(face.elem_L, face.face_L);
        auto R_bc = weak_form->viscous_boundary_face_residual(u_coeffs[face.elem_L], face_data,
                                                              face.bc_euler, space);
        for (int i = 0; i < static_cast<int>(R_bc.extent(0)); ++i) {
            for (int j = 0; j < static_cast<int>(R_bc.extent(1)); ++j) {
                EXPECT_NEAR(R_bc(i, j), 0.0, 1e-10);
            }
        }
    }
}

TEST(NavierStokesWeakFormulationTest, ShearFlowVolumeResidualMatchesAnalytic) {
    const double gamma = 1.4;
    const double mu = 5.0e-2;
    const double prandtl = 0.72;
    auto setup = make_shear_flow_setup(gamma);
    auto mesh = setup.mesh;
    auto space = setup.space;
    const auto& u_coeffs = setup.u_coeffs;

    auto mapping = space->get_mapping();
    const auto& vol_quad = space->get_volume_quad();
    int n_quad = vol_quad->size();
    int n_basis = space->get_basis()->get_n_basis();

    auto weak_form = std::make_shared<NavierStokesWeakFormulation>(gamma, mu, prandtl, 5.0);
    for (int elem = 0; elem < mesh->get_n_elements(); ++elem) {
        const auto& elem_data = mesh->get_element_data(elem);
        DView2 computed = weak_form->viscous_volume_residual(u_coeffs[elem], elem_data, space);

        DView2 expected("expected", n_basis, 4);
        const DView2& J_det = elem_data.at("J_det_vol");
        const DView2& dphi_dx = elem_data.at("dphi_dx_vol");
        Vec2 grad_rhou{0.0, 0.0};
        Vec2 grad_E{0.0, 0.0};
        for (int i = 0; i < n_basis; ++i) {
            Vec2 grad_phi_const = row2(dphi_dx, i);
            grad_rhou = grad_rhou + u_coeffs[elem](i, 1) * grad_phi_const;
            grad_E = grad_E + u_coeffs[elem](i, 3) * grad_phi_const;
        }
        DView2 vertices = mesh->get_element_vertices(elem);
        double grad_u_y = grad_rhou[1];
        double grad_E_y = grad_E[1];
        double kappa = mu * gamma / (prandtl * (gamma - 1.0));

        for (int q = 0; q < n_quad; ++q) {
            double w_q = vol_quad->weights[q] * std::abs(J_det(q, 0));
            Vec2 xi = row2(vol_quad->points, q);
            Vec2 x = mapping->map_to_physical(vertices, xi);
            double y = x[1];
            for (int i = 0; i < n_basis; ++i) {
                Vec2 grad_phi = row2(dphi_dx, q * n_basis + i);
                expected(i, 1) -= w_q * mu * grad_phi[1];
                expected(i, 2) -= w_q * mu * grad_phi[0];
                double grad_p_y = (gamma - 1.0) * (grad_E_y - y * grad_u_y);
                double q_y = -kappa * grad_p_y;
                double Gv_energy = mu * y + q_y;
                expected(i, 3) -= w_q * Gv_energy * grad_phi[1];
            }
        }

        for (int i = 0; i < n_basis; ++i) {
            EXPECT_NEAR(computed(i, 1), expected(i, 1), 1e-10);
            EXPECT_NEAR(computed(i, 2), expected(i, 2), 1e-10);
            EXPECT_NEAR(computed(i, 0), 0.0, 1e-12);
            EXPECT_NEAR(computed(i, 3), expected(i, 3), 1e-10);
        }
    }
}

TEST(NavierStokesWeakFormulationTest, ShearFlowVolumeResidualMatchesAnalyticDubinerBasis) {
    // Same as ShearFlowVolumeResidualMatchesAnalytic above, but with make_shear_flow_setup
    // building its modal coefficients via the DubinerBasis-specific hardcoded formula (see
    // solve_coeffs in make_shear_flow_setup) instead of MonomialBasisTriangle's. The
    // residual-matching logic itself is already basis-agnostic (it reads gradients/values out
    // of elem_data, which reflect whichever basis actually produced them) -- the only thing
    // that needed a basis-specific fix was correctly reconstructing the intended shear-flow
    // field from nodal values into modal coefficients in the first place.
    const double gamma = 1.4;
    const double mu = 5.0e-2;
    const double prandtl = 0.72;
    auto setup = make_shear_flow_setup(gamma, "dubiner");
    auto mesh = setup.mesh;
    auto space = setup.space;
    const auto& u_coeffs = setup.u_coeffs;

    auto mapping = space->get_mapping();
    const auto& vol_quad = space->get_volume_quad();
    int n_quad = vol_quad->size();
    int n_basis = space->get_basis()->get_n_basis();

    auto weak_form = std::make_shared<NavierStokesWeakFormulation>(gamma, mu, prandtl, 5.0);
    for (int elem = 0; elem < mesh->get_n_elements(); ++elem) {
        const auto& elem_data = mesh->get_element_data(elem);
        DView2 computed = weak_form->viscous_volume_residual(u_coeffs[elem], elem_data, space);

        DView2 expected("expected", n_basis, 4);
        const DView2& J_det = elem_data.at("J_det_vol");
        const DView2& dphi_dx = elem_data.at("dphi_dx_vol");
        Vec2 grad_rhou{0.0, 0.0};
        Vec2 grad_E{0.0, 0.0};
        for (int i = 0; i < n_basis; ++i) {
            Vec2 grad_phi_const = row2(dphi_dx, i);
            grad_rhou = grad_rhou + u_coeffs[elem](i, 1) * grad_phi_const;
            grad_E = grad_E + u_coeffs[elem](i, 3) * grad_phi_const;
        }
        DView2 vertices = mesh->get_element_vertices(elem);
        double grad_u_y = grad_rhou[1];
        double grad_E_y = grad_E[1];
        double kappa = mu * gamma / (prandtl * (gamma - 1.0));

        for (int q = 0; q < n_quad; ++q) {
            double w_q = vol_quad->weights[q] * std::abs(J_det(q, 0));
            Vec2 xi = row2(vol_quad->points, q);
            Vec2 x = mapping->map_to_physical(vertices, xi);
            double y = x[1];
            for (int i = 0; i < n_basis; ++i) {
                Vec2 grad_phi = row2(dphi_dx, q * n_basis + i);
                expected(i, 1) -= w_q * mu * grad_phi[1];
                expected(i, 2) -= w_q * mu * grad_phi[0];
                double grad_p_y = (gamma - 1.0) * (grad_E_y - y * grad_u_y);
                double q_y = -kappa * grad_p_y;
                double Gv_energy = mu * y + q_y;
                expected(i, 3) -= w_q * Gv_energy * grad_phi[1];
            }
        }

        for (int i = 0; i < n_basis; ++i) {
            EXPECT_NEAR(computed(i, 1), expected(i, 1), 1e-10);
            EXPECT_NEAR(computed(i, 2), expected(i, 2), 1e-10);
            EXPECT_NEAR(computed(i, 0), 0.0, 1e-12);
            EXPECT_NEAR(computed(i, 3), expected(i, 3), 1e-10);
        }
    }
}

TEST(NavierStokesWeakFormulationTest, ShearFlowInteriorFaceResidualMatchesAnalytic) {
    const double gamma = 1.4;
    const double mu = 5.0e-2;
    const double prandtl = 0.72;
    auto setup = make_shear_flow_setup(gamma);
    auto mesh = setup.mesh;
    auto space = setup.space;
    const auto& u_coeffs = setup.u_coeffs;
    auto weak_form = std::make_shared<NavierStokesWeakFormulation>(gamma, mu, prandtl, 5.0);
    const auto& interior_faces = mesh->get_interior_faces();
    ASSERT_FALSE(interior_faces.empty());

    const auto& face = interior_faces.front();
    const auto& face_data_L = mesh->get_element_face_data(face.elem_L, face.face_L);
    const auto& face_data_R = mesh->get_element_face_data(face.elem_R, face.face_R);

    auto [computed_L, computed_R] = weak_form->viscous_interior_face_residual(
        u_coeffs[face.elem_L], u_coeffs[face.elem_R], face_data_L, face_data_R, space,
        face.permutation);

    DView2 expected_L("expected_L", computed_L.extent(0), computed_L.extent(1));
    DView2 expected_R("expected_R", computed_R.extent(0), computed_R.extent(1));
    const DView2& phi_L = face_data_L.at("phi");
    const DView2& phi_R = face_data_R.at("phi");
    const DView2& grad_phi_L = face_data_L.at("dphi_dx_face");
    const DView2& grad_phi_R = face_data_R.at("dphi_dx_face");
    const DView2& weights = face_data_L.at("weights");
    Vec2 normal = to_vec2(face_data_L.at("normal"));
    double face_length = face_data_L.at("length")(0, 0);
    int n_quad = static_cast<int>(weights.extent(0));
    int n_basis_face = static_cast<int>(phi_L.extent(1));
    int n_basis = space->get_basis()->get_n_basis();
    ASSERT_EQ(n_basis_face, n_basis);

    AnalyticViscousFlux flux{gamma, mu, prandtl};

    for (int q = 0; q < n_quad; ++q) {
        double w_q = weights(q, 0) * face_length * 0.5;
        int qR = face.permutation.size() > 0 ? face.permutation[q] : q;

        Vec4 U_L_q{0.0, 0.0, 0.0, 0.0};
        Vec4 U_R_q{0.0, 0.0, 0.0, 0.0};
        for (int i = 0; i < n_basis; ++i) {
            for (int v = 0; v < 4; ++v) {
                U_L_q[v] += phi_L(q, i) * u_coeffs[face.elem_L](i, v);
                U_R_q[v] += phi_R(qR, i) * u_coeffs[face.elem_R](i, v);
            }
        }

        GradU4 grad_UL{Vec2{0.0, 0.0}, Vec2{0.0, 0.0}, Vec2{0.0, 0.0}, Vec2{0.0, 0.0}};
        GradU4 grad_UR{Vec2{0.0, 0.0}, Vec2{0.0, 0.0}, Vec2{0.0, 0.0}, Vec2{0.0, 0.0}};
        for (int i = 0; i < n_basis; ++i) {
            Vec2 grad_phi_i_L = row2(grad_phi_L, q * n_basis + i);
            Vec2 grad_phi_i_R = row2(grad_phi_R, qR * n_basis + i);
            for (int v = 0; v < 4; ++v) {
                grad_UL[v][0] += u_coeffs[face.elem_L](i, v) * grad_phi_i_L[0];
                grad_UL[v][1] += u_coeffs[face.elem_L](i, v) * grad_phi_i_L[1];
                grad_UR[v][0] += u_coeffs[face.elem_R](i, v) * grad_phi_i_R[0];
                grad_UR[v][1] += u_coeffs[face.elem_R](i, v) * grad_phi_i_R[1];
            }
        }

        auto [Fv_L, Gv_L] = flux(U_L_q, grad_UL);
        auto [Fv_R, Gv_R] = flux(U_R_q, grad_UR);

        Vec4 flux_avg = 0.5 * (Fv_L * normal[0] + Gv_L * normal[1]);
        flux_avg = flux_avg + 0.5 * (Fv_R * (-normal[0]) + Gv_R * (-normal[1]));
        Vec4 Fn = flux_avg;

        for (int i = 0; i < n_basis_face; ++i) {
            for (int v = 0; v < 4; ++v) {
                expected_L(i, v) -= w_q * phi_L(q, i) * Fn[v];
                expected_R(i, v) += w_q * phi_R(qR, i) * Fn[v];
            }
        }
    }

    for (int i = 0; i < static_cast<int>(computed_L.extent(0)); ++i) {
        for (int j = 0; j < static_cast<int>(computed_L.extent(1)); ++j) {
            EXPECT_NEAR(computed_L(i, j), expected_L(i, j), 1e-10);
            EXPECT_NEAR(computed_R(i, j), expected_R(i, j), 1e-10);
        }
    }
}

TEST(NavierStokesWeakFormulationTest, ShearFlowInteriorFaceResidualMatchesAnalyticDubinerBasis) {
    // Same as ShearFlowInteriorFaceResidualMatchesAnalytic above, but with make_shear_flow_setup
    // using the DubinerBasis-specific hardcoded coefficient formula -- see the comment on
    // ShearFlowVolumeResidualMatchesAnalyticDubinerBasis above for why this is sufficient.
    const double gamma = 1.4;
    const double mu = 5.0e-2;
    const double prandtl = 0.72;
    auto setup = make_shear_flow_setup(gamma, "dubiner");
    auto mesh = setup.mesh;
    auto space = setup.space;
    const auto& u_coeffs = setup.u_coeffs;
    auto weak_form = std::make_shared<NavierStokesWeakFormulation>(gamma, mu, prandtl, 5.0);
    const auto& interior_faces = mesh->get_interior_faces();
    ASSERT_FALSE(interior_faces.empty());

    const auto& face = interior_faces.front();
    const auto& face_data_L = mesh->get_element_face_data(face.elem_L, face.face_L);
    const auto& face_data_R = mesh->get_element_face_data(face.elem_R, face.face_R);

    auto [computed_L, computed_R] = weak_form->viscous_interior_face_residual(
        u_coeffs[face.elem_L], u_coeffs[face.elem_R], face_data_L, face_data_R, space,
        face.permutation);

    DView2 expected_L("expected_L", computed_L.extent(0), computed_L.extent(1));
    DView2 expected_R("expected_R", computed_R.extent(0), computed_R.extent(1));
    const DView2& phi_L = face_data_L.at("phi");
    const DView2& phi_R = face_data_R.at("phi");
    const DView2& grad_phi_L = face_data_L.at("dphi_dx_face");
    const DView2& grad_phi_R = face_data_R.at("dphi_dx_face");
    const DView2& weights = face_data_L.at("weights");
    Vec2 normal = to_vec2(face_data_L.at("normal"));
    double face_length = face_data_L.at("length")(0, 0);
    int n_quad = static_cast<int>(weights.extent(0));
    int n_basis_face = static_cast<int>(phi_L.extent(1));
    int n_basis = space->get_basis()->get_n_basis();
    ASSERT_EQ(n_basis_face, n_basis);

    AnalyticViscousFlux flux{gamma, mu, prandtl};

    for (int q = 0; q < n_quad; ++q) {
        double w_q = weights(q, 0) * face_length * 0.5;
        int qR = face.permutation.size() > 0 ? face.permutation[q] : q;

        Vec4 U_L_q{0.0, 0.0, 0.0, 0.0};
        Vec4 U_R_q{0.0, 0.0, 0.0, 0.0};
        for (int i = 0; i < n_basis; ++i) {
            for (int v = 0; v < 4; ++v) {
                U_L_q[v] += phi_L(q, i) * u_coeffs[face.elem_L](i, v);
                U_R_q[v] += phi_R(qR, i) * u_coeffs[face.elem_R](i, v);
            }
        }

        GradU4 grad_UL{Vec2{0.0, 0.0}, Vec2{0.0, 0.0}, Vec2{0.0, 0.0}, Vec2{0.0, 0.0}};
        GradU4 grad_UR{Vec2{0.0, 0.0}, Vec2{0.0, 0.0}, Vec2{0.0, 0.0}, Vec2{0.0, 0.0}};
        for (int i = 0; i < n_basis; ++i) {
            Vec2 grad_phi_i_L = row2(grad_phi_L, q * n_basis + i);
            Vec2 grad_phi_i_R = row2(grad_phi_R, qR * n_basis + i);
            for (int v = 0; v < 4; ++v) {
                grad_UL[v][0] += u_coeffs[face.elem_L](i, v) * grad_phi_i_L[0];
                grad_UL[v][1] += u_coeffs[face.elem_L](i, v) * grad_phi_i_L[1];
                grad_UR[v][0] += u_coeffs[face.elem_R](i, v) * grad_phi_i_R[0];
                grad_UR[v][1] += u_coeffs[face.elem_R](i, v) * grad_phi_i_R[1];
            }
        }

        auto [Fv_L, Gv_L] = flux(U_L_q, grad_UL);
        auto [Fv_R, Gv_R] = flux(U_R_q, grad_UR);

        Vec4 flux_avg = 0.5 * (Fv_L * normal[0] + Gv_L * normal[1]);
        flux_avg = flux_avg + 0.5 * (Fv_R * (-normal[0]) + Gv_R * (-normal[1]));
        Vec4 Fn = flux_avg;

        for (int i = 0; i < n_basis_face; ++i) {
            for (int v = 0; v < 4; ++v) {
                expected_L(i, v) -= w_q * phi_L(q, i) * Fn[v];
                expected_R(i, v) += w_q * phi_R(qR, i) * Fn[v];
            }
        }
    }

    for (int i = 0; i < static_cast<int>(computed_L.extent(0)); ++i) {
        for (int j = 0; j < static_cast<int>(computed_L.extent(1)); ++j) {
            EXPECT_NEAR(computed_L(i, j), expected_L(i, j), 1e-10);
            EXPECT_NEAR(computed_R(i, j), expected_R(i, j), 1e-10);
        }
    }
}

TEST(NavierStokesWeakFormulationTest, ShearFlowBoundaryFaceResidualMatchesAnalytic) {
    const double gamma = 1.4;
    const double mu = 5.0e-2;
    const double prandtl = 0.72;
    auto setup = make_shear_flow_setup(gamma);
    auto mesh = setup.mesh;
    auto space = setup.space;
    const auto& u_coeffs = setup.u_coeffs;
    auto shear_bc =
        std::make_shared<BoundaryConditionEuler>(BCTypeEuler::FAR_FIELD, [gamma](const Vec2& x) {
            Vec4 primitive{1.0, x[1], 0.0, 1.0};
            return primitive_to_conserved(primitive, gamma);
        });
    set_all_boundaries(mesh, shear_bc);
    mesh->build_precomputed_faces();
    set_all_boundaries(mesh, shear_bc);

    auto weak_form = std::make_shared<NavierStokesWeakFormulation>(gamma, mu, prandtl, 5.0);
    const auto& boundary_faces = mesh->get_boundary_face_data();
    ASSERT_FALSE(boundary_faces.empty());

    for (const auto& face : boundary_faces) {
        const auto& face_data = mesh->get_element_face_data(face.elem_L, face.face_L);
        auto computed = weak_form->viscous_boundary_face_residual(u_coeffs[face.elem_L], face_data,
                                                                  face.bc_euler, space);

        DView2 expected("expected", computed.extent(0), computed.extent(1));
        const DView2& weights = face_data.at("weights");
        const DView2& phi = face_data.at("phi");
        const DView2& grad_phi = face_data.at("dphi_dx_face");
        Vec2 normal = to_vec2(face_data.at("normal"));
        double face_length = face_data.at("length")(0, 0);
        const DView2& quad_points = face_data.at("quad_points");
        int n_quad = static_cast<int>(weights.extent(0));

        int n_face_basis = static_cast<int>(phi.extent(1));
        int n_basis = space->get_basis()->get_n_basis();
        ASSERT_EQ(n_face_basis, n_basis);
        AnalyticViscousFlux flux{gamma, mu, prandtl};
        double sigma = weak_form->compute_penalty_parameter(space->get_order(), face_length);
        auto bc_ptr = face.bc_euler;

        for (int q = 0; q < n_quad; ++q) {
            double w_q = weights(q, 0) * face_length * 0.5;

            Vec4 U_L_q{0.0, 0.0, 0.0, 0.0};
            for (int i = 0; i < n_basis; ++i) {
                for (int v = 0; v < 4; ++v) {
                    U_L_q[v] += phi(q, i) * u_coeffs[face.elem_L](i, v);
                }
            }

            GradU4 grad_UL{Vec2{0.0, 0.0}, Vec2{0.0, 0.0}, Vec2{0.0, 0.0}, Vec2{0.0, 0.0}};
            for (int i = 0; i < n_basis; ++i) {
                Vec2 grad_phi_i = row2(grad_phi, q * n_basis + i);
                for (int v = 0; v < 4; ++v) {
                    grad_UL[v][0] += u_coeffs[face.elem_L](i, v) * grad_phi_i[0];
                    grad_UL[v][1] += u_coeffs[face.elem_L](i, v) * grad_phi_i[1];
                }
            }

            auto [Fv_L, Gv_L] = flux(U_L_q, grad_UL);
            Vec2 x_q = row2(quad_points, q);
            Vec4 U_bc = U_L_q;
            if (bc_ptr) {
                Vec4 bc_data = bc_ptr->evaluate(x_q);
                switch (bc_ptr->get_type()) {
                case BCTypeEuler::NO_SLIP_WALL:
                    U_bc = primitive_to_conserved(bc_data, gamma);
                    break;
                case BCTypeEuler::FAR_FIELD:
                case BCTypeEuler::SLIP_WALL:
                case BCTypeEuler::PERIODIC:
                default:
                    U_bc = bc_data;
                    break;
                }
            }
            Vec4 penalty = sigma * (U_bc - U_L_q);
            Vec4 Fn = (Fv_L * normal[0] + Gv_L * normal[1]) - penalty;

            for (int i = 0; i < n_face_basis; ++i) {
                for (int v = 0; v < 4; ++v) {
                    expected(i, v) -= w_q * phi(q, i) * Fn[v];
                }
            }
        }

        for (int i = 0; i < static_cast<int>(computed.extent(0)); ++i) {
            for (int j = 0; j < static_cast<int>(computed.extent(1)); ++j) {
                EXPECT_NEAR(computed(i, j), expected(i, j), 1e-10);
            }
        }
    }
}

TEST(NavierStokesWeakFormulationTest, NoSlipBoundaryGeneratesResidual) {
    const double gamma = 1.4;
    auto mesh = create_test_mesh();
    auto space = std::make_shared<DGSpace>(mesh->get_element_type(), 1);
    mesh->initialize_dg_space(space, 4);

    Vec4 uniform_primitive{1.0, 0.5, 0.0, 1.0};
    Vec4 uniform_conserved = primitive_to_conserved(uniform_primitive, gamma);

    // Apply no-slip boundary on all faces
    auto no_slip_bc = make_no_slip_bc(uniform_primitive[0], uniform_primitive[3]);
    set_all_boundaries(mesh, no_slip_bc);

    auto weak_form = std::make_shared<NavierStokesWeakFormulation>(gamma, 1.0e-3, 0.72, 5.0);
    auto assembler = std::make_shared<DGAssembler>(mesh, weak_form);

    int n_elements = mesh->get_n_elements();
    int n_basis = space->get_basis()->get_n_basis();
    std::vector<DView2> u_coeffs(n_elements);
    for (int elem = 0; elem < n_elements; ++elem) {
        u_coeffs[elem] = make_constant_coeffs(n_basis, uniform_conserved);
    }

    std::vector<DView2> residuals;
    assembler->assemble_euler_residual(u_coeffs, residuals);
    double total_norm = 0.0;
    for (const auto& R_elem : residuals) {
        total_norm += frobenius_norm(R_elem);
    }
    EXPECT_GT(total_norm, 1e-6);
}

TEST(NavierStokesWeakFormulationTest, FarFieldBoundaryGeneratesResidual) {
    const double gamma = 1.4;
    auto mesh = create_test_mesh();
    auto space = std::make_shared<DGSpace>(mesh->get_element_type(), 1);
    mesh->initialize_dg_space(space, 4);

    Vec4 interior_primitive{1.0, 0.5, 0.0, 1.0};
    Vec4 interior_conserved = primitive_to_conserved(interior_primitive, gamma);

    // Far-field state deliberately differs from the interior state -- a matching far-field
    // state would just be the zero-residual case already covered by UniformFlowHasZeroResidual.
    Vec4 far_field_primitive{1.2, 0.8, 0.1, 1.1};
    auto far_field_bc = make_far_field_bc(primitive_to_conserved(far_field_primitive, gamma));
    set_all_boundaries(mesh, far_field_bc);

    auto weak_form = std::make_shared<NavierStokesWeakFormulation>(gamma, 1.0e-3, 0.72, 5.0);
    auto assembler = std::make_shared<DGAssembler>(mesh, weak_form);

    int n_elements = mesh->get_n_elements();
    int n_basis = space->get_basis()->get_n_basis();
    std::vector<DView2> u_coeffs(n_elements);
    for (int elem = 0; elem < n_elements; ++elem) {
        u_coeffs[elem] = make_constant_coeffs(n_basis, interior_conserved);
    }

    std::vector<DView2> residuals;
    assembler->assemble_euler_residual(u_coeffs, residuals);
    double total_norm = 0.0;
    for (const auto& R_elem : residuals) {
        total_norm += frobenius_norm(R_elem);
    }
    EXPECT_GT(total_norm, 1e-6);
}

TEST(NavierStokesWeakFormulationTest, SlipWallBoundaryGeneratesResidual) {
    const double gamma = 1.4;
    auto mesh = create_test_mesh();
    auto space = std::make_shared<DGSpace>(mesh->get_element_type(), 1);
    mesh->initialize_dg_space(space, 4);

    Vec4 uniform_primitive{1.0, 0.5, 0.0, 1.0};
    Vec4 uniform_conserved = primitive_to_conserved(uniform_primitive, gamma);

    // Wall ghost state has zero velocity (same intent as the no-slip case above); tagged
    // SLIP_WALL here since the DG boundary residual dispatch treats FAR_FIELD/SLIP_WALL/
    // PERIODIC identically (uses whatever ghost state the BoundaryConditionEuler supplies
    // directly, see navier_stokes_weak_formulation_test.cpp's own analytic reference code
    // a few tests above) -- there's no separate velocity-mirroring formula to test here yet.
    Vec4 wall_primitive{uniform_primitive[0], 0.0, 0.0, uniform_primitive[3]};
    auto slip_wall_bc = std::make_shared<BoundaryConditionEuler>(
        BCTypeEuler::SLIP_WALL, primitive_to_conserved(wall_primitive, gamma));
    set_all_boundaries(mesh, slip_wall_bc);

    auto weak_form = std::make_shared<NavierStokesWeakFormulation>(gamma, 1.0e-3, 0.72, 5.0);
    auto assembler = std::make_shared<DGAssembler>(mesh, weak_form);

    int n_elements = mesh->get_n_elements();
    int n_basis = space->get_basis()->get_n_basis();
    std::vector<DView2> u_coeffs(n_elements);
    for (int elem = 0; elem < n_elements; ++elem) {
        u_coeffs[elem] = make_constant_coeffs(n_basis, uniform_conserved);
    }

    std::vector<DView2> residuals;
    assembler->assemble_euler_residual(u_coeffs, residuals);
    double total_norm = 0.0;
    for (const auto& R_elem : residuals) {
        total_norm += frobenius_norm(R_elem);
    }
    EXPECT_GT(total_norm, 1e-6);
}

TEST(NavierStokesWeakFormulationTest, NavierStokesSolverConstructs) {
    auto mesh = create_test_mesh();
    auto space = std::make_shared<DGSpace>(mesh->get_element_type(), 1);
    mesh->initialize_dg_space(space, 4);

    NavierStokesDGSolver solver(mesh, 1.4, 1.0e-3, 0.72, 5.0);
    auto system_matrix = solver.get_system_matrix();
    EXPECT_EQ(system_matrix->getGlobalNumRows(), system_matrix->getGlobalNumCols());
}

TEST(NavierStokesWeakFormulationTest, DetectsDivergenceFromNonFiniteState) {
    auto mesh = create_test_mesh();
    auto space = std::make_shared<DGSpace>(mesh->get_element_type(), 1);
    mesh->initialize_dg_space(space, 4);

    NavierStokesDGSolver solver(mesh, 1.4, 1.0e-3, 0.72, 5.0);

    // Zero density with nonzero momentum drives conserved_to_primitive's rho_inv = 1/0 = Inf
    // (or, after L2 projection, a near-zero density that blows up within a step or two) --
    // that should propagate through the residual and get caught by the divergence detector
    // (all_finite(), see compressible_solver_base.cpp) rather than corrupt the solution
    // silently or throw partway through the run.
    auto bad_ic = [](const Vec2&) -> Vec4 { return Vec4{0.0, 1.0, 0.0, 2.5}; };

    std::vector<DView2> frames;
    EXPECT_NO_THROW(frames = solver.solve(bad_ic, 0.1, 0.01));
    EXPECT_TRUE(solver.has_diverged());
}

TEST(NavierStokesWeakFormulationTest, SolveRejectsNonPositiveDtOrTFinal) {
    auto mesh = create_test_mesh();
    auto space = std::make_shared<DGSpace>(mesh->get_element_type(), 1);
    mesh->initialize_dg_space(space, 4);

    NavierStokesDGSolver solver(mesh, 1.4, 1.0e-3, 0.72, 5.0);
    auto ic = [](const Vec2&) -> Vec4 { return Vec4{1.0, 0.1, 0.0, 2.5}; };

    EXPECT_THROW(solver.solve(ic, 1.0, 0.0), std::invalid_argument);
    EXPECT_THROW(solver.solve(ic, 1.0, -0.01), std::invalid_argument);
    EXPECT_THROW(solver.solve(ic, 0.0, 0.01), std::invalid_argument);
    EXPECT_THROW(solver.solve(ic, -1.0, 0.01), std::invalid_argument);
}

TEST(NavierStokesWeakFormulationTest, EulerVolumeResidualPerformance) {
    const double gamma = 1.4;
    auto mesh = create_test_mesh();
    // Use a moderately high polynomial order and quadrature count to stress the kernel
    constexpr int poly_order = 4;
    constexpr int quad_level = 6;
    auto space = std::make_shared<DGSpace>(mesh->get_element_type(), poly_order);
    mesh->initialize_dg_space(space, quad_level);

    auto weak_form = std::make_shared<EulerWeakFormulation>(gamma);
    const int n_elements = mesh->get_n_elements();
    const int n_basis = space->get_basis()->get_n_basis();
    const int n_vars = weak_form->get_n_vars();

    std::vector<DView2> u_coeffs(n_elements);
    for (int elem = 0; elem < n_elements; ++elem) {
        Vec4 state{1.0 + 0.1 * elem, 0.5, 0.25, 1.0 + 0.05 * elem};
        u_coeffs[elem] = make_constant_coeffs(n_basis, state);
    }

    // Warm up caches, and check the kernel actually produces a sane (finite, non-trivial)
    // result -- a correctness regression that made every call return zero or NaN would
    // otherwise still pass this test on timing alone.
    for (int elem = 0; elem < n_elements; ++elem) {
        const auto& elem_data = mesh->get_element_data(elem);
        DView2 sample = weak_form->volume_residual(u_coeffs[elem], elem_data, space);
        double sample_norm = frobenius_norm(sample);
        ASSERT_TRUE(std::isfinite(sample_norm));
        ASSERT_GT(sample_norm, 0.0);
    }

    const int iterations = 10000;
    double best_time = std::numeric_limits<double>::max();

    for (int iter = 0; iter < iterations; ++iter) {
        const auto start = std::chrono::high_resolution_clock::now();
        for (int elem = 0; elem < n_elements; ++elem) {
            const auto& elem_data = mesh->get_element_data(elem);
            (void)weak_form->volume_residual(u_coeffs[elem], elem_data, space);
        }
        const auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::micro>(end - start).count();
        best_time = std::min(best_time, elapsed);
    }

    const double avg_time_per_elem = best_time / static_cast<double>(n_elements);
    // Guard against non-sensical (negative) timings while keeping the assertion loose enough
    EXPECT_GT(avg_time_per_elem, 0.0);
    // For CI visibility, log the measured time in microseconds per element
    std::cout << "[PERF] Euler volume residual: order=" << poly_order
              << ", best_time_per_elem_us=" << avg_time_per_elem << std::endl;
}

TEST(NavierStokesWeakFormulationTest, EulerInteriorFaceResidualPerformance) {
    const double gamma = 1.4;
    auto mesh = create_test_mesh();
    constexpr int poly_order = 4;
    constexpr int quad_level = 6;
    auto space = std::make_shared<DGSpace>(mesh->get_element_type(), poly_order);
    mesh->initialize_dg_space(space, quad_level);

    auto weak_form = std::make_shared<EulerWeakFormulation>(gamma);
    const auto& interior_faces = mesh->get_interior_faces();
    ASSERT_FALSE(interior_faces.empty());

    const int n_elements = mesh->get_n_elements();
    const int n_basis = space->get_basis()->get_n_basis();

    std::vector<DView2> u_coeffs(n_elements);
    for (int elem = 0; elem < n_elements; ++elem) {
        Vec4 state{1.0 + 0.05 * elem, 0.25 + 0.1 * elem, -0.15 * elem, 1.0 + 0.02 * elem};
        u_coeffs[elem] = make_constant_coeffs(n_basis, state);
    }

    // Warm up caches, and check the kernel actually produces a sane (finite, non-trivial)
    // result -- a correctness regression that made every call return zero or NaN would
    // otherwise still pass this test on timing alone.
    for (const auto& face : interior_faces) {
        const auto& face_data_L = mesh->get_element_face_data(face.elem_L, face.face_L);
        const auto& face_data_R = mesh->get_element_face_data(face.elem_R, face.face_R);
        auto [R_L, R_R] =
            weak_form->interior_face_residual(u_coeffs[face.elem_L], u_coeffs[face.elem_R],
                                              face_data_L, face_data_R, space, face.permutation);
        double norm_L = frobenius_norm(R_L);
        double norm_R = frobenius_norm(R_R);
        ASSERT_TRUE(std::isfinite(norm_L) && std::isfinite(norm_R));
        ASSERT_GT(norm_L + norm_R, 0.0);
    }

    constexpr int iterations = 20;
    double best_time = std::numeric_limits<double>::max();
    volatile double sink = 0.0;

    for (int iter = 0; iter < iterations; ++iter) {
        const auto start = std::chrono::high_resolution_clock::now();
        for (const auto& face : interior_faces) {
            const auto& face_data_L = mesh->get_element_face_data(face.elem_L, face.face_L);
            const auto& face_data_R = mesh->get_element_face_data(face.elem_R, face.face_R);
            auto [R_face_L, R_face_R] = weak_form->interior_face_residual(
                u_coeffs[face.elem_L], u_coeffs[face.elem_R], face_data_L, face_data_R, space,
                face.permutation);
            sink += R_face_L(0, 0) + R_face_R(0, 0);
        }
        const auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::micro>(end - start).count();
        best_time = std::min(best_time, elapsed);
    }

    const double avg_time_per_face = best_time / static_cast<double>(interior_faces.size());
    EXPECT_GT(avg_time_per_face, 0.0);
    std::cout << "[PERF] Euler interior face residual: order=" << poly_order
              << ", best_time_per_face_us=" << avg_time_per_face << ", sink=" << sink << std::endl;
}
