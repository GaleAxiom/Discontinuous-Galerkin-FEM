#include <dgfem/boundary/conditions.hpp>
#include <dgfem/core/mesh.hpp>
#include <dgfem/core/space.hpp>
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

Eigen::Vector4d make_uniform_conserved(double rho, double u, double v, double p, double gamma) {
    Eigen::Vector4d primitive;
    primitive << rho, u, v, p;
    return primitive_to_conserved(primitive, gamma);
}

Eigen::MatrixXd make_constant_coeffs(int n_basis, const Eigen::Vector4d& conserved_state) {
    Eigen::MatrixXd coeffs = Eigen::MatrixXd::Zero(n_basis, conserved_state.size());
    coeffs.row(0) = conserved_state.transpose();
    return coeffs;
}

std::shared_ptr<BoundaryConditionEuler> make_far_field_bc(const Eigen::Vector4d& conserved_state) {
    return std::make_shared<BoundaryConditionEuler>(BCTypeEuler::FAR_FIELD, conserved_state);
}

std::shared_ptr<BoundaryConditionEuler> make_no_slip_bc(double rho, double p) {
    Eigen::Vector4d primitive;
    primitive << rho, 0.0, 0.0, p;
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
    std::vector<Eigen::MatrixXd> u_coeffs;
};

struct AnalyticViscousFlux {
    double gamma;
    double mu;
    double prandtl;

    std::pair<Eigen::Vector4d, Eigen::Vector4d>
    operator()(const Eigen::Vector4d& U, const Eigen::Matrix<double, 4, 2>& grad_U) const {
        constexpr double gas_constant = 1.0;
        double rho = U[0];
        double rho_safe = std::max(rho, 1e-12);

        Eigen::Vector4d W = conserved_to_primitive(U, gamma);
        double rho_p = W[0];
        double u = W[1];
        double v = W[2];
        double p = W[3];

        Eigen::Vector2d grad_rho = grad_U.row(0).transpose();
        Eigen::Vector2d grad_rhou = grad_U.row(1).transpose();
        Eigen::Vector2d grad_rhov = grad_U.row(2).transpose();
        Eigen::Vector2d grad_E = grad_U.row(3).transpose();

        double rho_inv = 1.0 / rho_safe;
        Eigen::Vector2d grad_u = (grad_rhou - u * grad_rho) * rho_inv;
        Eigen::Vector2d grad_v = (grad_rhov - v * grad_rho) * rho_inv;

        double kinetic_sq = u * u + v * v;
        Eigen::Vector2d grad_velocity_norm = 2.0 * u * grad_u + 2.0 * v * grad_v;
        (void)grad_velocity_norm;  // Not used explicitly
        Eigen::Vector2d momentum_term = rho_p * (u * grad_u + v * grad_v);
        Eigen::Vector2d grad_p =
            (gamma - 1.0) * (grad_E - 0.5 * kinetic_sq * grad_rho - momentum_term);

        double rho_sq = std::max(rho_p * rho_p, 1e-24);
        double temperature = p / (rho_p * gas_constant);
        (void)temperature;  // Only kept for completeness
        Eigen::Vector2d grad_T = (rho_p * grad_p - p * grad_rho) / (rho_sq * gas_constant);

        double divergence = grad_u[0] + grad_v[1];
        double lambda = -2.0 * mu / 3.0;

        double tau_xx = 2.0 * mu * grad_u[0] + lambda * divergence;
        double tau_yy = 2.0 * mu * grad_v[1] + lambda * divergence;
        double tau_xy = mu * (grad_u[1] + grad_v[0]);

        double kappa = mu * gamma / (prandtl * (gamma - 1.0));
        double q_x = -kappa * grad_T[0];
        double q_y = -kappa * grad_T[1];

        Eigen::Vector4d Fv = Eigen::Vector4d::Zero();
        Eigen::Vector4d Gv = Eigen::Vector4d::Zero();

        Fv[1] = tau_xx;
        Fv[2] = tau_xy;
        Fv[3] = u * tau_xx + v * tau_xy + q_x;

        Gv[1] = tau_xy;
        Gv[2] = tau_yy;
        Gv[3] = u * tau_xy + v * tau_yy + q_y;

        return {Fv, Gv};
    }
};

ShearFlowSetup make_shear_flow_setup(double gamma) {
    ShearFlowSetup setup;
    setup.mesh = create_test_mesh();
    setup.space = std::make_shared<DGSpace>(setup.mesh->get_element_type(), 1);
    setup.mesh->initialize_dg_space(setup.space, 4);

    int n_basis = setup.space->get_basis()->get_n_basis();
    if (n_basis != 3) {
        throw std::runtime_error("Shear flow helper assumes P1 triangle basis");
    }

    setup.u_coeffs.assign(setup.mesh->get_n_elements(), Eigen::MatrixXd::Zero(n_basis, 4));

    auto solve_coeffs = [&](std::function<double(const Eigen::Vector2d&)> func, int elem_id) {
        std::array<double, 3> values{};
        for (int v = 0; v < 3; ++v) {
            int global_vertex = setup.mesh->get_elements()(elem_id, v);
            Eigen::Vector2d x = setup.mesh->get_vertices().row(global_vertex).transpose();
            values[v] = func(x);
        }
        Eigen::VectorXd coeffs = Eigen::VectorXd::Zero(n_basis);
        coeffs[0] = values[0];
        coeffs[1] = values[2] - values[0];
        coeffs[2] = values[1] - values[0];
        return coeffs;
    };

    auto rho_func = [](const Eigen::Vector2d&) { return 1.0; };
    auto rho_u_func = [](const Eigen::Vector2d& x) { return x[1]; };
    auto rho_v_func = [](const Eigen::Vector2d&) { return 0.0; };
    auto energy_func = [gamma](const Eigen::Vector2d& x) {
        double u = x[1];
        double v = 0.0;
        double rho = 1.0;
        double p = 1.0;
        double internal = p / (gamma - 1.0);
        double kinetic = 0.5 * rho * (u * u + v * v);
        return internal + kinetic;
    };

    for (int elem = 0; elem < setup.mesh->get_n_elements(); ++elem) {
        Eigen::VectorXd coeff_rho = solve_coeffs(rho_func, elem);
        Eigen::VectorXd coeff_rho_u = solve_coeffs(rho_u_func, elem);
        Eigen::VectorXd coeff_rho_v = solve_coeffs(rho_v_func, elem);
        Eigen::VectorXd coeff_E = solve_coeffs(energy_func, elem);
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
    auto space = std::make_shared<DGSpace>(mesh->get_element_type(), 1);
    mesh->initialize_dg_space(space, 4);

    Eigen::Vector4d uniform_primitive;
    uniform_primitive << 1.0, 0.5, 0.0, 1.0;
    Eigen::Vector4d uniform_conserved = primitive_to_conserved(uniform_primitive, gamma);

    auto far_field_bc = make_far_field_bc(uniform_conserved);
    set_all_boundaries(mesh, far_field_bc);

    auto weak_form = std::make_shared<NavierStokesWeakFormulation>(gamma, 1.0e-3, 0.72, 5.0);
    int n_elements = mesh->get_n_elements();
    int n_basis = space->get_basis()->get_n_basis();
    std::vector<Eigen::MatrixXd> u_coeffs(n_elements);
    for (int elem = 0; elem < n_elements; ++elem) {
        u_coeffs[elem] = make_constant_coeffs(n_basis, uniform_conserved);
    }

    for (int elem = 0; elem < n_elements; ++elem) {
        const auto& elem_data = mesh->get_element_data(elem);
        Eigen::MatrixXd R_vol =
            weak_form->viscous_volume_residual(u_coeffs[elem], elem_data, space);
        for (int i = 0; i < R_vol.rows(); ++i) {
            for (int j = 0; j < R_vol.cols(); ++j) {
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
        for (int i = 0; i < R_L.rows(); ++i) {
            for (int j = 0; j < R_L.cols(); ++j) {
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
        for (int i = 0; i < R_bc.rows(); ++i) {
            for (int j = 0; j < R_bc.cols(); ++j) {
                EXPECT_NEAR(R_bc(i, j), 0.0, 1e-12);
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
        Eigen::MatrixXd computed =
            weak_form->viscous_volume_residual(u_coeffs[elem], elem_data, space);

        Eigen::MatrixXd expected = Eigen::MatrixXd::Zero(n_basis, 4);
        const Eigen::VectorXd& J_det = elem_data.at("J_det_vol");
        const Eigen::MatrixXd& dphi_dx = elem_data.at("dphi_dx_vol");
        Eigen::Vector2d grad_rhou = Eigen::Vector2d::Zero();
        Eigen::Vector2d grad_E = Eigen::Vector2d::Zero();
        for (int i = 0; i < n_basis; ++i) {
            Eigen::Vector2d grad_phi_const = dphi_dx.block(i, 0, 1, 2).transpose();
            grad_rhou += u_coeffs[elem](i, 1) * grad_phi_const;
            grad_E += u_coeffs[elem](i, 3) * grad_phi_const;
        }
        int n_elem_nodes = mesh->get_elements().cols();
        Eigen::MatrixXd vertices(n_elem_nodes, 2);
        for (int i = 0; i < n_elem_nodes; ++i) {
            vertices.row(i) = mesh->get_vertices().row(mesh->get_elements()(elem, i));
        }
        double grad_u_y = grad_rhou[1];
        double grad_E_y = grad_E[1];
        double kappa = mu * gamma / (prandtl * (gamma - 1.0));

        for (int q = 0; q < n_quad; ++q) {
            double w_q = vol_quad->weights[q] * std::abs(J_det[q]);
            Eigen::Vector2d xi = vol_quad->points.row(q).transpose();
            Eigen::Vector2d x = mapping->map_to_physical(vertices, xi);
            double y = x[1];
            for (int i = 0; i < n_basis; ++i) {
                Eigen::Vector2d grad_phi = dphi_dx.block(q * n_basis + i, 0, 1, 2).transpose();
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

    Eigen::MatrixXd expected_L = Eigen::MatrixXd::Zero(computed_L.rows(), computed_L.cols());
    Eigen::MatrixXd expected_R = Eigen::MatrixXd::Zero(computed_R.rows(), computed_R.cols());
    const Eigen::MatrixXd& phi_L = face_data_L.at("phi");
    const Eigen::MatrixXd& phi_R = face_data_R.at("phi");
    const Eigen::MatrixXd& grad_phi_L = face_data_L.at("dphi_dx_face");
    const Eigen::MatrixXd& grad_phi_R = face_data_R.at("dphi_dx_face");
    const Eigen::VectorXd weights = face_data_L.at("weights").col(0);
    Eigen::Vector2d normal = face_data_L.at("normal").col(0);
    double face_length = face_data_L.at("length")(0, 0);
    int n_quad = weights.size();
    int n_basis_face = phi_L.cols();
    int n_basis = space->get_basis()->get_n_basis();
    ASSERT_EQ(n_basis_face, n_basis);

    AnalyticViscousFlux flux{gamma, mu, prandtl};

    for (int q = 0; q < n_quad; ++q) {
        double w_q = weights[q] * face_length * 0.5;
        int qR = face.permutation.size() > 0 ? face.permutation[q] : q;

        Eigen::Vector4d U_L_q = Eigen::Vector4d::Zero();
        Eigen::Vector4d U_R_q = Eigen::Vector4d::Zero();
        for (int i = 0; i < n_basis; ++i) {
            U_L_q += phi_L(q, i) * u_coeffs[face.elem_L].row(i).transpose();
            U_R_q += phi_R(qR, i) * u_coeffs[face.elem_R].row(i).transpose();
        }

        Eigen::Matrix<double, 4, 2> grad_UL = Eigen::Matrix<double, 4, 2>::Zero();
        Eigen::Matrix<double, 4, 2> grad_UR = Eigen::Matrix<double, 4, 2>::Zero();
        for (int i = 0; i < n_basis; ++i) {
            Eigen::Vector2d grad_phi_i_L = grad_phi_L.block(q * n_basis + i, 0, 1, 2).transpose();
            Eigen::Vector2d grad_phi_i_R = grad_phi_R.block(qR * n_basis + i, 0, 1, 2).transpose();
            for (int v = 0; v < 4; ++v) {
                grad_UL(v, 0) += u_coeffs[face.elem_L](i, v) * grad_phi_i_L[0];
                grad_UL(v, 1) += u_coeffs[face.elem_L](i, v) * grad_phi_i_L[1];
                grad_UR(v, 0) += u_coeffs[face.elem_R](i, v) * grad_phi_i_R[0];
                grad_UR(v, 1) += u_coeffs[face.elem_R](i, v) * grad_phi_i_R[1];
            }
        }

        auto [Fv_L, Gv_L] = flux(U_L_q, grad_UL);
        auto [Fv_R, Gv_R] = flux(U_R_q, grad_UR);

        Eigen::Vector4d flux_avg = 0.5 * (Fv_L * normal[0] + Gv_L * normal[1]);
        flux_avg += 0.5 * (Fv_R * (-normal[0]) + Gv_R * (-normal[1]));
        Eigen::Vector4d Fn = flux_avg;

        for (int i = 0; i < n_basis_face; ++i) {
            expected_L.row(i) -= w_q * phi_L(q, i) * Fn.transpose();
            expected_R.row(i) += w_q * phi_R(qR, i) * Fn.transpose();
        }
    }

    for (int i = 0; i < computed_L.rows(); ++i) {
        for (int j = 0; j < computed_L.cols(); ++j) {
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
    auto shear_bc = std::make_shared<BoundaryConditionEuler>(
        BCTypeEuler::FAR_FIELD, [gamma](const Eigen::Vector2d& x) {
            Eigen::Vector4d primitive;
            primitive << 1.0, x[1], 0.0, 1.0;
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

        Eigen::MatrixXd expected = Eigen::MatrixXd::Zero(computed.rows(), computed.cols());
        const Eigen::VectorXd weights = face_data.at("weights").col(0);
        const Eigen::MatrixXd& phi = face_data.at("phi");
        const Eigen::MatrixXd& grad_phi = face_data.at("dphi_dx_face");
        Eigen::Vector2d normal = face_data.at("normal").col(0);
        double face_length = face_data.at("length")(0, 0);
        const Eigen::MatrixXd& face_quad_points_flat = face_data.at("quad_points");
        int n_quad = weights.size();
        Eigen::MatrixXd quad_points(n_quad, 2);
        for (int q = 0; q < n_quad; ++q) {
            quad_points(q, 0) = face_quad_points_flat(q, 0);
            quad_points(q, 1) = face_quad_points_flat(q + n_quad, 0);
        }

        int n_face_basis = phi.cols();
        int n_basis = space->get_basis()->get_n_basis();
        ASSERT_EQ(n_face_basis, n_basis);
        AnalyticViscousFlux flux{gamma, mu, prandtl};
        double sigma = weak_form->compute_penalty_parameter(space->get_order(), face_length);
        auto bc_ptr = face.bc_euler;

        for (int q = 0; q < n_quad; ++q) {
            double w_q = weights[q] * face_length * 0.5;

            Eigen::Vector4d U_L_q = Eigen::Vector4d::Zero();
            for (int i = 0; i < n_basis; ++i) {
                U_L_q += phi(q, i) * u_coeffs[face.elem_L].row(i).transpose();
            }

            Eigen::Matrix<double, 4, 2> grad_UL = Eigen::Matrix<double, 4, 2>::Zero();
            for (int i = 0; i < n_basis; ++i) {
                Eigen::Vector2d grad_phi_i = grad_phi.block(q * n_basis + i, 0, 1, 2).transpose();
                for (int v = 0; v < 4; ++v) {
                    grad_UL(v, 0) += u_coeffs[face.elem_L](i, v) * grad_phi_i[0];
                    grad_UL(v, 1) += u_coeffs[face.elem_L](i, v) * grad_phi_i[1];
                }
            }

            auto [Fv_L, Gv_L] = flux(U_L_q, grad_UL);
            Eigen::Vector2d x_q = quad_points.row(q).transpose();
            Eigen::Vector4d U_bc = U_L_q;
            if (bc_ptr) {
                Eigen::Vector4d bc_data = bc_ptr->evaluate(x_q);
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
            Eigen::Vector4d penalty = sigma * (U_bc - U_L_q);
            Eigen::Vector4d Fn = (Fv_L * normal[0] + Gv_L * normal[1]) - penalty;

            for (int i = 0; i < n_face_basis; ++i) {
                expected.row(i) -= w_q * phi(q, i) * Fn.transpose();
            }
        }

        for (int i = 0; i < computed.rows(); ++i) {
            for (int j = 0; j < computed.cols(); ++j) {
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

    Eigen::Vector4d uniform_primitive;
    uniform_primitive << 1.0, 0.5, 0.0, 1.0;
    Eigen::Vector4d uniform_conserved = primitive_to_conserved(uniform_primitive, gamma);

    // Apply no-slip boundary on all faces
    auto no_slip_bc = make_no_slip_bc(uniform_primitive[0], uniform_primitive[3]);
    set_all_boundaries(mesh, no_slip_bc);

    auto weak_form = std::make_shared<NavierStokesWeakFormulation>(gamma, 1.0e-3, 0.72, 5.0);
    auto assembler = std::make_shared<DGAssembler>(mesh, weak_form);

    int n_elements = mesh->get_n_elements();
    int n_basis = space->get_basis()->get_n_basis();
    std::vector<Eigen::MatrixXd> u_coeffs(n_elements);
    for (int elem = 0; elem < n_elements; ++elem) {
        u_coeffs[elem] = make_constant_coeffs(n_basis, uniform_conserved);
    }

    std::vector<Eigen::MatrixXd> residuals;
    assembler->assemble_euler_residual(u_coeffs, residuals);
    double total_norm = 0.0;
    for (const auto& R_elem : residuals) {
        total_norm += R_elem.norm();
    }
    EXPECT_GT(total_norm, 1e-6);
}

TEST(NavierStokesWeakFormulationTest, NavierStokesSolverConstructs) {
    auto mesh = create_test_mesh();
    auto space = std::make_shared<DGSpace>(mesh->get_element_type(), 1);
    mesh->initialize_dg_space(space, 4);

    NavierStokesDGSolver solver(mesh, 1.4, 1.0e-3, 0.72, 5.0);
    const auto& system_matrix = solver.get_system_matrix();
    EXPECT_EQ(system_matrix.rows(), system_matrix.cols());
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

    std::vector<Eigen::MatrixXd> u_coeffs(n_elements);
    for (int elem = 0; elem < n_elements; ++elem) {
        Eigen::Vector4d state;
        state << 1.0 + 0.1 * elem, 0.5, 0.25, 1.0 + 0.05 * elem;
        u_coeffs[elem] = make_constant_coeffs(n_basis, state);
    }

    // Warm up caches
    for (int elem = 0; elem < n_elements; ++elem) {
        const auto& elem_data = mesh->get_element_data(elem);
        (void)weak_form->volume_residual(u_coeffs[elem], elem_data, space);
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

    std::vector<Eigen::MatrixXd> u_coeffs(n_elements);
    for (int elem = 0; elem < n_elements; ++elem) {
        Eigen::Vector4d state;
        state << 1.0 + 0.05 * elem, 0.25 + 0.1 * elem, -0.15 * elem, 1.0 + 0.02 * elem;
        u_coeffs[elem] = make_constant_coeffs(n_basis, state);
    }

    for (const auto& face : interior_faces) {
        const auto& face_data_L = mesh->get_element_face_data(face.elem_L, face.face_L);
        const auto& face_data_R = mesh->get_element_face_data(face.elem_R, face.face_R);
        (void)weak_form->interior_face_residual(u_coeffs[face.elem_L], u_coeffs[face.elem_R],
                                                face_data_L, face_data_R, space, face.permutation);
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
