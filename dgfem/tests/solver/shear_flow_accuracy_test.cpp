#include <Eigen/Dense>
#include <cmath>
#include <dgfem/boundary/conditions.hpp>
#include <dgfem/core/space.hpp>
#include <dgfem/solver/dg_solver.hpp>
#include <dgfem/solver/weak_form.hpp>
#include <dgfem/utils/mesh_creation.hpp>

#include <array>
#include <memory>

#include <gtest/gtest.h>

using namespace dgfem;

namespace {

struct ShearFlowAnalytic {
    double gamma;

    [[nodiscard]] Eigen::Vector4d primitive(const Eigen::Vector2d& x) const {
        Eigen::Vector4d W;
        W << 1.0, x[1], 0.0, 1.0;
        return W;
    }

    [[nodiscard]] Eigen::Vector4d conserved(const Eigen::Vector2d& x) const {
        return primitive_to_conserved(primitive(x), gamma);
    }
};

[[nodiscard]] double compute_variable_error(const std::shared_ptr<DGMesh>& mesh,
                                            const Eigen::MatrixXd& numerical_sol,
                                            const ShearFlowAnalytic& exact, double gamma,
                                            int var_idx) {
    auto space = mesh->get_dg_space();
    auto mapping = space->get_mapping();
    const auto& phi = space->get_volume_basis_values();
    const auto& quad_pts = space->get_volume_quad()->points;
    const auto& quad_wts = space->get_volume_quad()->weights;

    int n_basis = space->get_basis()->get_n_basis();
    double error_sq = 0.0;
    double norm_sq = 0.0;

    for (int elem = 0; elem < mesh->get_n_elements(); ++elem) {
        const auto& elem_data = mesh->get_element_data(elem);
        const Eigen::VectorXd& J_det = elem_data.at("J_det_vol");

        Eigen::MatrixXd vertices(mesh->get_elements().cols(), 2);
        for (int i = 0; i < vertices.rows(); ++i) {
            vertices.row(i) = mesh->get_vertices().row(mesh->get_elements()(elem, i));
        }

        for (int q = 0; q < quad_wts.size(); ++q) {
            Eigen::Vector2d x_phys = mapping->map_to_physical(vertices, quad_pts.row(q));

            Eigen::Vector4d U_num = Eigen::Vector4d::Zero();
            for (int i = 0; i < n_basis; ++i) {
                for (int v = 0; v < 4; ++v) {
                    U_num[v] += numerical_sol(elem, i * 4 + v) * phi(q, i);
                }
            }

            Eigen::Vector4d W_num = conserved_to_primitive(U_num, gamma);
            Eigen::Vector4d W_exact = exact.primitive(x_phys);

            double weight = quad_wts(q) * std::abs(J_det(q));
            double diff = W_num[var_idx] - W_exact[var_idx];
            error_sq += diff * diff * weight;
            norm_sq += W_exact[var_idx] * W_exact[var_idx] * weight;
        }
    }

    if (norm_sq < 1e-14) {
        return std::sqrt(error_sq);
    }
    return std::sqrt(error_sq / norm_sq);
}

}  // namespace

class NavierStokesShearFlowAccuracyTest : public ::testing::Test {
protected:
    void SetUp() override { MeshCreator::initialize_gmsh(); }

    void TearDown() override { MeshCreator::finalize_gmsh(); }
};

TEST_F(NavierStokesShearFlowAccuracyTest, MaintainsManufacturedShearProfile) {
    constexpr double gamma = 1.4;
    constexpr double mu = 5.0e-6;
    constexpr double prandtl = 0.72;
    constexpr double penalty = 10000.0;

    constexpr double dx = 0.05;
    constexpr int order = 2;

    auto mesh = MeshCreator::create_rectangular_mesh(dx, false, 0.0, 1.0, 0.0, 1.0);
    auto space = std::make_shared<DGSpace>(mesh->get_element_type(), order);
    mesh->initialize_dg_space(space, 4);

    ShearFlowAnalytic exact{gamma};

    auto shear_bc = std::make_shared<BoundaryConditionEuler>(
        BCTypeEuler::FAR_FIELD, [exact](const Eigen::Vector2d& x) { return exact.conserved(x); });

    for (const auto& [name, _] : mesh->get_boundary_tags()) {
        mesh->set_boundary_condition_euler(name, shear_bc);
    }
    mesh->build_precomputed_faces();
    for (const auto& [name, _] : mesh->get_boundary_tags()) {
        mesh->set_boundary_condition_euler(name, shear_bc);
    }

    constexpr double dt = 1.0e-5;
    constexpr double T_final = 5.0e-4;
    constexpr int save_every = 50;

    NavierStokesDGSolver solver(mesh, gamma, mu, prandtl, penalty);
    auto frames = solver.solve([exact](const Eigen::Vector2d& x) { return exact.conserved(x); },
                               T_final, dt, save_every);

    ASSERT_FALSE(frames.empty());
    const auto& final_frame = frames.back();

    std::array<double, 4> errors{};
    for (int v = 0; v < 4; ++v) {
        errors[v] = compute_variable_error(mesh, final_frame, exact, gamma, v);
    }

    EXPECT_LT(errors[0], 5e-4) << "Density drifted from analytic profile";
    EXPECT_LT(errors[1], 5e-3) << "Velocity u deviated beyond tolerance";
    EXPECT_LT(errors[2], 5e-4) << "Velocity v deviated beyond tolerance";
    EXPECT_LT(errors[3], 5e-3) << "Pressure deviated beyond tolerance";
}
