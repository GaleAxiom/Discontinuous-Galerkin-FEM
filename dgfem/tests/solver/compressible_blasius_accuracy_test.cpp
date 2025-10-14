#include <Eigen/Dense>
#include <cmath>
#include <dgfem/boundary/conditions.hpp>
#include <dgfem/core/space.hpp>
#include <dgfem/solver/dg_solver.hpp>
#include <dgfem/solver/weak_form.hpp>
#include <dgfem/utils/mesh_creation.hpp>
#include <dgfem/utils/vtk_writer.hpp>
#include <filesystem>

#include <algorithm>
#include <array>
#include <iostream>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

using namespace dgfem;

namespace {

constexpr double kGasConstant = 1.0;

struct BlasiusTable {
    std::vector<double> eta;
    std::vector<double> f;
    std::vector<double> fp;
};

BlasiusTable build_blasius_table() {
    BlasiusTable table;
    constexpr double eta_max = 12.0;
    constexpr double d_eta = 0.01;
    size_t steps = static_cast<size_t>(eta_max / d_eta) + 1;
    table.eta.reserve(steps);
    table.f.reserve(steps);
    table.fp.reserve(steps);

    double eta = 0.0;
    double f = 0.0;
    double fp = 0.0;
    double fpp = 0.332057336215;

    auto deriv = [](double f_val, double fp_val, double fpp_val) {
        double fprime = fp_val;
        double fprimeprime = fpp_val;
        double fprimeprimeprime = -0.5 * f_val * fpp_val;
        return std::array<double, 3>{fprime, fprimeprime, fprimeprimeprime};
    };

    for (size_t i = 0; i < steps; ++i) {
        table.eta.push_back(eta);
        table.f.push_back(f);
        table.fp.push_back(fp);

        // RK4 step
        auto k1 = deriv(f, fp, fpp);
        auto k2 =
            deriv(f + 0.5 * d_eta * k1[0], fp + 0.5 * d_eta * k1[1], fpp + 0.5 * d_eta * k1[2]);
        auto k3 =
            deriv(f + 0.5 * d_eta * k2[0], fp + 0.5 * d_eta * k2[1], fpp + 0.5 * d_eta * k2[2]);
        auto k4 = deriv(f + d_eta * k3[0], fp + d_eta * k3[1], fpp + d_eta * k3[2]);

        f += (d_eta / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]);
        fp += (d_eta / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]);
        fpp += (d_eta / 6.0) * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]);

        eta += d_eta;
        if (fp > 0.999999) {
            fp = 1.0;
            fpp = 0.0;
        }
    }

    return table;
}

const BlasiusTable& get_blasius_table() {
    static const BlasiusTable table = build_blasius_table();
    return table;
}

double interpolate(const std::vector<double>& xs, const std::vector<double>& ys, double x) {
    if (x <= xs.front()) {
        return ys.front();
    }
    if (x >= xs.back()) {
        return ys.back();
    }

    auto upper = std::upper_bound(xs.begin(), xs.end(), x);
    size_t idx = std::distance(xs.begin(), upper);
    if (idx == 0) {
        return ys.front();
    }
    double x0 = xs[idx - 1];
    double x1 = xs[idx];
    double y0 = ys[idx - 1];
    double y1 = ys[idx];
    double t = (x - x0) / (x1 - x0);
    return y0 + t * (y1 - y0);
}

struct CompressibleBlasiusAnalytic {
    double gamma;
    double mu;
    double prandtl;
    double mach_inf;
    double T_inf;
    double T_wall;
    double p_inf;
    double x_min;

    [[nodiscard]] Eigen::Vector4d primitive(const Eigen::Vector2d& x) const {
        const auto& table = get_blasius_table();

        double x_pos = std::max(x[0], x_min);
        double nu_edge = mu / (p_inf / (kGasConstant * T_inf));
        double U_inf = mach_inf * std::sqrt(gamma * kGasConstant * T_inf);
        double eta_scale = std::sqrt(U_inf / (2.0 * nu_edge * x_pos));
        double eta = std::clamp(x[1] * eta_scale, table.eta.front(), table.eta.back());

        double f_val = interpolate(table.eta, table.f, eta);
        double fp_val = interpolate(table.eta, table.fp, eta);

        double recovery = std::pow(prandtl, 1.0 / 3.0);
        double term1 = recovery * (T_wall / T_inf - 1.0) * fp_val;
        double term2 = 0.5 * (gamma - 1.0) * mach_inf * mach_inf * (1.0 - fp_val * fp_val);
        double T = T_inf * (1.0 + term1 + term2);
        double rho = p_inf / (kGasConstant * T);
        double u = U_inf * fp_val;
        double v_prefactor = 0.5 * std::sqrt(nu_edge * U_inf / x_pos);
        double v = v_prefactor * (eta * fp_val - f_val);
        double p = p_inf;

        Eigen::Vector4d W;
        W << rho, u, v, p;
        return W;
    }

    [[nodiscard]] Eigen::Vector4d conserved(const Eigen::Vector2d& x) const {
        return primitive_to_conserved(primitive(x), gamma);
    }
};

[[nodiscard]] double compute_variable_error(const std::shared_ptr<DGMesh>& mesh,
                                            const Eigen::MatrixXd& numerical_sol,
                                            const CompressibleBlasiusAnalytic& exact, double gamma,
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

class NavierStokesBlasiusAccuracyTest : public ::testing::Test {
protected:
    void SetUp() override { MeshCreator::initialize_gmsh(); }

    void TearDown() override { MeshCreator::finalize_gmsh(); }
};

TEST_F(NavierStokesBlasiusAccuracyTest, MaintainsCompressibleBlasiusProfile) {
    constexpr double gamma = 1.4;
    constexpr double mu = 5.0e-5;
    constexpr double prandtl = 0.72;
    constexpr double penalty = 10000.0;
    constexpr double mach_inf = 0.3;
    constexpr double T_inf = 1.0;
    constexpr double T_wall = 1.15;
    constexpr double p_inf = 1.0;
    constexpr double x_min = 0.05;

    CompressibleBlasiusAnalytic exact{gamma, mu, prandtl, mach_inf, T_inf, T_wall, p_inf, x_min};

    constexpr double dx = 0.02;
    constexpr int order = 2;

    auto mesh = MeshCreator::create_rectangular_mesh(dx, false, x_min, 0.55, 0.0, 0.2);
    auto space = std::make_shared<DGSpace>(mesh->get_element_type(), order);
    mesh->initialize_dg_space(space, 4);

    auto blasius_bc = std::make_shared<BoundaryConditionEuler>(
        BCTypeEuler::FAR_FIELD, [exact](const Eigen::Vector2d& x) { return exact.conserved(x); });

    for (const auto& [name, _] : mesh->get_boundary_tags()) {
        mesh->set_boundary_condition_euler(name, blasius_bc);
    }
    mesh->build_precomputed_faces();
    for (const auto& [name, _] : mesh->get_boundary_tags()) {
        mesh->set_boundary_condition_euler(name, blasius_bc);
    }

    constexpr double dt = 2.0e-6;
    constexpr double T_final = 2.0e-4;
    constexpr int save_every = 50;

    NavierStokesDGSolver solver(mesh, gamma, mu, prandtl, penalty);
    auto frames = solver.solve([exact](const Eigen::Vector2d& x) { return exact.conserved(x); },
                               T_final, dt, save_every);

    ASSERT_FALSE(frames.empty());
    const auto& final_frame = frames.back();

    const std::filesystem::path output_dir = std::filesystem::path("../..") / "output";
    std::error_code ec;
    std::filesystem::create_directories(output_dir, ec);
    if (ec) {
        std::cerr << "Failed to create visualization directory '" << output_dir.string()
                  << "': " << ec.message() << '\n';
    } else {
        const std::filesystem::path vtk_base = output_dir / "compressible_blasius_profile";
        VTKWriter::write_euler_solution(mesh, final_frame, vtk_base.string(), gamma, 4);
    }

    std::array<double, 4> errors{};
    for (int v = 0; v < 4; ++v) {
        errors[v] = compute_variable_error(mesh, final_frame, exact, gamma, v);
    }
    std::cout << "Errors (density, u, v, p): ";
    for (const auto& err : errors) {
        std::cout << err << " ";
    }
    std::cout << std::endl;
    EXPECT_LT(errors[0], 1.0e-3) << "Density deviated beyond tolerance";
    EXPECT_LT(errors[1], 5.0e-3) << "Streamwise velocity deviated beyond tolerance";
    EXPECT_LT(errors[2], 8.0e-3) << "Wall-normal velocity deviated beyond tolerance";
    EXPECT_LT(errors[3], 5.0e-3) << "Pressure deviated beyond tolerance";
}
