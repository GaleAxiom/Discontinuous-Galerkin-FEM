/**
 * @file navier_stokes_weak_formulation.hpp
 * @brief Weak formulation for compressible Navier-Stokes equations (laminar)
 */

#pragma once

#include "dgfem/weak_forms/euler_weak_formulation.hpp"

#include <Eigen/Dense>

namespace dgfem {

/**
 * @brief Navier-Stokes weak formulation with laminar viscous fluxes using SIPG-style penalties
 */
class NavierStokesWeakFormulation : public EulerWeakFormulation {
public:
    NavierStokesWeakFormulation(double gamma = 1.4, double dynamic_viscosity = 1.0e-3,
                                double prandtl = 0.72, double penalty_prefactor = 5.0);
    ~NavierStokesWeakFormulation() override = default;

    [[nodiscard]] std::string get_type() const override { return "NavierStokes"; }
    [[nodiscard]] bool has_viscous_terms() const noexcept override { return true; }

    [[nodiscard]] Eigen::MatrixXd
    viscous_volume_residual(const Eigen::MatrixXd& u_coeffs_elem,
                            const std::map<std::string, Eigen::MatrixXd>& elem_data,
                            std::shared_ptr<DGSpace> dg_space) const override;

    [[nodiscard]] std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> viscous_interior_face_residual(
        const Eigen::MatrixXd& u_coeffs_L, const Eigen::MatrixXd& u_coeffs_R,
        const std::map<std::string, Eigen::MatrixXd>& face_data_L,
        const std::map<std::string, Eigen::MatrixXd>& face_data_R,
        std::shared_ptr<DGSpace> dg_space,
        const Eigen::VectorXi& permutation = Eigen::VectorXi()) const override;

    [[nodiscard]] Eigen::MatrixXd
    viscous_boundary_face_residual(const Eigen::MatrixXd& u_coeffs,
                                   const std::map<std::string, Eigen::MatrixXd>& face_data,
                                   std::shared_ptr<BoundaryConditionEuler> bc,
                                   std::shared_ptr<DGSpace> dg_space) const override;

    [[nodiscard]] double compute_penalty_parameter(int p, double h) const override;

    [[nodiscard]] double get_dynamic_viscosity() const noexcept { return mu_; }
    [[nodiscard]] double get_prandtl() const noexcept { return prandtl_; }

private:
    struct PrimitiveGradientData {
        double rho{};
        double u{};
        double v{};
        double p{};
        double temperature{};
        Eigen::Vector2d grad_rho{Eigen::Vector2d::Zero()};
        Eigen::Vector2d grad_u{Eigen::Vector2d::Zero()};
        Eigen::Vector2d grad_v{Eigen::Vector2d::Zero()};
        Eigen::Vector2d grad_T{Eigen::Vector2d::Zero()};
        double divergence{};
        double tau_xx{};
        double tau_xy{};
        double tau_yy{};
        double q_x{};
        double q_y{};
    };

    [[nodiscard]] PrimitiveGradientData
    compute_primitive_gradients(const Eigen::Vector4d& U,
                                const Eigen::Matrix<double, 4, 2>& grad_U) const;

    [[nodiscard]] std::pair<Eigen::Vector4d, Eigen::Vector4d>
    compute_viscous_fluxes(const Eigen::Vector4d& U,
                           const Eigen::Matrix<double, 4, 2>& grad_U) const;

    double mu_;
    double prandtl_;
    double sigma0_;
    double gas_constant_;
};

}  // namespace dgfem
