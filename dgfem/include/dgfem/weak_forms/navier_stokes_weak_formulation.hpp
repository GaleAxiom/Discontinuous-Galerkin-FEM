/**
 * @file navier_stokes_weak_formulation.hpp
 * @brief Weak formulation for compressible Navier-Stokes equations (laminar)
 */

#pragma once

#include "dgfem/weak_forms/euler_weak_formulation.hpp"

#include <array>

namespace dgfem {

/// Gradient of the 4 conserved variables w.r.t. physical (x,y): GradU4[v] = (d U_v/dx, d U_v/dy)
using GradU4 = std::array<Vec2, 4>;

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

    [[nodiscard]] DView2 viscous_volume_residual(const DView2& u_coeffs_elem,
                                                 const std::map<std::string, DView2>& elem_data,
                                                 std::shared_ptr<DGSpace> dg_space) const override;

    [[nodiscard]] std::tuple<DView2, DView2>
    viscous_interior_face_residual(const DView2& u_coeffs_L, const DView2& u_coeffs_R,
                                   const std::map<std::string, DView2>& face_data_L,
                                   const std::map<std::string, DView2>& face_data_R,
                                   std::shared_ptr<DGSpace> dg_space,
                                   const IView1& permutation = IView1()) const override;

    [[nodiscard]] DView2
    viscous_boundary_face_residual(const DView2& u_coeffs,
                                   const std::map<std::string, DView2>& face_data,
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
        Vec2 grad_rho{0.0, 0.0};
        Vec2 grad_u{0.0, 0.0};
        Vec2 grad_v{0.0, 0.0};
        Vec2 grad_T{0.0, 0.0};
        double divergence{};
        double tau_xx{};
        double tau_xy{};
        double tau_yy{};
        double q_x{};
        double q_y{};
    };

    [[nodiscard]] PrimitiveGradientData compute_primitive_gradients(const Vec4& U,
                                                                    const GradU4& grad_U) const;

    [[nodiscard]] std::pair<Vec4, Vec4> compute_viscous_fluxes(const Vec4& U,
                                                               const GradU4& grad_U) const;

    double mu_;
    double prandtl_;
    double sigma0_;
    double gas_constant_;
};

}  // namespace dgfem
