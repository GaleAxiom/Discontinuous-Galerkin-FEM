/**
 * @file euler_weak_formulation.hpp
 * @brief Weak formulation for Euler equations
 */

#pragma once

#include "dgfem/boundary/conditions.hpp"
#include "dgfem/weak_forms/weak_formulation_base.hpp"

namespace dgfem {

/**
 * @brief Weak formulation for Euler equations
 */
class EulerWeakFormulation : public TimeDependentWeakFormulation {
public:
    explicit EulerWeakFormulation(double gamma = 1.4);
    ~EulerWeakFormulation() override = default;

    // Delete copy, default move
    EulerWeakFormulation(const EulerWeakFormulation&) = delete;
    EulerWeakFormulation& operator=(const EulerWeakFormulation&) = delete;
    EulerWeakFormulation(EulerWeakFormulation&&) noexcept = default;
    EulerWeakFormulation& operator=(EulerWeakFormulation&&) noexcept = default;

    [[nodiscard]] std::string get_type() const override { return "Euler"; }
    [[nodiscard]] int get_n_vars() const override { return 4; }

    /**
     * @brief Assemble for Euler - note: Euler uses residual-based assembly, not matrix assembly
     */
    void assemble(DGAssembler& assembler, std::function<double(const Vec2&)> source_func = nullptr,
                  std::function<double(const Vec2&)> bc_func = nullptr) const override;

    /**
     * @brief Not used for Euler (residual-based formulation)
     */
    [[nodiscard]] DView2 compute_volume_integral(const std::map<std::string, DView2>& elem_data,
                                                 std::shared_ptr<DGSpace> dg_space) const override {
        throw std::runtime_error("Euler formulation uses residual assembly, not matrix assembly");
    }

    /**
     * @brief Not used for Euler (residual-based formulation)
     */
    [[nodiscard]] std::tuple<DView2, DView2, DView2, DView2>
    compute_interior_face_integral(int elem_L, int face_L, int elem_R, int face_R,
                                   std::shared_ptr<DGMesh> mesh,
                                   const IView1& permutation = IView1()) const override {
        throw std::runtime_error("Euler formulation uses residual assembly, not matrix assembly");
    }

    /**
     * @brief Not used for Euler (residual-based formulation)
     */
    [[nodiscard]] std::tuple<DView2, DView1>
    compute_boundary_face_integral(int elem_id, int face_id, std::shared_ptr<DGMesh> mesh,
                                   std::function<double(const Vec2&)> bc_func) const override {
        throw std::runtime_error("Euler formulation uses residual assembly, not matrix assembly");
    }

public:
    /**
     * @brief Compute volume residual: R_vol_i = ∫_K (F ⋅ ∇φ_i_x + G ⋅ ∇φ_i_y) dK
     */
    [[nodiscard]] virtual DView2 volume_residual(const DView2& u_coeffs_elem,
                                                 const std::map<std::string, DView2>& elem_data,
                                                 std::shared_ptr<DGSpace> dg_space) const;

    /**
     * @brief Compute interior face residual using Rusanov flux
     */
    [[nodiscard]] virtual std::tuple<DView2, DView2>
    interior_face_residual(const DView2& u_coeffs_L, const DView2& u_coeffs_R,
                           const std::map<std::string, DView2>& face_data_L,
                           const std::map<std::string, DView2>& face_data_R,
                           std::shared_ptr<DGSpace> dg_space,
                           const IView1& permutation = IView1()) const;

    /**
     * @brief Compute boundary face residual
     */
    [[nodiscard]] virtual DView2
    boundary_face_residual(const DView2& u_coeffs, const std::map<std::string, DView2>& face_data,
                           std::shared_ptr<BoundaryConditionEuler> bc,
                           std::shared_ptr<DGSpace> dg_space) const;

    /**
     * @brief Get flux vectors F and G from conserved variables
     */
    [[nodiscard]] virtual std::tuple<Vec4, Vec4> get_fluxes(const Vec4& U) const;

    /**
     * @brief Compute Rusanov numerical flux
     */
    [[nodiscard]] virtual Vec4 rusanov_flux(const Vec4& U_L, const Vec4& U_R,
                                            const Vec2& normal) const;

    [[nodiscard]] virtual bool has_viscous_terms() const noexcept { return false; }

    [[nodiscard]] virtual DView2
    viscous_volume_residual(const DView2& u_coeffs_elem,
                            const std::map<std::string, DView2>& elem_data,
                            std::shared_ptr<DGSpace> dg_space) const;

    [[nodiscard]] virtual std::tuple<DView2, DView2>
    viscous_interior_face_residual(const DView2& u_coeffs_L, const DView2& u_coeffs_R,
                                   const std::map<std::string, DView2>& face_data_L,
                                   const std::map<std::string, DView2>& face_data_R,
                                   std::shared_ptr<DGSpace> dg_space,
                                   const IView1& permutation = IView1()) const;

    [[nodiscard]] virtual DView2 viscous_boundary_face_residual(
        const DView2& u_coeffs, const std::map<std::string, DView2>& face_data,
        std::shared_ptr<BoundaryConditionEuler> bc, std::shared_ptr<DGSpace> dg_space) const;

    [[nodiscard]] virtual double compute_penalty_parameter(int /*p*/, double /*h*/) const {
        return 0.0;
    }

    [[nodiscard]] double get_gamma() const noexcept { return gamma_; }

protected:
    double gamma_;            ///< Ratio of specific heats
    double gamma_minus_one_;  ///< Cached gamma - 1.0 for performance
};

// Utility functions for Euler equations
[[nodiscard]] Vec4 primitive_to_conserved(const Vec4& primitive, double gamma = 1.4);
[[nodiscard]] Vec4 conserved_to_primitive(const Vec4& conserved, double gamma = 1.4);

}  // namespace dgfem
