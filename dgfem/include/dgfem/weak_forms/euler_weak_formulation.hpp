/**
 * @file euler_weak_formulation.hpp
 * @brief Weak formulation for Euler equations
 */

#pragma once

#include "dgfem/weak_forms/weak_formulation_base.hpp"
#include "dgfem/boundary/conditions.hpp"

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
    void assemble(DGAssembler& assembler,
                 std::function<double(const Eigen::Vector2d&)> source_func = nullptr,
                 std::function<double(const Eigen::Vector2d&)> bc_func = nullptr) const override;

    /**
     * @brief Not used for Euler (residual-based formulation)
     */
    [[nodiscard]] Eigen::MatrixXd compute_volume_integral(
        const std::map<std::string, Eigen::MatrixXd>& elem_data,
        std::shared_ptr<DGSpace> dg_space) const override {
        throw std::runtime_error("Euler formulation uses residual assembly, not matrix assembly");
    }
    
    /**
     * @brief Not used for Euler (residual-based formulation)
     */
    [[nodiscard]] std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>
    compute_interior_face_integral(int elem_L, int face_L, int elem_R, int face_R,
                                   std::shared_ptr<DGMesh> mesh,
                                   const Eigen::VectorXi& permutation = Eigen::VectorXi()) const override {
        throw std::runtime_error("Euler formulation uses residual assembly, not matrix assembly");
    }
    
    /**
     * @brief Not used for Euler (residual-based formulation)
     */
    [[nodiscard]] std::tuple<Eigen::MatrixXd, Eigen::VectorXd> compute_boundary_face_integral(
        int elem_id, int face_id,
        std::shared_ptr<DGMesh> mesh,
        std::function<double(const Eigen::Vector2d&)> bc_func) const override {
        throw std::runtime_error("Euler formulation uses residual assembly, not matrix assembly");
    }

public:
    /**
     * @brief Compute volume residual: R_vol_i = ∫_K (F ⋅ ∇φ_i_x + G ⋅ ∇φ_i_y) dK
     */
    [[nodiscard]] virtual Eigen::MatrixXd volume_residual(const Eigen::MatrixXd& u_coeffs_elem,
                                   const std::map<std::string, Eigen::MatrixXd>& elem_data,
                                   std::shared_ptr<DGSpace> dg_space) const;
    
    /**
     * @brief Compute interior face residual using Rusanov flux
     */
    [[nodiscard]] virtual std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> interior_face_residual(
        const Eigen::MatrixXd& u_coeffs_L,
        const Eigen::MatrixXd& u_coeffs_R,
        const std::map<std::string, Eigen::MatrixXd>& face_data_L,
        const std::map<std::string, Eigen::MatrixXd>& face_data_R,
        std::shared_ptr<DGSpace> dg_space,
        const Eigen::VectorXi& permutation = Eigen::VectorXi()) const;
    
    /**
     * @brief Compute boundary face residual
     */
    [[nodiscard]] virtual Eigen::MatrixXd boundary_face_residual(
        const Eigen::MatrixXd& u_coeffs,
        const std::map<std::string, Eigen::MatrixXd>& face_data,
        std::shared_ptr<BoundaryConditionEuler> bc,
        std::shared_ptr<DGSpace> dg_space) const;
    
    /**
     * @brief Get flux vectors F and G from conserved variables
     */
    [[nodiscard]] virtual std::tuple<Eigen::Vector4d, Eigen::Vector4d> get_fluxes(const Eigen::Vector4d& U) const;
    
    /**
     * @brief Compute Rusanov numerical flux
     */
    [[nodiscard]] virtual Eigen::Vector4d rusanov_flux(const Eigen::Vector4d& U_L, const Eigen::Vector4d& U_R,
                                 const Eigen::Vector2d& normal) const;

    [[nodiscard]] virtual bool has_viscous_terms() const noexcept { return false; }

    [[nodiscard]] virtual Eigen::MatrixXd viscous_volume_residual(
        const Eigen::MatrixXd& u_coeffs_elem,
        const std::map<std::string, Eigen::MatrixXd>& elem_data,
        std::shared_ptr<DGSpace> dg_space) const;

    [[nodiscard]] virtual std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> viscous_interior_face_residual(
        const Eigen::MatrixXd& u_coeffs_L,
        const Eigen::MatrixXd& u_coeffs_R,
        const std::map<std::string, Eigen::MatrixXd>& face_data_L,
        const std::map<std::string, Eigen::MatrixXd>& face_data_R,
        std::shared_ptr<DGSpace> dg_space,
        const Eigen::VectorXi& permutation = Eigen::VectorXi()) const;

    [[nodiscard]] virtual Eigen::MatrixXd viscous_boundary_face_residual(
        const Eigen::MatrixXd& u_coeffs,
        const std::map<std::string, Eigen::MatrixXd>& face_data,
        std::shared_ptr<BoundaryConditionEuler> bc,
        std::shared_ptr<DGSpace> dg_space) const;

    [[nodiscard]] virtual double compute_penalty_parameter(int /*p*/, double /*h*/) const { return 0.0; }
    
    [[nodiscard]] double get_gamma() const noexcept { return gamma_; }

private:
    double gamma_;  ///< Ratio of specific heats
};

// Utility functions for Euler equations
[[nodiscard]] Eigen::Vector4d primitive_to_conserved(const Eigen::Vector4d& primitive, double gamma = 1.4);
[[nodiscard]] Eigen::Vector4d conserved_to_primitive(const Eigen::Vector4d& conserved, double gamma = 1.4);

} // namespace dgfem
