/**
 * @file advection_weak_formulation.hpp
 * @brief Weak formulation for advection equation
 */

#pragma once

#include "dgfem/weak_forms/weak_formulation_base.hpp"

namespace dgfem {

/**
 * @brief Weak formulation for advection equation
 */
class AdvectionWeakFormulation : public TimeDependentWeakFormulation {
public:
    explicit AdvectionWeakFormulation(const Eigen::Vector2d& velocity);
    ~AdvectionWeakFormulation() override = default;

    // Delete copy, default move
    AdvectionWeakFormulation(const AdvectionWeakFormulation&) = delete;
    AdvectionWeakFormulation& operator=(const AdvectionWeakFormulation&) = delete;
    AdvectionWeakFormulation(AdvectionWeakFormulation&&) noexcept = default;
    AdvectionWeakFormulation& operator=(AdvectionWeakFormulation&&) noexcept = default;

    [[nodiscard]] std::string get_type() const override { return "Advection"; }
    [[nodiscard]] int get_n_vars() const override { return 1; }

    /**
     * @brief Volume (stiffness) integral (implements base class method)
     */
    [[nodiscard]] Eigen::MatrixXd
    compute_volume_integral(const std::map<std::string, Eigen::MatrixXd>& elem_data,
                            std::shared_ptr<DGSpace> dg_space) const override;

    /**
     * @brief Interior face integral (implements base class method)
     */
    [[nodiscard]] std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>
    compute_interior_face_integral(
        int elem_L, int face_L, int elem_R, int face_R, std::shared_ptr<DGMesh> mesh,
        const Eigen::VectorXi& permutation = Eigen::VectorXi()) const override;

    /**
     * @brief Boundary face integral (implements base class method)
     */
    [[nodiscard]] std::tuple<Eigen::MatrixXd, Eigen::VectorXd> compute_boundary_face_integral(
        int elem_id, int face_id, std::shared_ptr<DGMesh> mesh,
        std::function<double(const Eigen::Vector2d&)> bc_func) const override;

private:
    Eigen::Vector2d beta_;  ///< Advection velocity
};

}  // namespace dgfem
