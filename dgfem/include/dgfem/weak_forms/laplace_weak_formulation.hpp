/**
 * @file laplace_weak_formulation.hpp
 * @brief Weak formulation for Laplace equation
 */

#pragma once

#include "dgfem/boundary/conditions.hpp"
#include "dgfem/weak_forms/weak_formulation_base.hpp"

namespace dgfem {

/**
 * @brief Weak formulation for Laplace equation using interior penalty DG method
 */
class LaplaceWeakFormulation : public TimeIndependentWeakFormulation {
public:
    explicit LaplaceWeakFormulation(double penalty_parameter = 10.0);
    ~LaplaceWeakFormulation() override = default;

    // Delete copy, default move
    LaplaceWeakFormulation(const LaplaceWeakFormulation&) = delete;
    LaplaceWeakFormulation& operator=(const LaplaceWeakFormulation&) = delete;
    LaplaceWeakFormulation(LaplaceWeakFormulation&&) noexcept = default;
    LaplaceWeakFormulation& operator=(LaplaceWeakFormulation&&) noexcept = default;

    [[nodiscard]] std::string get_type() const override { return "Laplace"; }

    /**
     * @brief Custom assembly for Laplace (overrides base class to use BoundaryCondition objects)
     */
    void assemble(DGAssembler& assembler, std::function<double(const Vec2&)> source_func = nullptr,
                  std::function<double(const Vec2&)> bc_func = nullptr) const override;

    /**
     * @brief Compute penalty parameter based on element size and polynomial order
     */
    [[nodiscard]] double compute_penalty_parameter(int p, double h) const;

    /**
     * @brief Volume integral contribution (implements base class method)
     */
    [[nodiscard]] DView2 compute_volume_integral(const std::map<std::string, DView2>& elem_data,
                                                 std::shared_ptr<DGSpace> dg_space) const override;

    /**
     * @brief Interior face integral contribution (implements base class method)
     */
    [[nodiscard]] std::tuple<DView2, DView2, DView2, DView2>
    compute_interior_face_integral(int elem_L, int face_L, int elem_R, int face_R,
                                   std::shared_ptr<DGMesh> mesh,
                                   const IView1& permutation = IView1()) const override;

    /**
     * @brief Boundary face integral contribution (implements base class method)
     */
    [[nodiscard]] DView2
    compute_boundary_face_integral(int elem_id, int face_id, std::shared_ptr<DGMesh> mesh,
                                   std::shared_ptr<BoundaryCondition> bc) const override;

    /**
     * @brief Source integral contribution (implements base class method)
     */
    [[nodiscard]] DView1 compute_source_integral(int elem_id,
                                                 std::function<double(const Vec2&)> source_func,
                                                 std::shared_ptr<DGMesh> mesh) const override;

    /**
     * @brief Boundary RHS contribution (implements base class method)
     */
    [[nodiscard]] DView1
    compute_boundary_rhs_integral(int elem_id, int face_id, std::shared_ptr<DGMesh> mesh,
                                  std::shared_ptr<BoundaryCondition> bc) const override;

    /**
     * @brief Helper function to interpolate gradients to face quadrature points
     */
    [[nodiscard]] DView2
    interpolate_gradient_to_face(const std::map<std::string, DView2>& elem_data, int face_id,
                                 std::shared_ptr<DGSpace> dg_space) const;

private:
    double sigma_0_;  ///< Base penalty parameter
};

}  // namespace dgfem
