/**
 * @file weak_formulation_base.hpp
 * @brief Base classes for weak formulations
 */

#pragma once

#include <Eigen/Dense>

#include <functional>
#include <map>
#include <memory>
#include <string>

namespace dgfem {

// Forward declarations
class DGMesh;
class DGSpace;
class DGAssembler;
class BoundaryCondition;

/**
 * @brief Base class for all weak formulations
 */
class WeakFormulation {
public:
    virtual ~WeakFormulation() = default;

    /**
     * @brief Get the type of weak formulation
     */
    [[nodiscard]] virtual std::string get_type() const = 0;

    /**
     * @brief Check if this is a time-dependent formulation
     */
    [[nodiscard]] virtual bool is_time_dependent() const = 0;

    /**
     * @brief Get number of variables (1 for scalar, 4 for Euler, etc.)
     */
    [[nodiscard]] virtual int get_n_vars() const = 0;

    /**
     * @brief Assemble the system for this weak formulation
     * @param assembler The assembler object providing assembly utilities
     * @param source_func Optional source function (for elliptic problems)
     * @param bc_func Optional boundary condition function (for time-dependent problems)
     */
    virtual void
    assemble(DGAssembler& assembler,
             std::function<double(const Eigen::Vector2d&)> source_func = nullptr,
             std::function<double(const Eigen::Vector2d&)> bc_func = nullptr) const = 0;
};

/**
 * @brief Base class for time-independent (elliptic) weak formulations
 */
class TimeIndependentWeakFormulation : public WeakFormulation {
public:
    [[nodiscard]] bool is_time_dependent() const override { return false; }
    [[nodiscard]] int get_n_vars() const override { return 1; }

    /**
     * @brief Assemble time-independent system
     */
    void assemble(DGAssembler& assembler,
                  std::function<double(const Eigen::Vector2d&)> source_func = nullptr,
                  std::function<double(const Eigen::Vector2d&)> bc_func = nullptr) const override;

    /**
     * @brief Volume integral contribution - must be implemented by derived classes
     */
    [[nodiscard]] virtual Eigen::MatrixXd
    compute_volume_integral(const std::map<std::string, Eigen::MatrixXd>& elem_data,
                            std::shared_ptr<DGSpace> dg_space) const = 0;

    /**
     * @brief Interior face integral contribution - must be implemented by derived classes
     */
    [[nodiscard]] virtual std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd,
                                     Eigen::MatrixXd>
    compute_interior_face_integral(
        int elem_L, int face_L, int elem_R, int face_R, std::shared_ptr<DGMesh> mesh,
        const Eigen::VectorXi& permutation = Eigen::VectorXi()) const = 0;

    /**
     * @brief Boundary face integral contribution - must be implemented by derived classes
     */
    [[nodiscard]] virtual Eigen::MatrixXd
    compute_boundary_face_integral(int elem_id, int face_id, std::shared_ptr<DGMesh> mesh,
                                   std::shared_ptr<BoundaryCondition> bc) const = 0;

    /**
     * @brief Boundary RHS contribution - must be implemented by derived classes
     */
    [[nodiscard]] virtual Eigen::VectorXd
    compute_boundary_rhs_integral(int elem_id, int face_id, std::shared_ptr<DGMesh> mesh,
                                  std::shared_ptr<BoundaryCondition> bc) const = 0;

    /**
     * @brief Source integral contribution - common implementation
     */
    [[nodiscard]] virtual Eigen::VectorXd
    compute_source_integral(int elem_id, std::function<double(const Eigen::Vector2d&)> source_func,
                            std::shared_ptr<DGMesh> mesh) const;
};

/**
 * @brief Base class for time-dependent (hyperbolic) weak formulations
 */
class TimeDependentWeakFormulation : public WeakFormulation {
public:
    [[nodiscard]] bool is_time_dependent() const override { return true; }

    /**
     * @brief Assemble time-dependent spatial operator
     */
    void assemble(DGAssembler& assembler,
                  std::function<double(const Eigen::Vector2d&)> source_func = nullptr,
                  std::function<double(const Eigen::Vector2d&)> bc_func = nullptr) const override;

    /**
     * @brief Common mass matrix integral for all time-dependent problems
     */
    [[nodiscard]] Eigen::MatrixXd
    compute_mass_integral(const std::map<std::string, Eigen::MatrixXd>& elem_data,
                          std::shared_ptr<DGSpace> dg_space) const;

    /**
     * @brief Volume (stiffness) integral contribution - must be implemented by derived classes
     */
    [[nodiscard]] virtual Eigen::MatrixXd
    compute_volume_integral(const std::map<std::string, Eigen::MatrixXd>& elem_data,
                            std::shared_ptr<DGSpace> dg_space) const = 0;

    /**
     * @brief Interior face integral contribution - must be implemented by derived classes
     */
    [[nodiscard]] virtual std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd,
                                     Eigen::MatrixXd>
    compute_interior_face_integral(
        int elem_L, int face_L, int elem_R, int face_R, std::shared_ptr<DGMesh> mesh,
        const Eigen::VectorXi& permutation = Eigen::VectorXi()) const = 0;

    /**
     * @brief Boundary face integral contribution - must be implemented by derived classes
     * Returns (L_bc, F_bc) - matrix and RHS contributions
     */
    [[nodiscard]] virtual std::tuple<Eigen::MatrixXd, Eigen::VectorXd>
    compute_boundary_face_integral(int elem_id, int face_id, std::shared_ptr<DGMesh> mesh,
                                   std::function<double(const Eigen::Vector2d&)> bc_func) const = 0;
};

}  // namespace dgfem
