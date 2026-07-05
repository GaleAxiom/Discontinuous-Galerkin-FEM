/**
 * @file conditions.hpp
 * @brief Boundary conditions for DGFEM
 */

#pragma once

#include "dgfem/kokkos_math.hpp"

#include <functional>
#include <memory>

namespace dgfem {

/**
 * @brief Boundary condition types
 */
enum class BCType {
    DIRICHLET,  ///< Dirichlet BC: u = g
    NEUMANN,    ///< Neumann BC: du/dn = g
    ROBIN       ///< Robin BC: alpha*u + du/dn = g
};

/**
 * @brief Boundary condition types for Euler equations
 */
enum class BCTypeEuler {
    FAR_FIELD,     ///< Far-field BC
    SLIP_WALL,     ///< Slip wall BC
    NO_SLIP_WALL,  ///< No-slip wall BC (for future use)
    PERIODIC       ///< Periodic BC
};

/**
 * @brief Base boundary condition class
 */
class BoundaryCondition {
public:
    BoundaryCondition(BCType type, double value);
    BoundaryCondition(BCType type, std::function<double(const Vec2&)> func);
    BoundaryCondition(BCType type, double value, double robin_alpha);
    BoundaryCondition(BCType type, std::function<double(const Vec2&)> func, double robin_alpha);

    virtual ~BoundaryCondition() = default;

    // Delete copy constructor and copy assignment
    BoundaryCondition(const BoundaryCondition&) = delete;
    BoundaryCondition& operator=(const BoundaryCondition&) = delete;

    // Default move constructor and move assignment
    BoundaryCondition(BoundaryCondition&&) noexcept = default;
    BoundaryCondition& operator=(BoundaryCondition&&) noexcept = default;

    /**
     * @brief Evaluate boundary condition at point x
     */
    [[nodiscard]] double evaluate(const Vec2& x) const;

    [[nodiscard]] BCType get_type() const noexcept { return type_; }
    [[nodiscard]] double get_robin_alpha() const noexcept { return robin_alpha_; }
    [[nodiscard]] double get_value() const noexcept { return constant_value_; }

private:
    BCType type_;
    double constant_value_;
    std::function<double(const Vec2&)> function_value_;
    bool is_function_;
    double robin_alpha_;
};

/**
 * @brief Boundary condition for Euler equations
 */
class BoundaryConditionEuler {
public:
    BoundaryConditionEuler(BCTypeEuler type, const Vec4& value);
    BoundaryConditionEuler(BCTypeEuler type, std::function<Vec4(const Vec2&)> func);

    virtual ~BoundaryConditionEuler() = default;

    // Delete copy constructor and copy assignment
    BoundaryConditionEuler(const BoundaryConditionEuler&) = delete;
    BoundaryConditionEuler& operator=(const BoundaryConditionEuler&) = delete;

    // Default move constructor and move assignment
    BoundaryConditionEuler(BoundaryConditionEuler&&) noexcept = default;
    BoundaryConditionEuler& operator=(BoundaryConditionEuler&&) noexcept = default;

    /**
     * @brief Evaluate boundary condition at point x
     */
    [[nodiscard]] Vec4 evaluate(const Vec2& x) const;

    [[nodiscard]] BCTypeEuler get_type() const noexcept { return type_; }

private:
    BCTypeEuler type_;
    Vec4 constant_value_;
    std::function<Vec4(const Vec2&)> function_value_;
    bool is_function_;
};

// Convenience factory functions
[[nodiscard]] std::shared_ptr<BoundaryCondition> make_dirichlet_bc(double value);
[[nodiscard]] std::shared_ptr<BoundaryCondition>
make_dirichlet_bc(std::function<double(const Vec2&)> func);
[[nodiscard]] std::shared_ptr<BoundaryCondition> make_neumann_bc(double value);
[[nodiscard]] std::shared_ptr<BoundaryCondition>
make_neumann_bc(std::function<double(const Vec2&)> func);
[[nodiscard]] std::shared_ptr<BoundaryCondition> make_robin_bc(double value, double alpha);
[[nodiscard]] std::shared_ptr<BoundaryCondition>
make_robin_bc(std::function<double(const Vec2&)> func, double alpha);

}  // namespace dgfem