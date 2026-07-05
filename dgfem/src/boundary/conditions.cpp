/**
 * @file conditions.cpp
 * @brief Implementation of boundary conditions
 */

#include "dgfem/boundary/conditions.hpp"

#include <stdexcept>

namespace dgfem {

// BoundaryCondition implementations
BoundaryCondition::BoundaryCondition(BCType type, double value)
    : type_(type), constant_value_(value), is_function_(false), robin_alpha_(0.0) {}

BoundaryCondition::BoundaryCondition(BCType type, std::function<double(const Vec2&)> func)
    : type_(type), function_value_(func), is_function_(true), robin_alpha_(0.0) {}

BoundaryCondition::BoundaryCondition(BCType type, double value, double robin_alpha)
    : type_(type), constant_value_(value), is_function_(false), robin_alpha_(robin_alpha) {
    if (type != BCType::ROBIN) {
        throw std::invalid_argument(
            "Robin alpha parameter only valid for Robin boundary conditions");
    }
}

BoundaryCondition::BoundaryCondition(BCType type, std::function<double(const Vec2&)> func,
                                     double robin_alpha)
    : type_(type), function_value_(func), is_function_(true), robin_alpha_(robin_alpha) {
    if (type != BCType::ROBIN) {
        throw std::invalid_argument(
            "Robin alpha parameter only valid for Robin boundary conditions");
    }
}

double BoundaryCondition::evaluate(const Vec2& x) const {
    if (is_function_) {
        return function_value_(x);
    } else {
        return constant_value_;
    }
}

// BoundaryConditionEuler implementations
BoundaryConditionEuler::BoundaryConditionEuler(BCTypeEuler type, const Vec4& value)
    : type_(type), constant_value_(value), is_function_(false) {}

BoundaryConditionEuler::BoundaryConditionEuler(BCTypeEuler type,
                                               std::function<Vec4(const Vec2&)> func)
    : type_(type), function_value_(func), is_function_(true) {}

Vec4 BoundaryConditionEuler::evaluate(const Vec2& x) const {
    if (is_function_) {
        return function_value_(x);
    } else {
        return constant_value_;
    }
}

// Factory functions
std::shared_ptr<BoundaryCondition> make_dirichlet_bc(double value) {
    return std::make_shared<BoundaryCondition>(BCType::DIRICHLET, value);
}

std::shared_ptr<BoundaryCondition> make_dirichlet_bc(std::function<double(const Vec2&)> func) {
    return std::make_shared<BoundaryCondition>(BCType::DIRICHLET, func);
}

std::shared_ptr<BoundaryCondition> make_neumann_bc(double value) {
    return std::make_shared<BoundaryCondition>(BCType::NEUMANN, value);
}

std::shared_ptr<BoundaryCondition> make_neumann_bc(std::function<double(const Vec2&)> func) {
    return std::make_shared<BoundaryCondition>(BCType::NEUMANN, func);
}

std::shared_ptr<BoundaryCondition> make_robin_bc(double value, double alpha) {
    return std::make_shared<BoundaryCondition>(BCType::ROBIN, value, alpha);
}

std::shared_ptr<BoundaryCondition> make_robin_bc(std::function<double(const Vec2&)> func,
                                                 double alpha) {
    return std::make_shared<BoundaryCondition>(BCType::ROBIN, func, alpha);
}

}  // namespace dgfem