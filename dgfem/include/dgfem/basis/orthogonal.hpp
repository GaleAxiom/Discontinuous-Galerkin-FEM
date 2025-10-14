/**
 * @file orthogonal.hpp
 * @brief Base class for orthogonal basis functions with modern C++20 features
 */

#pragma once

#include "dgfem/reference/elements.hpp"

#include <Eigen/Dense>

#include <memory>
#include <sstream>
#include <string_view>

namespace dgfem {

// Error types for better error handling with modern enum class
enum class BasisError { InvalidOrder, InvalidPoint, ComputationFailed, UnsupportedOperation };

struct BasisErrorInfo {
    BasisError error;
    std::string message;

    [[nodiscard]] std::string to_string() const {
        std::ostringstream oss;
        oss << "BasisError: " << message;
        return oss.str();
    }
};

/**
 * @brief Base class for orthogonal basis functions with modern C++ design
 */
class OrthogonalBasis {
public:
    OrthogonalBasis(std::shared_ptr<ReferenceElement> ref_element, int order);
    virtual ~OrthogonalBasis() = default;

    // Delete copy operations, allow move
    OrthogonalBasis(const OrthogonalBasis&) = delete;
    OrthogonalBasis& operator=(const OrthogonalBasis&) = delete;
    OrthogonalBasis(OrthogonalBasis&&) noexcept = default;
    OrthogonalBasis& operator=(OrthogonalBasis&&) noexcept = default;

    /**
     * @brief Evaluate basis functions at reference point xi
     * @param xi Reference coordinates
     * @return Vector of basis function values
     */
    virtual Eigen::VectorXd evaluate(const Eigen::Vector2d& xi) const = 0;

    /**
     * @brief Evaluate basis function gradients at reference point xi
     * @param xi Reference coordinates
     * @return Matrix where row i contains gradient of basis function i
     */
    virtual Eigen::MatrixXd evaluate_gradient(const Eigen::Vector2d& xi) const = 0;

    // Modern getters with [[nodiscard]]
    [[nodiscard]] constexpr int get_order() const noexcept { return order_; }
    [[nodiscard]] constexpr int get_n_basis() const noexcept { return n_basis_; }
    [[nodiscard]] const std::shared_ptr<ReferenceElement>& get_ref_element() const noexcept {
        return ref_element_;
    }

    /**
     * @brief Compute number of basis functions for given order
     * Must be implemented by derived classes
     */
    virtual int compute_n_basis() const = 0;

    /**
     * @brief Validate if order is supported
     */
    [[nodiscard]] static constexpr bool is_order_valid(int order) noexcept {
        return order >= 1 && order <= 10;
    }

protected:
    std::shared_ptr<ReferenceElement> ref_element_;
    int order_;
    int n_basis_;
};

}  // namespace dgfem