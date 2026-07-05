/**
 * @file orthogonal.hpp
 * @brief Base class for orthogonal basis functions with modern C++20 features
 */

#pragma once

#include "dgfem/kokkos_math.hpp"
#include "dgfem/reference/elements.hpp"

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
    virtual DView1 evaluate(const Vec2& xi) const = 0;

    /**
     * @brief Evaluate basis function gradients at reference point xi
     * @param xi Reference coordinates
     * @return Matrix where row i contains gradient of basis function i
     */
    virtual DView2 evaluate_gradient(const Vec2& xi) const = 0;

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

/**
 * @brief Thin CRTP layer between OrthogonalBasis and each concrete basis (Legendre/
 * Dubiner/Monomial). `DGSpace` only ever holds a `std::shared_ptr<OrthogonalBasis>`
 * chosen at runtime, so the virtual boundary at `OrthogonalBasis` is preserved exactly
 * -- this class exists purely to dispatch that one virtual call to each derived
 * class's `*_impl` method via `static_cast`, which concrete classes implement as
 * plain (non-virtual) member functions.
 */
template <typename Derived>
class OrthogonalBasisCRTP : public OrthogonalBasis {
public:
    using OrthogonalBasis::OrthogonalBasis;

    DView1 evaluate(const Vec2& xi) const final {
        return static_cast<const Derived*>(this)->evaluate_impl(xi);
    }
    DView2 evaluate_gradient(const Vec2& xi) const final {
        return static_cast<const Derived*>(this)->evaluate_gradient_impl(xi);
    }
    int compute_n_basis() const final {
        return static_cast<const Derived*>(this)->compute_n_basis_impl();
    }
};

}  // namespace dgfem