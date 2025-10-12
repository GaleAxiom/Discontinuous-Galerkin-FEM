/**
 * @file factory.hpp
 * @brief Quadrature rules for numerical integration
 */

#pragma once

#include <Eigen/Dense>
#include <vector>
#include <memory>

namespace dgfem {

/**
 * @brief Quadrature rule container
 */
struct QuadratureRule {
    Eigen::MatrixXd points;    ///< Quadrature points (n_points x dim)
    Eigen::VectorXd weights;   ///< Quadrature weights (n_points)
    
    // Constructor with move semantics
    QuadratureRule(Eigen::MatrixXd pts, Eigen::VectorXd wts)
        : points(std::move(pts)), weights(std::move(wts)) {}
    
    // Delete copy, default move
    QuadratureRule(const QuadratureRule&) = delete;
    QuadratureRule& operator=(const QuadratureRule&) = delete;
    QuadratureRule(QuadratureRule&&) noexcept = default;
    QuadratureRule& operator=(QuadratureRule&&) noexcept = default;
    
    [[nodiscard]] int size() const noexcept { return static_cast<int>(weights.size()); }
    [[nodiscard]] int dimension() const noexcept { return static_cast<int>(points.cols()); }
    
    // Validate the rule
    [[nodiscard]] bool is_valid() const noexcept {
        return points.rows() == weights.size() && points.rows() > 0;
    }
};

/**
 * @brief Factory class for creating quadrature rules
 */
class QuadratureFactory {
public:
    // Maximum supported Gauss-Legendre points
    static constexpr int MAX_GL_POINTS = 8;
    // Maximum supported Dunavant order
    static constexpr int MAX_DUNAVANT_ORDER = 14;
    
    /**
     * @brief Create 1D Gauss-Legendre quadrature rule
     * @param n_points Number of quadrature points (1-4)
     * @return Quadrature rule on [-1, 1]
     */
    [[nodiscard]] static std::unique_ptr<QuadratureRule> gauss_legendre_1d(int n_points);
    
    /**
     * @brief Create 2D Gauss-Legendre quadrature rule for quadrilaterals
     * @param n_points_1d Number of points in each direction
     * @return Tensor product quadrature rule on [-1, 1]^2
     */
    [[nodiscard]] static std::unique_ptr<QuadratureRule> gauss_legendre_quad(int n_points_1d);
    
    /**
     * @brief Create Dunavant quadrature rule for triangles
     * @param order Polynomial order to integrate exactly (0-5)
     * @return Quadrature rule on reference triangle
     */
    [[nodiscard]] static std::unique_ptr<QuadratureRule> dunavant_triangle(int order);
    
    // Validate inputs
    [[nodiscard]] static constexpr bool is_valid_gl_order(int n) noexcept {
        return n >= 1 && n <= MAX_GL_POINTS;
    }
    
    [[nodiscard]] static constexpr bool is_valid_dunavant_order(int order) noexcept {
        return order >= 0 && order <= MAX_DUNAVANT_ORDER;
    }
    
private:
    /**
     * @brief Compute Gauss-Legendre points and weights
     */
    static void compute_gauss_legendre(int n, Eigen::VectorXd& points, Eigen::VectorXd& weights);
    
    /**
     * @brief Get pre-computed Dunavant rules
     */
    static void get_dunavant_rule(int order, Eigen::MatrixXd& points, Eigen::VectorXd& weights);
};

} // namespace dgfem