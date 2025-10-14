/**
 * @file space.hpp
 * @brief DG space definition and element data computation
 */

#pragma once

#include "dgfem/basis/orthogonal.hpp"
#include "dgfem/quadrature/factory.hpp"
#include "dgfem/reference/elements.hpp"
#include "dgfem/reference/mapping.hpp"

#include <Eigen/Dense>

#include <map>
#include <memory>
#include <string>
#include <string_view>

namespace dgfem {

/**
 * @brief Main DG space class that combines all components
 */
class DGSpace {
public:
    DGSpace(std::string_view element_type, int order);

    // Delete copy, default move
    DGSpace(const DGSpace&) = delete;
    DGSpace& operator=(const DGSpace&) = delete;
    DGSpace(DGSpace&&) noexcept = default;
    DGSpace& operator=(DGSpace&&) noexcept = default;

    /**
     * @brief Compute element data for physical element
     * @param vertices Physical element vertices
     * @param face_neighbors Neighbor information for each face
     * @return Element data dictionary
     */
    [[nodiscard]] std::map<std::string, Eigen::MatrixXd>
    compute_element_data(const Eigen::MatrixXd& vertices,
                         const std::vector<std::pair<int, int>>& face_neighbors = {}) const;

    /**
     * @brief Compute face data for element face
     */
    [[nodiscard]] std::map<std::string, Eigen::VectorXd>
    compute_face_data(const Eigen::MatrixXd& vertices, int face_id,
                      const std::pair<int, int>& neighbor_info = {-1, -1}) const;

    /**
     * @brief Get precomputed basis values at volume quadrature points
     */
    [[nodiscard]] const Eigen::MatrixXd& get_volume_basis_values() const noexcept {
        return phi_vol_;
    }

    /**
     * @brief Get precomputed basis gradients at volume quadrature points
     */
    [[nodiscard]] const std::vector<Eigen::MatrixXd>& get_volume_basis_gradients() const noexcept {
        return dphi_vol_;
    }

    /**
     * @brief Get precomputed basis values at face quadrature points
     */
    [[nodiscard]] const std::vector<Eigen::MatrixXd>& get_face_basis_values() const noexcept {
        return phi_face_;
    }

    /**
     * @brief Compute face permutation for neighboring elements
     */
    [[nodiscard]] Eigen::VectorXi
    compute_face_permutation(int elem1_face, const Eigen::MatrixXd& elem1_vertices, int elem2_face,
                             const Eigen::MatrixXd& elem2_vertices) const;

    /**
     * @brief Map 1D face quadrature point to reference element face
     */
    [[nodiscard]] Eigen::Vector2d map_face_quad_point(int face_id, double s) const;

    /**
     * @brief Get the mapping from local DoF index to local vertex index.
     * Assumes a P1 nodal basis where DoFs correspond directly to vertices.
     */
    [[nodiscard]] std::vector<int> get_dof_to_vertex_map() const;

    // Getters
    [[nodiscard]] int get_order() const noexcept { return order_; }
    [[nodiscard]] int get_n_dofs() const noexcept { return basis_->get_n_basis(); }
    [[nodiscard]] const std::string& get_element_type() const noexcept { return element_type_; }
    [[nodiscard]] std::shared_ptr<ReferenceElement> get_ref_element() const noexcept {
        return ref_element_;
    }
    [[nodiscard]] std::shared_ptr<OrthogonalBasis> get_basis() const noexcept { return basis_; }
    [[nodiscard]] std::shared_ptr<QuadratureRule> get_volume_quad() const noexcept {
        return volume_quad_;
    }
    [[nodiscard]] std::shared_ptr<QuadratureRule> get_face_quad() const noexcept {
        return face_quad_;
    }
    [[nodiscard]] std::shared_ptr<GeometricMapping> get_mapping() const noexcept {
        return mapping_;
    }

private:
    int order_;
    std::string element_type_;

    // Components
    std::shared_ptr<ReferenceElement> ref_element_;
    std::shared_ptr<OrthogonalBasis> basis_;
    std::shared_ptr<QuadratureRule> volume_quad_;
    std::shared_ptr<QuadratureRule> face_quad_;
    std::shared_ptr<GeometricMapping> mapping_;

    // Precomputed basis values
    Eigen::MatrixXd phi_vol_;                ///< Volume basis values
    std::vector<Eigen::MatrixXd> dphi_vol_;  ///< Volume basis gradients
    std::vector<Eigen::MatrixXd> phi_face_;  ///< Face basis values

    /**
     * @brief Precompute basis values at quadrature points
     */
    void precompute_basis_values();

    /**
     * @brief Create appropriate basis for element type
     */
    [[nodiscard]] std::shared_ptr<OrthogonalBasis> create_basis(std::string_view element_type,
                                                                int order) const;

    /**
     * @brief Create appropriate reference element
     */
    [[nodiscard]] std::shared_ptr<ReferenceElement>
    create_ref_element(std::string_view element_type) const;
};

}  // namespace dgfem