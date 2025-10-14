/**
 * @file mesh.hpp
 * @brief DG mesh handling and connectivity
 */

#pragma once

#include <Eigen/Dense>
#include <optional>

#include <map>
#include <memory>
#include <string_view>
#include <tuple>
#include <vector>

#include "solution.hpp"

namespace dgfem {

// Forward declarations
class DGSpace;
class BoundaryCondition;
class BoundaryConditionEuler;

/**
 * @brief Container for mesh data with DG-specific connectivity information
 */
class DGMesh {
public:
    DGMesh(const Eigen::MatrixXd& vertices, const Eigen::MatrixXi& elements,
           const Eigen::VectorXi& element_tags, const std::map<std::string, int>& boundary_tags,
           const std::map<int, std::vector<std::pair<int, int>>>& boundary_edges);

    // Delete copy, default move
    DGMesh(const DGMesh&) = delete;
    DGMesh& operator=(const DGMesh&) = delete;
    DGMesh(DGMesh&&) noexcept = default;
    DGMesh& operator=(DGMesh&&) noexcept = default;

    /**
     * @brief Initialize DG space and allocate solution storage
     */
    void initialize_dg_space(std::shared_ptr<DGSpace> dg_space, int n_variables = 1);

    /**
     * @brief Get element neighbors for given element
     * @param elem_id Element index
     * @return Vector of (neighbor_elem, neighbor_face) pairs
     */
    [[nodiscard]] std::vector<std::pair<int, int>> get_element_neighbors(int elem_id) const;

    /**
     * @brief Get element data (computed by DG space)
     */
    [[nodiscard]] const std::map<std::string, Eigen::MatrixXd>& get_element_data(int elem_id) const;

    /**
     * @brief Get face data for specific element face
     */
    [[nodiscard]] const std::map<std::string, Eigen::VectorXd>& get_face_data(int elem_id,
                                                                              int face_id) const;

    /**
     * @brief Get element face data (returns map with MatrixXd for compatibility)
     */
    [[nodiscard]] const std::map<std::string, Eigen::MatrixXd>&
    get_element_face_data(int elem_id, int face_id) const;

    /**
     * @brief Set boundary condition for a boundary tag
     */
    void set_boundary_condition(std::string_view tag_name, std::shared_ptr<BoundaryCondition> bc);

    /**
     * @brief Get boundary condition for element face
     */
    [[nodiscard]] std::shared_ptr<BoundaryCondition> get_boundary_condition(int elem_id,
                                                                            int face_id) const;

    /**
     * @brief Set Euler boundary condition for a boundary tag
     */
    void set_boundary_condition_euler(std::string_view tag_name,
                                      std::shared_ptr<BoundaryConditionEuler> bc);

    /**
     * @brief Get Euler boundary condition for element face
     */
    [[nodiscard]] std::shared_ptr<BoundaryConditionEuler>
    get_boundary_condition_euler(int elem_id, int face_id) const;

    /**
     * @brief Get Euler boundary condition by boundary tag name
     */
    [[nodiscard]] std::shared_ptr<BoundaryConditionEuler>
    get_boundary_condition_euler(std::string_view tag_name) const;

    /**
     * @brief Set periodic boundary conditions between two boundaries
     */
    void set_periodic_boundaries(std::string_view tag_name_1, std::string_view tag_name_2);

    /**
     * @brief Get boundary tag name for a boundary face
     */
    [[nodiscard]] std::string get_face_boundary_tag(int elem_id, int face_id) const;

    /**
     * @brief Check if face is on boundary
     */
    [[nodiscard]] bool is_boundary_face(int elem_id, int face_id) const noexcept;

    // Getters with [[nodiscard]]
    [[nodiscard]] const Eigen::MatrixXd& get_vertices() const noexcept { return vertices_; }
    [[nodiscard]] const Eigen::MatrixXi& get_elements() const noexcept { return elements_; }
    [[nodiscard]] const Eigen::VectorXi& get_element_tags() const noexcept { return element_tags_; }
    [[nodiscard]] const std::map<std::string, int>& get_boundary_tags() const noexcept {
        return boundary_tags_;
    }

    [[nodiscard]] int get_n_elements() const noexcept { return n_elements_; }
    [[nodiscard]] const std::string& get_element_type() const noexcept { return element_type_; }
    [[nodiscard]] int get_n_faces_per_element() const noexcept { return n_faces_per_elem_; }

    [[nodiscard]] std::shared_ptr<DGSpace> get_dg_space() const noexcept { return dg_space_; }
    [[nodiscard]] std::shared_ptr<DGSolution> get_solution() const noexcept { return solution_; }

    /**
     * @brief Get boundary faces (elem_id, face_id) pairs
     */
    [[nodiscard]] const std::vector<std::pair<int, int>>& get_boundary_faces() const noexcept {
        return boundary_faces_;
    }

    /**
     * @brief Get map from node index to contributing elements
     */
    [[nodiscard]] const std::map<int, std::vector<std::pair<int, int>>>&
    get_node_to_elements() const noexcept {
        return node_to_elements_;
    }

    /**
     * @brief Precomputed face data structure for efficient assembly
     */
    struct FaceConnectivity {
        int elem_L, elem_R;
        int face_L, face_R;
        Eigen::VectorXi permutation;
        Eigen::MatrixXd vertices_L, vertices_R;
        std::shared_ptr<BoundaryConditionEuler> bc_euler;  // For boundary faces
        std::string bc_tag;
        bool is_boundary;

        FaceConnectivity() : elem_L(-1), elem_R(-1), face_L(-1), face_R(-1), is_boundary(false) {}
    };

    /**
     * @brief Build precomputed face connectivity for fast assembly
     */
    void build_precomputed_faces();

    /**
     * @brief Get precomputed interior faces
     */
    [[nodiscard]] const std::vector<FaceConnectivity>& get_interior_faces() const noexcept {
        return interior_faces_;
    }

    /**
     * @brief Get precomputed boundary faces
     */
    [[nodiscard]] const std::vector<FaceConnectivity>& get_boundary_face_data() const noexcept {
        return boundary_face_data_;
    }

private:
    /**
     * @brief Build face-to-element connectivity
     */
    void build_face_connectivity();

    /**
     * @brief Identify boundary faces and their tags
     */
    void
    identify_boundary_faces(const std::map<int, std::vector<std::pair<int, int>>>& boundary_edges);

    // Mesh data
    Eigen::MatrixXd vertices_;
    Eigen::MatrixXi elements_;
    Eigen::VectorXi element_tags_;
    std::map<std::string, int> boundary_tags_;

    int n_elements_;
    std::string element_type_;
    int n_faces_per_elem_;

    // Connectivity
    Eigen::MatrixXi face_neighbors_;  ///< (n_elements x n_faces x 2) neighbor info
    std::vector<std::pair<int, int>> boundary_faces_;
    std::map<int, std::vector<std::pair<int, int>>>
        node_to_elements_;  ///< node_id -> list of (elem_id, local_node_id)

    // Precomputed face data for fast assembly
    std::vector<FaceConnectivity> interior_faces_;
    std::vector<FaceConnectivity> boundary_face_data_;

    // DG data
    std::shared_ptr<DGSpace> dg_space_;
    std::shared_ptr<DGSolution> solution_;

    // Element and face data computed by DG space
    std::vector<std::map<std::string, Eigen::MatrixXd>> element_data_;
    std::vector<std::vector<std::map<std::string, Eigen::VectorXd>>> face_data_;
    std::vector<std::vector<std::map<std::string, Eigen::MatrixXd>>> face_data_matrix_;

    // Boundary conditions
    std::map<std::string, std::shared_ptr<BoundaryCondition>> boundary_conditions_;
    std::map<std::string, std::shared_ptr<BoundaryConditionEuler>> boundary_conditions_euler_;
    std::map<std::pair<int, int>, int> boundary_face_tags_;
    std::map<std::pair<int, int>, std::string> boundary_face_tag_names_;
    std::map<std::pair<int, int>, std::vector<int>> face_to_vertices_;

    // Periodic boundary mapping: (elem_id, face_id) -> (periodic_elem_id, periodic_face_id)
    std::map<std::pair<int, int>, std::pair<int, int>> periodic_face_map_;
};

}  // namespace dgfem