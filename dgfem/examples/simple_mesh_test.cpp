/**
 * @file simple_mesh_test.cpp
 * @brief Simple test of DGFEM C++ implementation
 */

#include <iostream>
#include <memory>

// Include the main DGFEM headers
#include "dgfem/config/config.hpp"
#include "dgfem/core/mesh.hpp"
#include "dgfem/core/space.hpp"
#include "dgfem/utils/mesh_creation.hpp"

int main() {
    try {
        std::cout << "=== DGFEM C++ Simple Test ===" << std::endl;

        // Initialize GMSH
        dgfem::MeshCreator::initialize_gmsh();

        // Get configuration
        auto& config = dgfem::Config::instance();
        std::cout << "Using element order: " << config.order << std::endl;
        std::cout << "Using triangles: " << (config.use_triangles ? "true" : "false") << std::endl;

        // Create a simple rectangular mesh
        std::cout << "\nCreating rectangular mesh..." << std::endl;
        auto mesh = dgfem::MeshCreator::create_rectangular_mesh(config.dx, config.use_triangles,
                                                                0.0, 1.0, 0.0, 1.0);

        std::cout << "Mesh created with " << mesh->get_n_elements() << " "
                  << mesh->get_element_type() << " elements" << std::endl;

        // Create DG space
        std::cout << "\nInitializing DG space..." << std::endl;
        auto dg_space = std::make_shared<dgfem::DGSpace>(mesh->get_element_type(), config.order);

        // Initialize mesh with DG space
        mesh->initialize_dg_space(dg_space, 1);  // 1 variable (e.g., for Laplace)

        std::cout << "DG space initialized with " << dg_space->get_basis()->get_n_basis()
                  << " basis functions per element" << std::endl;

        // Test solution storage
        auto solution = mesh->get_solution();
        std::cout << "Solution storage created with " << solution->get_total_dofs() << " total DOFs"
                  << std::endl;

        // Set some test values
        Eigen::VectorXd test_coeffs = Eigen::VectorXd::Random(dg_space->get_basis()->get_n_basis());
        solution->set_element_coeffs(0, 0, test_coeffs);

        // Retrieve and verify
        Eigen::VectorXd retrieved_coeffs = solution->get_element_coeffs(0, 0);
        std::cout << "Coefficient storage test: "
                  << ((test_coeffs - retrieved_coeffs).norm() < 1e-14 ? "PASSED" : "FAILED")
                  << std::endl;

        // Test basis functions
        std::cout << "\nTesting basis functions..." << std::endl;
        auto basis = dg_space->get_basis();
        Eigen::Vector2d test_point;

        if (mesh->get_element_type() == "triangle") {
            test_point << 0.3, 0.2;  // Point in reference triangle
        } else {
            test_point << 0.1, -0.5;  // Point in reference quad
        }

        Eigen::VectorXd phi = basis->evaluate(test_point);
        Eigen::MatrixXd grad_phi = basis->evaluate_gradient(test_point);

        std::cout << "Basis evaluation at test point: " << phi.transpose() << std::endl;
        std::cout << "Gradient dimensions: " << grad_phi.rows() << "x" << grad_phi.cols()
                  << std::endl;

        // Test quadrature
        auto vol_quad = dg_space->get_volume_quad();
        auto face_quad = dg_space->get_face_quad();

        std::cout << "\nQuadrature rules:" << std::endl;
        std::cout << "Volume quadrature points: " << vol_quad->size() << std::endl;
        std::cout << "Face quadrature points: " << face_quad->size() << std::endl;

        // Test that weights sum to reference element measure
        double vol_weights_sum = vol_quad->weights.sum();
        double expected_measure = (mesh->get_element_type() == "triangle") ? 0.5 : 4.0;
        std::cout << "Volume weights sum: " << vol_weights_sum << " (expected: " << expected_measure
                  << ")" << std::endl;

        dgfem::MeshCreator::finalize_gmsh();

        std::cout << "\n=== Test completed successfully! ===" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        dgfem::MeshCreator::finalize_gmsh();
        return 1;
    }
}