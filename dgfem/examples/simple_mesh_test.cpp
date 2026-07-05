/**
 * @file simple_mesh_test.cpp
 * @brief Simple test of DGFEM C++ implementation
 */

#include <Kokkos_Core.hpp>

#include <iostream>
#include <memory>

// Include the main DGFEM headers
#include "dgfem/config/config.hpp"
#include "dgfem/core/mesh.hpp"
#include "dgfem/core/space.hpp"
#include "dgfem/kokkos_math.hpp"
#include "dgfem/utils/example_helpers.hpp"

#include <cstdlib>

int main(int argc, char** argv) {
    Kokkos::ScopeGuard kokkos_guard(argc, argv);
    try {
        std::cout << "=== DGFEM C++ Simple Test ===" << std::endl;

        // Report configuration defaults before use
        auto& config = dgfem::Config::instance();
        std::cout << "Using element order: " << config.order << std::endl;
        std::cout << "Using triangles: " << (config.use_triangles ? "true" : "false") << std::endl;

        // Create mesh + DG space (1 variable, e.g. for Laplace) in one call
        std::cout << "\nCreating rectangular mesh..." << std::endl;
        auto mesh = dgfem::MeshSetup::create_standard_mesh(
            config.use_triangles, config.order, config.dx, /*n_vars=*/1, 0.0, 1.0, 0.0, 1.0);
        auto dg_space = mesh->get_dg_space();

        dgfem::MeshSetup::print_info(mesh);

        // Test solution storage
        auto solution = mesh->get_solution();
        std::cout << "Solution storage created with " << solution->get_total_dofs() << " total DOFs"
                  << std::endl;

        // Set some test values
        int n_basis_test = dg_space->get_basis()->get_n_basis();
        dgfem::DView1 test_coeffs("test_coeffs", n_basis_test);
        for (int i = 0; i < n_basis_test; ++i) {
            test_coeffs(i) = static_cast<double>(std::rand()) / RAND_MAX;
        }
        solution->set_element_coeffs(0, 0, test_coeffs);

        // Retrieve and verify
        dgfem::DView1 retrieved_coeffs = solution->get_element_coeffs(0, 0);
        dgfem::DView1 diff("diff", n_basis_test);
        for (int i = 0; i < n_basis_test; ++i) {
            diff(i) = test_coeffs(i) - retrieved_coeffs(i);
        }
        std::cout << "Coefficient storage test: "
                  << (dgfem::norm(diff) < 1e-14 ? "PASSED" : "FAILED") << std::endl;

        // Test basis functions
        std::cout << "\nTesting basis functions..." << std::endl;
        auto basis = dg_space->get_basis();
        dgfem::Vec2 test_point;

        if (mesh->get_element_type() == "triangle") {
            test_point = dgfem::Vec2{0.3, 0.2};  // Point in reference triangle
        } else {
            test_point = dgfem::Vec2{0.1, -0.5};  // Point in reference quad
        }

        dgfem::DView1 phi = basis->evaluate(test_point);
        dgfem::DView2 grad_phi = basis->evaluate_gradient(test_point);

        std::cout << "Basis evaluation at test point: ";
        for (int i = 0; i < static_cast<int>(phi.extent(0)); ++i) {
            std::cout << phi(i) << " ";
        }
        std::cout << std::endl;
        std::cout << "Gradient dimensions: " << grad_phi.extent(0) << "x" << grad_phi.extent(1)
                  << std::endl;

        // Test quadrature
        auto vol_quad = dg_space->get_volume_quad();
        auto face_quad = dg_space->get_face_quad();

        std::cout << "\nQuadrature rules:" << std::endl;
        std::cout << "Volume quadrature points: " << vol_quad->size() << std::endl;
        std::cout << "Face quadrature points: " << face_quad->size() << std::endl;

        // Test that weights sum to reference element measure
        double vol_weights_sum = 0.0;
        for (int i = 0; i < static_cast<int>(vol_quad->weights.extent(0)); ++i) {
            vol_weights_sum += vol_quad->weights(i);
        }
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