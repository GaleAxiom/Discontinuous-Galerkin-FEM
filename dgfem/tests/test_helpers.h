#pragma once

#include "dgfem/core/mesh.hpp"
#include "dgfem/boundary/conditions.hpp"
#include "dgfem/kokkos_math.hpp"
#include <memory>
#include <map>
#include <string>

// Inline function to prevent duplicate symbol errors
inline std::shared_ptr<dgfem::DGMesh> create_test_mesh() {
    dgfem::DView2 vertices("vertices", 4, 2);
    dgfem::set_row2(vertices, 0, dgfem::Vec2{0.0, 0.0});
    dgfem::set_row2(vertices, 1, dgfem::Vec2{1.0, 0.0});
    dgfem::set_row2(vertices, 2, dgfem::Vec2{1.0, 1.0});
    dgfem::set_row2(vertices, 3, dgfem::Vec2{0.0, 1.0});

    dgfem::IView2 elements("elements", 2, 3);
    elements(0, 0) = 0; elements(0, 1) = 1; elements(0, 2) = 2;
    elements(1, 0) = 0; elements(1, 1) = 2; elements(1, 2) = 3;

    dgfem::IView1 element_tags("element_tags", elements.extent(0));
    std::map<std::string, int> boundary_tags = {
        {"bottom", 1},
        {"right", 2},
        {"top", 3},
        {"left", 4}
    };

    // Boundary edges: map from physical tag to list of (element_id, local_face_id)
    std::map<int, std::vector<std::pair<int, int>>> boundary_edges = {
        {1, {{0, 0}}}, // bottom edge of element 0
        {2, {{0, 1}, {1, 1}}}, // right edge of elements 0 and 1
        {3, {{1, 2}}}, // top edge of element 1
        {4, {{0, 2}}}  // left edge of elements 0 and
    };

    // Create the DGMesh
    auto mesh = std::make_shared<dgfem::DGMesh>(vertices, elements, element_tags, boundary_tags, boundary_edges);

    // Set up boundary conditions for each face
    auto dirichlet = dgfem::make_dirichlet_bc(0.0);

    // Set boundary conditions for each face
    mesh->set_boundary_condition("bottom", dirichlet); // bottom edge of element 0
    mesh->set_boundary_condition("right", dirichlet);  // right edge of elements 0 and 1
    mesh->set_boundary_condition("top", dirichlet);    // top edge of element 1
    mesh->set_boundary_condition("left", dirichlet);   // left edge of elements 0 and

    return mesh;
}

