/**
 * @file gmsh_vtk_writer_test.cpp
 * @brief Advanced tests for VTK writer with GMSH meshes
 */

#include "dgfem/boundary/conditions.hpp"
#include "dgfem/core/mesh.hpp"
#include "dgfem/core/space.hpp"
#include "dgfem/kokkos_math.hpp"
#include "dgfem/solver/dg_solver.hpp"
#include "dgfem/utils/mesh_creation.hpp"
#include "dgfem/utils/vtk_writer.hpp"

#include <cmath>
#include <cstdlib>

#include <fstream>

#include <gtest/gtest.h>

using namespace dgfem;

class GMSHVTKWriterTest : public ::testing::Test {
protected:
    void SetUp() override { MeshCreator::initialize_gmsh(); }

    void TearDown() override { MeshCreator::finalize_gmsh(); }
};

TEST_F(GMSHVTKWriterTest, WriteWithAnalyticalComparison) {
    auto mesh = MeshCreator::create_rectangular_mesh(0.1, true, 0.0, 1.0, 0.0, 1.0);
    auto dg_space = std::make_shared<DGSpace>("triangle", 2);
    mesh->initialize_dg_space(dg_space, 1);

    auto bc_zero = make_dirichlet_bc(0.0);
    mesh->set_boundary_condition("Bottom", bc_zero);
    mesh->set_boundary_condition("Top", bc_zero);
    mesh->set_boundary_condition("Left", bc_zero);
    mesh->set_boundary_condition("Right", bc_zero);

    auto exact_solution = [](const Vec2& x) -> double {
        return x[0] * (1 - x[0]) * x[1] * (1 - x[1]);
    };

    auto source_func = [](const Vec2& x) -> double {
        return 2.0 * (x[1] * (1 - x[1]) + x[0] * (1 - x[0]));
    };

    LaplaceDGSolver solver(mesh, 10.0);
    DView1 solution = solver.solve(source_func);

    std::string filename = "gtest_vtk_writer_gmsh_output";
    VTKWriter::write_with_analytical(mesh, solution, exact_solution, filename, 4);

    // Check file exists
    std::ifstream file(filename + ".vtk");
    ASSERT_TRUE(file.good());

    // Check file contains all expected fields
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    EXPECT_TRUE(content.find("SCALARS numerical") != std::string::npos);
    EXPECT_TRUE(content.find("SCALARS analytical") != std::string::npos);
    EXPECT_TRUE(content.find("SCALARS error") != std::string::npos);

    // Note: We keep this file for manual inspection in ParaView
    std::cout << "VTK output written to: " << filename << ".vtk" << std::endl;
    std::cout << "You can open this in ParaView to visualize the solution!" << std::endl;
}

TEST_F(GMSHVTKWriterTest, MultiVariableOutput) {
    auto mesh = MeshCreator::create_rectangular_mesh(0.15, true, 0.0, 1.0, 0.0, 1.0);
    auto dg_space = std::make_shared<DGSpace>("triangle", 2);
    mesh->initialize_dg_space(dg_space, 1);

    int n_dofs = mesh->get_n_elements() * dg_space->get_n_dofs();

    // Create multiple fields
    DView1 field1("field1", n_dofs);
    DView1 field2("field2", n_dofs);
    DView1 field3("field3", n_dofs);
    for (int i = 0; i < n_dofs; ++i) {
        field1(i) = static_cast<double>(std::rand()) / RAND_MAX;
        field2(i) = static_cast<double>(std::rand()) / RAND_MAX;
        field3(i) = field1(i) * field2(i);
    }

    std::vector<DView1> solutions = {field1, field2, field3};
    std::vector<std::string> names = {"rho", "u", "p"};

    std::string filename = "test_multi_field";
    VTKWriter::write_multi_variable_solution(mesh, solutions, names, filename, 2);

    std::ifstream file(filename + ".vtk");
    ASSERT_TRUE(file.good());

    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    EXPECT_TRUE(content.find("SCALARS rho") != std::string::npos);
    EXPECT_TRUE(content.find("SCALARS u") != std::string::npos);
    EXPECT_TRUE(content.find("SCALARS p") != std::string::npos);

    std::remove((filename + ".vtk").c_str());
}
