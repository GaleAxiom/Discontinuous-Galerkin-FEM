/**
 * @file vtk_writer_test.cpp
 * @brief Basic tests for VTK writer functionality
 */

#include <gtest/gtest.h>
#include "dgfem/utils/vtk_writer.hpp"
#include "dgfem/utils/mesh_creation.hpp"
#include "dgfem/core/mesh.hpp"
#include "dgfem/core/space.hpp"
#include "dgfem/solver/dg_solver.hpp"
#include "dgfem/boundary/conditions.hpp"

#include <fstream>
#include <cmath>

using namespace dgfem;

class VTKWriterTest : public ::testing::Test {
protected:
    void SetUp() override {
        MeshCreator::initialize_gmsh();
    }
    
    void TearDown() override {
        MeshCreator::finalize_gmsh();
    }
    
    bool file_exists_and_valid(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.good()) return false;
        
        std::string first_line;
        std::getline(file, first_line);
        return first_line.find("# vtk DataFile") != std::string::npos;
    }
};

TEST_F(VTKWriterTest, WriteMeshOnly) {
    auto mesh = MeshCreator::create_rectangular_mesh(0.2, true, 0.0, 1.0, 0.0, 1.0);
    
    std::string filename = "test_mesh_only";
    VTKWriter::write_mesh(mesh, filename);
    
    EXPECT_TRUE(file_exists_and_valid(filename + ".vtk"));
    std::remove((filename + ".vtk").c_str());
}

TEST_F(VTKWriterTest, WriteSimpleTriangleSolution) {
    auto mesh = MeshCreator::create_rectangular_mesh(0.15, true, 0.0, 1.0, 0.0, 1.0);
    auto dg_space = std::make_shared<DGSpace>("triangle", 2);
    mesh->initialize_dg_space(dg_space, 1);
    
    auto bc_zero = make_dirichlet_bc(0.0);
    mesh->set_boundary_condition("Bottom", bc_zero);
    mesh->set_boundary_condition("Top", bc_zero);
    mesh->set_boundary_condition("Left", bc_zero);
    mesh->set_boundary_condition("Right", bc_zero);
    
    auto source_func = [](const Eigen::Vector2d& x) -> double {
        return 2.0 * (x[1] * (1 - x[1]) + x[0] * (1 - x[0]));
    };
    
    LaplaceDGSolver solver(mesh, 10.0);
    Eigen::VectorXd solution = solver.solve(source_func);
    
    std::string filename = "test_triangle_solution";
    VTKWriter::write_solution(mesh, solution, filename, "u", 3);
    
    EXPECT_TRUE(file_exists_and_valid(filename + ".vtk"));
    std::remove((filename + ".vtk").c_str());
}

TEST_F(VTKWriterTest, WriteQuadSolution) {
    auto mesh = MeshCreator::create_rectangular_mesh(0.2, false, 0.0, 1.0, 0.0, 1.0);
    auto dg_space = std::make_shared<DGSpace>("quad", 1);
    mesh->initialize_dg_space(dg_space, 1);
    
    auto bc_zero = make_dirichlet_bc(0.0);
    mesh->set_boundary_condition("Bottom", bc_zero);
    mesh->set_boundary_condition("Top", bc_zero);
    mesh->set_boundary_condition("Left", bc_zero);
    mesh->set_boundary_condition("Right", bc_zero);
    
    auto source_func = [](const Eigen::Vector2d& x) -> double {
        return 1.0;
    };
    
    LaplaceDGSolver solver(mesh, 10.0);
    Eigen::VectorXd solution = solver.solve(source_func);
    
    std::string filename = "test_quad_solution";
    VTKWriter::write_solution(mesh, solution, filename, "temperature", 3);
    
    EXPECT_TRUE(file_exists_and_valid(filename + ".vtk"));
    std::remove((filename + ".vtk").c_str());
}
