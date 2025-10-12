/**
 * @file test_vtk_writer.cpp
 * @brief Tests for VTK writer functionality
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
    
    // Helper to check if file exists and is non-empty
    bool file_exists_and_valid(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.good()) return false;
        
        std::string first_line;
        std::getline(file, first_line);
        return first_line.find("# vtk DataFile") != std::string::npos;
    }
    
    // Helper to count lines in file
    int count_lines(const std::string& filename) {
        std::ifstream file(filename);
        int count = 0;
        std::string line;
        while (std::getline(file, line)) {
            count++;
        }
        return count;
    }
};

TEST_F(VTKWriterTest, WriteMeshOnly) {
    // Create a simple triangular mesh
    auto mesh = MeshCreator::create_rectangular_mesh(0.2, true, 0.0, 1.0, 0.0, 1.0);
    
    // Write mesh to VTK
    std::string filename = "test_mesh_only";
    VTKWriter::write_mesh(mesh, filename);
    
    // Check file exists
    EXPECT_TRUE(file_exists_and_valid(filename + ".vtk"));
    
    // Clean up
    std::remove((filename + ".vtk").c_str());
}

TEST_F(VTKWriterTest, WriteTriangleSolution) {
    // Create triangular mesh
    auto mesh = MeshCreator::create_rectangular_mesh(0.15, true, 0.0, 1.0, 0.0, 1.0);
    
    // Initialize DG space
    auto dg_space = std::make_shared<DGSpace>("triangle", 2);
    mesh->initialize_dg_space(dg_space, 1);
    
    // Set up boundary conditions
    auto bc_zero = make_dirichlet_bc(0.0);
    mesh->set_boundary_condition("Bottom", bc_zero);
    mesh->set_boundary_condition("Top", bc_zero);
    mesh->set_boundary_condition("Left", bc_zero);
    mesh->set_boundary_condition("Right", bc_zero);
    
    // Define source function
    auto source_func = [](const Eigen::Vector2d& x) -> double {
        return 2.0 * (x[1] * (1 - x[1]) + x[0] * (1 - x[0]));
    };
    
    // Solve Laplace equation
    LaplaceDGSolver solver(mesh, 10.0);
    Eigen::VectorXd solution = solver.solve(source_func);
    
    // Write solution with different resolution factors
    for (int res : {1, 3, 5}) {
        std::string filename = "test_triangle_solution_res" + std::to_string(res);
        VTKWriter::write_solution(mesh, solution, filename, "u", res);
        
        EXPECT_TRUE(file_exists_and_valid(filename + ".vtk"));
        
        // Higher resolution should produce more lines
        int n_lines = count_lines(filename + ".vtk");
        EXPECT_GT(n_lines, 50);  // Should have significant content
        
        std::remove((filename + ".vtk").c_str());
    }
}

TEST_F(VTKWriterTest, WriteQuadSolution) {
    // Create quad mesh
    auto mesh = MeshCreator::create_rectangular_mesh(0.2, false, 0.0, 1.0, 0.0, 1.0);
    
    // Initialize DG space
    auto dg_space = std::make_shared<DGSpace>("quad", 1);
    mesh->initialize_dg_space(dg_space, 1);
    
    // Set up boundary conditions
    auto bc_zero = make_dirichlet_bc(0.0);
    mesh->set_boundary_condition("Bottom", bc_zero);
    mesh->set_boundary_condition("Top", bc_zero);
    mesh->set_boundary_condition("Left", bc_zero);
    mesh->set_boundary_condition("Right", bc_zero);
    
    // Define source function
    auto source_func = [](const Eigen::Vector2d& x) -> double {
        return 1.0;
    };
    
    // Solve
    LaplaceDGSolver solver(mesh, 10.0);
    Eigen::VectorXd solution = solver.solve(source_func);
    
    // Write solution
    std::string filename = "test_quad_solution";
    VTKWriter::write_solution(mesh, solution, filename, "temperature", 3);
    
    EXPECT_TRUE(file_exists_and_valid(filename + ".vtk"));
    
    // Check that file contains "temperature" variable
    std::ifstream file(filename + ".vtk");
    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    EXPECT_TRUE(content.find("SCALARS temperature") != std::string::npos);
    
    std::remove((filename + ".vtk").c_str());
}

TEST_F(VTKWriterTest, WriteWithAnalytical) {
    // Create mesh
    auto mesh = MeshCreator::create_rectangular_mesh(0.1, true, 0.0, 1.0, 0.0, 1.0);
    
    // Initialize DG space with higher order for better accuracy
    auto dg_space = std::make_shared<DGSpace>("triangle", 3);
    mesh->initialize_dg_space(dg_space, 1);
    
    // Set up boundary conditions
    auto bc_zero = make_dirichlet_bc(0.0);
    mesh->set_boundary_condition("Bottom", bc_zero);
    mesh->set_boundary_condition("Top", bc_zero);
    mesh->set_boundary_condition("Left", bc_zero);
    mesh->set_boundary_condition("Right", bc_zero);
    
    // Define problem with known solution: u = x(1-x)y(1-y)
    auto exact_solution = [](const Eigen::Vector2d& x) -> double {
        return x[0] * (1 - x[0]) * x[1] * (1 - x[1]);
    };
    
    auto source_func = [](const Eigen::Vector2d& x) -> double {
        return 2.0 * (x[1] * (1 - x[1]) + x[0] * (1 - x[0]));
    };
    
    // Solve
    LaplaceDGSolver solver(mesh, 10.0);
    Eigen::VectorXd solution = solver.solve(source_func);
    
    // Write with analytical comparison
    std::string filename = "test_with_analytical";
    VTKWriter::write_with_analytical(mesh, solution, exact_solution, filename, 4);
    
    EXPECT_TRUE(file_exists_and_valid(filename + ".vtk"));
    
    // Check that file contains all three fields
    std::ifstream file(filename + ".vtk");
    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    EXPECT_TRUE(content.find("SCALARS numerical") != std::string::npos);
    EXPECT_TRUE(content.find("SCALARS analytical") != std::string::npos);
    EXPECT_TRUE(content.find("SCALARS error") != std::string::npos);
    
    std::remove((filename + ".vtk").c_str());
}

TEST_F(VTKWriterTest, WriteMultipleVariables) {
    // Create mesh
    auto mesh = MeshCreator::create_rectangular_mesh(0.15, true, 0.0, 1.0, 0.0, 1.0);
    
    // Initialize DG space
    auto dg_space = std::make_shared<DGSpace>("triangle", 2);
    mesh->initialize_dg_space(dg_space, 1);
    
    int n_dofs_total = mesh->get_n_elements() * dg_space->get_n_dofs();
    
    // Create multiple solution fields
    Eigen::VectorXd solution1 = Eigen::VectorXd::Random(n_dofs_total);
    Eigen::VectorXd solution2 = Eigen::VectorXd::Random(n_dofs_total);
    Eigen::VectorXd solution3 = solution1 + 0.5 * solution2;
    
    std::vector<Eigen::VectorXd> solutions = {solution1, solution2, solution3};
    std::vector<std::string> names = {"density", "velocity", "pressure"};
    
    // Write multi-variable solution
    std::string filename = "test_multi_variable";
    VTKWriter::write_multi_variable_solution(mesh, solutions, names, filename, 2);
    
    EXPECT_TRUE(file_exists_and_valid(filename + ".vtk"));
    
    // Check that all variables are present
    std::ifstream file(filename + ".vtk");
    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    EXPECT_TRUE(content.find("SCALARS density") != std::string::npos);
    EXPECT_TRUE(content.find("SCALARS velocity") != std::string::npos);
    EXPECT_TRUE(content.find("SCALARS pressure") != std::string::npos);
    
    std::remove((filename + ".vtk").c_str());
}

TEST_F(VTKWriterTest, HighOrderVisualization) {
    // Test with high-order polynomial (order 5) and high resolution
    auto mesh = MeshCreator::create_rectangular_mesh(0.3, true, 0.0, 1.0, 0.0, 1.0);
    
    auto dg_space = std::make_shared<DGSpace>("triangle", 5);
    mesh->initialize_dg_space(dg_space, 1);
    
    // Set up a smooth polynomial solution
    int n_dofs_total = mesh->get_n_elements() * dg_space->get_n_dofs();
    Eigen::VectorXd solution = Eigen::VectorXd::Random(n_dofs_total);
    
    // Write with high resolution to capture polynomial features
    std::string filename = "test_high_order";
    VTKWriter::write_solution(mesh, solution, filename, "high_order_poly", 7);
    
    EXPECT_TRUE(file_exists_and_valid(filename + ".vtk"));
    
    // High resolution should produce many points
    int n_lines = count_lines(filename + ".vtk");
    EXPECT_GT(n_lines, 500);  // Should be quite large
    
    std::remove((filename + ".vtk").c_str());
}

TEST_F(VTKWriterTest, CheckVTKFormat) {
    // Create simple mesh
    auto mesh = MeshCreator::create_rectangular_mesh(0.25, true, 0.0, 1.0, 0.0, 1.0);
    auto dg_space = std::make_shared<DGSpace>("triangle", 1);
    mesh->initialize_dg_space(dg_space, 1);
    
    int n_dofs = mesh->get_n_elements() * dg_space->get_n_dofs();
    Eigen::VectorXd solution = Eigen::VectorXd::Ones(n_dofs);
    
    std::string filename = "test_vtk_format";
    VTKWriter::write_solution(mesh, solution, filename, "ones", 1);
    
    // Read and check VTK format
    std::ifstream file(filename + ".vtk");
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(file, line)) {
        lines.push_back(line);
    }
    
    ASSERT_GE(lines.size(), 5);
    EXPECT_EQ(lines[0], "# vtk DataFile Version 3.0");
    EXPECT_TRUE(lines[2] == "ASCII");
    EXPECT_TRUE(lines[3] == "DATASET UNSTRUCTURED_GRID");
    
    // Find POINTS line
    bool found_points = false;
    bool found_cells = false;
    bool found_point_data = false;
    
    for (const auto& l : lines) {
        if (l.find("POINTS") != std::string::npos) found_points = true;
        if (l.find("CELLS") != std::string::npos) found_cells = true;
        if (l.find("POINT_DATA") != std::string::npos) found_point_data = true;
    }
    
    EXPECT_TRUE(found_points);
    EXPECT_TRUE(found_cells);
    EXPECT_TRUE(found_point_data);
    
    std::remove((filename + ".vtk").c_str());
}

TEST_F(VTKWriterTest, SolutionValuesCorrect) {
    // Create a very simple mesh with known solution
    auto mesh = MeshCreator::create_rectangular_mesh(0.5, true, 0.0, 1.0, 0.0, 1.0);
    
    // Use P0 (constant) elements for simplicity
    auto dg_space = std::make_shared<DGSpace>("triangle", 0);
    mesh->initialize_dg_space(dg_space, 1);
    
    // Create constant solution = 1.0 everywhere
    int n_elems = mesh->get_n_elements();
    Eigen::VectorXd solution = Eigen::VectorXd::Constant(n_elems, 2.5);
    
    std::string filename = "test_constant_solution";
    VTKWriter::write_solution(mesh, solution, filename, "constant", 2);
    
    // Read the file and check that all values are approximately 2.5
    std::ifstream file(filename + ".vtk");
    std::string line;
    bool in_data_section = false;
    std::vector<double> values;
    
    while (std::getline(file, line)) {
        if (line.find("LOOKUP_TABLE") != std::string::npos) {
            in_data_section = true;
            continue;
        }
        if (in_data_section && !line.empty()) {
            try {
                double val = std::stod(line);
                values.push_back(val);
            } catch (...) {
                // Not a number, might be end of section
                break;
            }
        }
    }
    
    // All values should be close to 2.5
    for (double val : values) {
        EXPECT_NEAR(val, 2.5, 1e-8);
    }
    
    std::remove((filename + ".vtk").c_str());
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
