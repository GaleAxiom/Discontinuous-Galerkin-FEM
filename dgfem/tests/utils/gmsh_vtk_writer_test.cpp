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

#include <cctype>
#include <cmath>
#include <cstdlib>

#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

using namespace dgfem;

namespace {
// Legacy VTK ASCII format writes each named SCALARS field as a "SCALARS <name> ..." line,
// followed by a "LOOKUP_TABLE ..." line, followed by one value per line until the next
// non-numeric line. Parsing this (rather than only checking the field name appears in the
// file) lets tests catch a writer that emits correct headers but wrong/garbage data.
std::vector<double> read_vtk_scalar_field(const std::string& path, const std::string& field_name) {
    std::ifstream file(path);
    std::string line;
    std::vector<double> values;
    bool in_field = false;
    bool past_lookup_table = false;
    while (std::getline(file, line)) {
        if (line.rfind("SCALARS " + field_name + " ", 0) == 0) {
            in_field = true;
            past_lookup_table = false;
            continue;
        }
        if (in_field && !past_lookup_table) {
            if (line.rfind("LOOKUP_TABLE", 0) == 0) {
                past_lookup_table = true;
            }
            continue;
        }
        if (in_field && past_lookup_table) {
            if (line.empty() || (!std::isdigit(static_cast<unsigned char>(line[0])) &&
                                 line[0] != '-' && line[0] != '+')) {
                break;
            }
            values.push_back(std::stod(line));
        }
    }
    return values;
}
}  // namespace

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

    // Parse the actual written values, not just header presence: a writer that emits correct
    // headers but wrong/garbage data would still pass the substring checks above.
    auto analytical_values = read_vtk_scalar_field(filename + ".vtk", "analytical");
    auto error_values = read_vtk_scalar_field(filename + ".vtk", "error");
    ASSERT_FALSE(analytical_values.empty());
    ASSERT_EQ(analytical_values.size(), error_values.size());

    // exact_solution = x(1-x)y(1-y) is 0 on the domain boundary and reaches its max of
    // 0.5*0.5*0.5*0.5 = 0.0625 at the center -- every written "analytical" value must fall in
    // that range regardless of where it was sampled.
    for (double v : analytical_values) {
        EXPECT_GE(v, -1e-9);
        EXPECT_LE(v, 0.0625 + 1e-9);
    }
    // Order-2 DG can't exactly represent this biquadratic (total degree 4) solution, but the
    // error should still be small and bounded, not garbage -- 1e-2 is a generous margin above
    // what an order-2 space on this mesh actually achieves.
    for (double e : error_values) {
        EXPECT_LT(std::abs(e), 1e-2);
    }

    // Note: We keep this file for manual inspection in ParaView
    std::cout << "VTK output written to: " << filename << ".vtk" << std::endl;
    std::cout << "You can open this in ParaView to visualize the solution!" << std::endl;
}

TEST_F(GMSHVTKWriterTest, MultiVariableOutput) {
    auto mesh = MeshCreator::create_rectangular_mesh(0.15, true, 0.0, 1.0, 0.0, 1.0);
    auto dg_space = std::make_shared<DGSpace>("triangle", 2);
    mesh->initialize_dg_space(dg_space, 1);

    int n_dofs_per_elem = dg_space->get_n_dofs();
    int n_dofs = mesh->get_n_elements() * n_dofs_per_elem;

    // Set only each field's constant mode (local DOF 0 per element), leaving every
    // higher-order coefficient at its Kokkos zero-init default. A DG space always reproduces
    // constants exactly, so this makes every field a known, exactly-verifiable constant
    // everywhere -- unlike random per-DOF coefficients, whose interpolated point values
    // wouldn't be hand-checkable (interpolating a product of fields isn't the product of the
    // interpolants for a modal basis).
    const double rho_value = 1.5;
    const double u_value = 2.0;
    const double p_value = 3.25;
    // DGSpace::create_basis uses DubinerBasis for triangles, whose mode-0 (constant) basis
    // function evaluates to 2.0 everywhere (not 1.0, unlike MonomialBasisTriangle) -- see
    // dgfem/src/basis/dubiner.cpp and HANDOFF.md. Local DOF 0's coefficient must be divided
    // by this value for the reconstructed field to equal the target constant exactly.
    const double dubiner_phi0_value = 2.0;
    DView1 field1("field1", n_dofs);
    DView1 field2("field2", n_dofs);
    DView1 field3("field3", n_dofs);
    for (int elem_id = 0; elem_id < mesh->get_n_elements(); ++elem_id) {
        int base = elem_id * n_dofs_per_elem;
        field1(base) = rho_value / dubiner_phi0_value;
        field2(base) = u_value / dubiner_phi0_value;
        field3(base) = p_value / dubiner_phi0_value;
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

    // Parse the actual written values: every point should read back exactly the constant that
    // field was set to, not just have the right header/name present.
    auto rho_values = read_vtk_scalar_field(filename + ".vtk", "rho");
    auto u_values = read_vtk_scalar_field(filename + ".vtk", "u");
    auto p_values = read_vtk_scalar_field(filename + ".vtk", "p");
    ASSERT_FALSE(rho_values.empty());
    ASSERT_EQ(rho_values.size(), u_values.size());
    ASSERT_EQ(rho_values.size(), p_values.size());
    for (std::size_t i = 0; i < rho_values.size(); ++i) {
        EXPECT_NEAR(rho_values[i], rho_value, 1e-9);
        EXPECT_NEAR(u_values[i], u_value, 1e-9);
        EXPECT_NEAR(p_values[i], p_value, 1e-9);
    }

    std::remove((filename + ".vtk").c_str());
}
