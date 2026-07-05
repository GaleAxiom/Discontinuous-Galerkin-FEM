/**
 * @file vtk_writer.cpp
 * @brief Implementation of VTK writer for DGFEM solutions
 */

#include "dgfem/utils/vtk_writer.hpp"

#include "dgfem/basis/orthogonal.hpp"
#include "dgfem/core/space.hpp"
#include "dgfem/reference/mapping.hpp"

#include <cmath>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace dgfem {

void VTKWriter::write_solution(std::shared_ptr<DGMesh> mesh, const DView1& solution,
                               const std::string& filename, const std::string& variable_name,
                               int resolution_factor) {
    std::vector<DView1> solutions = {solution};
    std::vector<std::string> names = {variable_name};
    write_multi_variable_solution(mesh, solutions, names, filename, resolution_factor);
}

void VTKWriter::write_multi_variable_solution(std::shared_ptr<DGMesh> mesh,
                                              const std::vector<DView1>& solutions,
                                              const std::vector<std::string>& variable_names,
                                              const std::string& filename, int resolution_factor) {
    if (solutions.size() != variable_names.size()) {
        throw std::runtime_error("Number of solutions must match number of variable names");
    }

    std::cout << "Writing VTK file: " << filename << ".vtk" << std::endl;
    std::cout << "  Elements: " << mesh->get_n_elements() << std::endl;
    std::cout << "  Resolution factor: " << resolution_factor << std::endl;

    // Generate visualization mesh
    DView2 vis_vertices;
    IView2 vis_connectivity;
    generate_visualization_mesh(mesh, resolution_factor, vis_vertices, vis_connectivity);

    std::cout << "  Visualization points: " << vis_vertices.extent(0) << std::endl;
    std::cout << "  Visualization cells: " << vis_connectivity.extent(0) << std::endl;

    // Evaluate solutions at visualization points
    std::vector<std::pair<std::string, DView1>> data_arrays;
    for (size_t i = 0; i < solutions.size(); ++i) {
        DView1 values = evaluate_at_points(mesh, solutions[i], vis_vertices, resolution_factor);
        data_arrays.push_back({variable_names[i], values});
    }

    // Write to file
    std::ofstream file(filename + ".vtk");
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename + ".vtk");
    }

    write_vtk_header(file);
    write_vtk_points(file, vis_vertices);
    write_vtk_cells(file, vis_connectivity, mesh->get_element_type());
    write_vtk_point_data(file, data_arrays);

    file.close();
    std::cout << "  VTK file written successfully!" << std::endl;
}

void VTKWriter::write_euler_solution(std::shared_ptr<DGMesh> mesh, const DView2& solution_matrix,
                                     const std::string& filename, double gamma,
                                     int resolution_factor) {
    std::cout << "Writing Euler solution to VTK file: " << filename << ".vtk" << std::endl;
    std::cout << "  Elements: " << mesh->get_n_elements() << std::endl;
    std::cout << "  Resolution factor: " << resolution_factor << std::endl;

    int n_elem = mesh->get_n_elements();
    int n_basis = mesh->get_dg_space()->get_basis()->get_n_basis();
    int n_vars = 4;
    int n_dofs_per_elem = n_basis * n_vars;

    // Convert flat solution matrix to separate vectors for each conserved variable
    std::vector<DView1> conserved_vars(n_vars);
    for (int v = 0; v < n_vars; ++v) {
        conserved_vars[v] = DView1("conserved_var", n_elem * n_basis);
    }

    // Extract conserved variables from flat format
    for (int e = 0; e < n_elem; ++e) {
        for (int i = 0; i < n_basis; ++i) {
            for (int v = 0; v < n_vars; ++v) {
                conserved_vars[v](e * n_basis + i) = solution_matrix(e, i * n_vars + v);
            }
        }
    }

    // Generate visualization mesh
    DView2 vis_vertices;
    IView2 vis_connectivity;
    generate_visualization_mesh(mesh, resolution_factor, vis_vertices, vis_connectivity);

    std::cout << "  Visualization points: " << vis_vertices.extent(0) << std::endl;
    std::cout << "  Visualization cells: " << vis_connectivity.extent(0) << std::endl;

    // Evaluate conserved variables at visualization points
    std::vector<DView1> conserved_at_points(n_vars);
    for (int v = 0; v < n_vars; ++v) {
        conserved_at_points[v] =
            evaluate_at_points(mesh, conserved_vars[v], vis_vertices, resolution_factor);
    }

    // Convert to primitive variables at each point
    int n_points = static_cast<int>(vis_vertices.extent(0));
    DView1 density("density", n_points);
    DView1 u_velocity("u_velocity", n_points);
    DView1 v_velocity("v_velocity", n_points);
    DView1 pressure("pressure", n_points);
    DView1 temperature("temperature", n_points);
    DView1 mach("mach", n_points);
    DView1 velocity_mag("velocity_mag", n_points);

    for (int i = 0; i < n_points; ++i) {
        double rho = conserved_at_points[0](i);
        double rho_u = conserved_at_points[1](i);
        double rho_v = conserved_at_points[2](i);
        double E = conserved_at_points[3](i);

        // Compute primitive variables
        double u = rho_u / rho;
        double v = rho_v / rho;
        double p = (gamma - 1.0) * (E - 0.5 * rho * (u * u + v * v));
        double T = p / rho;  // Assuming ideal gas with R=1
        double vel_mag = std::sqrt(u * u + v * v);
        double c = std::sqrt(gamma * p / rho);  // Speed of sound
        double M = vel_mag / c;

        density(i) = rho;
        u_velocity(i) = u;
        v_velocity(i) = v;
        pressure(i) = p;
        temperature(i) = T;
        mach(i) = M;
        velocity_mag(i) = vel_mag;
    }

    // Prepare data arrays
    std::vector<std::pair<std::string, DView1>> data_arrays = {{"density", density},
                                                               {"u_velocity", u_velocity},
                                                               {"v_velocity", v_velocity},
                                                               {"velocity_magnitude", velocity_mag},
                                                               {"pressure", pressure},
                                                               {"temperature", temperature},
                                                               {"mach", mach},
                                                               {"rho", conserved_at_points[0]},
                                                               {"rho_u", conserved_at_points[1]},
                                                               {"rho_v", conserved_at_points[2]},
                                                               {"E", conserved_at_points[3]}};

    // Write to file
    std::ofstream file(filename + ".vtk");
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename + ".vtk");
    }

    write_vtk_header(file, "DGFEM Euler Solution");
    write_vtk_points(file, vis_vertices);
    write_vtk_cells(file, vis_connectivity, mesh->get_element_type());
    write_vtk_point_data(file, data_arrays);

    file.close();
    std::cout << "  Euler VTK file written successfully!" << std::endl;
}

void VTKWriter::write_mesh(std::shared_ptr<DGMesh> mesh, const std::string& filename) {
    std::ofstream file(filename + ".vtk");
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename + ".vtk");
    }

    write_vtk_header(file, "DGFEM Mesh");
    write_vtk_points(file, mesh->get_vertices());
    write_vtk_cells(file, mesh->get_elements(), mesh->get_element_type());

    file.close();
    std::cout << "Mesh written to: " << filename << ".vtk" << std::endl;
}

void VTKWriter::write_with_analytical(std::shared_ptr<DGMesh> mesh, const DView1& solution,
                                      std::function<double(const Vec2&)> analytical_func,
                                      const std::string& filename, int resolution_factor) {
    // Generate visualization mesh
    DView2 vis_vertices;
    IView2 vis_connectivity;
    generate_visualization_mesh(mesh, resolution_factor, vis_vertices, vis_connectivity);

    // Evaluate numerical solution
    DView1 numerical_values = evaluate_at_points(mesh, solution, vis_vertices, resolution_factor);

    // Evaluate analytical solution
    int n_points = static_cast<int>(vis_vertices.extent(0));
    DView1 analytical_values("analytical_values", n_points);
    for (int i = 0; i < n_points; ++i) {
        analytical_values(i) = analytical_func(row2(vis_vertices, i));
    }

    // Compute error
    DView1 error("error", n_points);
    for (int i = 0; i < n_points; ++i) {
        error(i) = std::abs(numerical_values(i) - analytical_values(i));
    }

    // Write to file
    std::ofstream file(filename + ".vtk");
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename + ".vtk");
    }

    write_vtk_header(file, "DGFEM Solution with Analytical Comparison");
    write_vtk_points(file, vis_vertices);
    write_vtk_cells(file, vis_connectivity, mesh->get_element_type());

    std::vector<std::pair<std::string, DView1>> data_arrays = {
        {"numerical", numerical_values}, {"analytical", analytical_values}, {"error", error}};
    write_vtk_point_data(file, data_arrays);

    file.close();
    std::cout << "VTK file with analytical comparison written to: " << filename << ".vtk"
              << std::endl;
}

void VTKWriter::generate_visualization_mesh(std::shared_ptr<DGMesh> mesh, int resolution_factor,
                                            DView2& vis_vertices, IView2& vis_connectivity) {
    const std::string& element_type = mesh->get_element_type();

    // Get reference subdivision points and connectivity
    DView2 ref_points = get_reference_subdivision_points(element_type, resolution_factor);
    IView2 ref_connectivity =
        get_reference_subdivision_connectivity(element_type, resolution_factor);

    int n_points_per_elem = static_cast<int>(ref_points.extent(0));
    int n_sub_elems = static_cast<int>(ref_connectivity.extent(0));
    int n_vertices_per_sub_elem = static_cast<int>(ref_connectivity.extent(1));

    // Allocate storage
    int total_points = mesh->get_n_elements() * n_points_per_elem;
    int total_sub_elems = mesh->get_n_elements() * n_sub_elems;

    vis_vertices = DView2("vis_vertices", total_points, 2);
    vis_connectivity = IView2("vis_connectivity", total_sub_elems, n_vertices_per_sub_elem);

    // Get mapping object
    auto mapping = mesh->get_dg_space()->get_mapping();

    // Process each element
    for (int elem = 0; elem < mesh->get_n_elements(); ++elem) {
        // Get physical element vertices
        DView2 elem_vertices = mesh->get_element_vertices(elem);

        // Map reference points to physical space
        int point_offset = elem * n_points_per_elem;
        for (int p = 0; p < n_points_per_elem; ++p) {
            Vec2 ref_pt = row2(ref_points, p);
            MappingData mapping_data = mapping->compute_mapping(elem_vertices, ref_pt);
            set_row2(vis_vertices, point_offset + p, mapping_data.x_phys);
        }

        // Set up connectivity for sub-elements
        int elem_offset = elem * n_sub_elems;
        for (int s = 0; s < n_sub_elems; ++s) {
            for (int v = 0; v < n_vertices_per_sub_elem; ++v) {
                vis_connectivity(elem_offset + s, v) = point_offset + ref_connectivity(s, v);
            }
        }
    }
}

DView1 VTKWriter::evaluate_at_points(std::shared_ptr<DGMesh> mesh, const DView1& solution,
                                     const DView2& vis_vertices, int resolution_factor) {
    const std::string& element_type = mesh->get_element_type();
    auto dg_space = mesh->get_dg_space();
    auto basis = dg_space->get_basis();
    auto mapping = dg_space->get_mapping();

    // Get reference subdivision points
    DView2 ref_points = get_reference_subdivision_points(element_type, resolution_factor);
    int n_points_per_elem = static_cast<int>(ref_points.extent(0));

    // Allocate output
    DView1 values("values", vis_vertices.extent(0));

    // Process each element
    for (int elem = 0; elem < mesh->get_n_elements(); ++elem) {
        // Get element coefficients
        int n_basis = dg_space->get_n_dofs();
        DView1 elem_coeffs("elem_coeffs", n_basis);
        for (int i = 0; i < n_basis; ++i) {
            elem_coeffs(i) = solution(elem * n_basis + i);
        }

        // Evaluate at each subdivision point
        int point_offset = elem * n_points_per_elem;
        for (int p = 0; p < n_points_per_elem; ++p) {
            Vec2 ref_pt = row2(ref_points, p);

            // Evaluate basis functions at reference point
            DView1 basis_vals = basis->evaluate(ref_pt);

            // Compute solution value
            double val = 0.0;
            for (int i = 0; i < n_basis; ++i) {
                val += elem_coeffs(i) * basis_vals(i);
            }
            values(point_offset + p) = val;
        }
    }

    return values;
}

DView2 VTKWriter::get_reference_subdivision_points(std::string_view element_type,
                                                   int resolution_factor) {
    int n_pts_1d = resolution_factor + 1;
    std::vector<Vec2> points;

    if (element_type == "triangle") {
        // Generate points in reference triangle: (0,0), (1,0), (0,1)
        for (int j = 0; j < n_pts_1d; ++j) {
            for (int i = 0; i < n_pts_1d - j; ++i) {
                double xi = static_cast<double>(i) / resolution_factor;
                double eta = static_cast<double>(j) / resolution_factor;
                points.push_back(Vec2{xi, eta});
            }
        }
    } else if (element_type == "quad") {
        // Generate points in reference quad: [-1,1] x [-1,1]
        for (int j = 0; j < n_pts_1d; ++j) {
            for (int i = 0; i < n_pts_1d; ++i) {
                double xi = -1.0 + 2.0 * i / resolution_factor;
                double eta = -1.0 + 2.0 * j / resolution_factor;
                points.push_back(Vec2{xi, eta});
            }
        }
    } else {
        throw std::runtime_error("Unknown element type: " + std::string(element_type));
    }

    // Convert to matrix
    DView2 result("ref_subdivision_points", points.size(), 2);
    for (size_t i = 0; i < points.size(); ++i) {
        set_row2(result, static_cast<int>(i), points[i]);
    }

    return result;
}

IView2 VTKWriter::get_reference_subdivision_connectivity(std::string_view element_type,
                                                         int resolution_factor) {
    std::vector<std::vector<int>> connectivity;
    int n_pts_1d = resolution_factor + 1;

    if (element_type == "triangle") {
        // Helper to get linear index for (i,j) in triangular grid
        auto tri_index = [&](int i, int j) -> int {
            int idx = 0;
            for (int jj = 0; jj < j; ++jj) {
                idx += n_pts_1d - jj;
            }
            idx += i;
            return idx;
        };

        // Create sub-triangles
        for (int j = 0; j < resolution_factor; ++j) {
            for (int i = 0; i < resolution_factor - j; ++i) {
                // Lower-left triangle
                connectivity.push_back({tri_index(i, j), tri_index(i + 1, j), tri_index(i, j + 1)});

                // Upper-right triangle (if not on diagonal)
                if (i < resolution_factor - j - 1) {
                    connectivity.push_back(
                        {tri_index(i + 1, j), tri_index(i + 1, j + 1), tri_index(i, j + 1)});
                }
            }
        }
    } else if (element_type == "quad") {
        // Helper to get linear index for (i,j) in quad grid
        auto quad_index = [&](int i, int j) -> int { return j * n_pts_1d + i; };

        // Create sub-quads
        for (int j = 0; j < resolution_factor; ++j) {
            for (int i = 0; i < resolution_factor; ++i) {
                connectivity.push_back({quad_index(i, j), quad_index(i + 1, j),
                                        quad_index(i + 1, j + 1), quad_index(i, j + 1)});
            }
        }
    } else {
        throw std::runtime_error("Unknown element type: " + std::string(element_type));
    }

    // Convert to matrix
    IView2 result("ref_subdivision_connectivity", connectivity.size(), connectivity[0].size());
    for (size_t i = 0; i < connectivity.size(); ++i) {
        for (size_t j = 0; j < connectivity[i].size(); ++j) {
            result(static_cast<int>(i), static_cast<int>(j)) = connectivity[i][j];
        }
    }

    return result;
}

void VTKWriter::write_vtk_header(std::ofstream& file, const std::string& title) {
    file << "# vtk DataFile Version 3.0\n";
    file << title << "\n";
    file << "ASCII\n";
    file << "DATASET UNSTRUCTURED_GRID\n";
}

void VTKWriter::write_vtk_points(std::ofstream& file, const DView2& vertices) {
    file << "POINTS " << vertices.extent(0) << " double\n";
    file << std::scientific << std::setprecision(10);
    for (int i = 0; i < static_cast<int>(vertices.extent(0)); ++i) {
        file << vertices(i, 0) << " " << vertices(i, 1) << " 0.0\n";
    }
}

void VTKWriter::write_vtk_cells(std::ofstream& file, const IView2& connectivity,
                                const std::string& element_type) {
    int n_cells = static_cast<int>(connectivity.extent(0));
    int n_vertices_per_cell = static_cast<int>(connectivity.extent(1));
    int size = n_cells * (1 + n_vertices_per_cell);

    file << "CELLS " << n_cells << " " << size << "\n";
    for (int i = 0; i < n_cells; ++i) {
        file << n_vertices_per_cell;
        for (int j = 0; j < n_vertices_per_cell; ++j) {
            file << " " << connectivity(i, j);
        }
        file << "\n";
    }

    int cell_type = get_vtk_cell_type(element_type);
    file << "CELL_TYPES " << n_cells << "\n";
    for (int i = 0; i < n_cells; ++i) {
        file << cell_type << "\n";
    }
}

void VTKWriter::write_vtk_point_data(
    std::ofstream& file, const std::vector<std::pair<std::string, DView1>>& data_arrays) {
    if (data_arrays.empty())
        return;

    int n_points = data_arrays[0].second.size();
    file << "POINT_DATA " << n_points << "\n";

    file << std::scientific << std::setprecision(10);
    for (const auto& [name, values] : data_arrays) {
        file << "SCALARS " << name << " double 1\n";
        file << "LOOKUP_TABLE default\n";
        for (int i = 0; i < values.size(); ++i) {
            file << values(i) << "\n";
        }
    }
}

int VTKWriter::get_vtk_cell_type(std::string_view element_type) noexcept {
    if (element_type == "triangle") {
        return 5;  // VTK_TRIANGLE
    } else if (element_type == "quad") {
        return 9;  // VTK_QUAD
    } else {
        return -1;  // Unknown element type
    }
}

}  // namespace dgfem
