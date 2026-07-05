/**
 * @file vtk_writer.hpp
 * @brief VTK file writer for visualizing DGFEM solutions in ParaView
 *
 * This module exports DG solutions to VTK format for visualization in ParaView.
 * It interpolates the DG solution to element vertices for continuous visualization.
 */

#pragma once

#include "dgfem/core/mesh.hpp"
#include "dgfem/core/solution.hpp"
#include "dgfem/kokkos_math.hpp"

#include <functional>
#include <memory>
#include <string>
#include <string_view>

namespace dgfem {

/**
 * @brief VTK file writer for DG solutions
 *
 * Exports DGFEM solutions to VTK legacy format (.vtk) which can be opened in ParaView.
 * The writer interpolates high-order DG solutions to mesh vertices for visualization.
 */
class VTKWriter {
public:
    /**
     * @brief Write DG solution to VTK file
     *
     * @param mesh The DG mesh containing geometry and connectivity
     * @param solution The DG solution coefficients
     * @param filename Output filename (without extension)
     * @param variable_name Name of the solution variable (default: "solution")
     * @param resolution_factor Subdivision factor for high-order visualization (default: 3)
     *
     * The resolution_factor controls how finely elements are subdivided for visualization:
     * - 1: Use only element vertices (fastest, lowest quality for high-order)
     * - 3: Subdivide each element 3x (good balance)
     * - 5+: High quality visualization for high-order solutions
     */
    static void write_solution(std::shared_ptr<DGMesh> mesh, const DView1& solution,
                               const std::string& filename,
                               const std::string& variable_name = "solution",
                               int resolution_factor = 3);

    /**
     * @brief Write multiple solution variables to VTK file
     *
     * @param mesh The DG mesh
     * @param solutions Vector of solution vectors (one per variable)
     * @param variable_names Names for each variable
     * @param filename Output filename (without extension)
     * @param resolution_factor Subdivision factor for visualization
     */
    static void write_multi_variable_solution(std::shared_ptr<DGMesh> mesh,
                                              const std::vector<DView1>& solutions,
                                              const std::vector<std::string>& variable_names,
                                              const std::string& filename,
                                              int resolution_factor = 3);

    /**
     * @brief Write Euler solution to VTK file (converts conserved to primitive variables)
     *
     * @param mesh The DG mesh
     * @param solution_matrix Solution in flat format (n_elem x (n_basis * n_vars))
     * @param filename Output filename (without extension)
     * @param gamma Ratio of specific heats (default: 1.4)
     * @param resolution_factor Subdivision factor for visualization
     *
     * Exports both conserved and primitive variables:
     * - Conserved: rho, rho*u, rho*v, E
     * - Primitive: density, u_velocity, v_velocity, pressure, temperature, Mach
     */
    static void write_euler_solution(std::shared_ptr<DGMesh> mesh, const DView2& solution_matrix,
                                     const std::string& filename, double gamma = 1.4,
                                     int resolution_factor = 3);

    /**
     * @brief Write mesh only (no solution data) to VTK file
     *
     * @param mesh The DG mesh
     * @param filename Output filename (without extension)
     */
    static void write_mesh(std::shared_ptr<DGMesh> mesh, const std::string& filename);

    /**
     * @brief Write solution with analytical comparison
     *
     * Exports both numerical and analytical solutions, plus error field
     *
     * @param mesh The DG mesh
     * @param solution Numerical solution
     * @param analytical_func Function to evaluate analytical solution
     * @param filename Output filename (without extension)
     * @param resolution_factor Subdivision factor for visualization
     */
    static void write_with_analytical(std::shared_ptr<DGMesh> mesh, const DView1& solution,
                                      std::function<double(const Vec2&)> analytical_func,
                                      const std::string& filename, int resolution_factor = 3);

private:
    /**
     * @brief Generate high-resolution visualization points and connectivity
     *
     * Subdivides each element into smaller sub-elements for better visualization
     * of high-order solutions
     *
     * @param mesh The DG mesh
     * @param resolution_factor Subdivision factor
     * @param vis_vertices Output: physical coordinates of visualization points
     * @param vis_connectivity Output: connectivity of visualization sub-elements
     */
    static void generate_visualization_mesh(std::shared_ptr<DGMesh> mesh, int resolution_factor,
                                            DView2& vis_vertices, IView2& vis_connectivity);

    /**
     * @brief Evaluate DG solution at visualization points
     *
     * @param mesh The DG mesh
     * @param solution DG solution coefficients
     * @param vis_vertices Physical coordinates of visualization points
     * @param resolution_factor Subdivision factor used
     * @return Solution values at visualization points
     */
    [[nodiscard]] static DView1 evaluate_at_points(std::shared_ptr<DGMesh> mesh,
                                                   const DView1& solution,
                                                   const DView2& vis_vertices,
                                                   int resolution_factor);

    /**
     * @brief Get reference coordinates for subdivision points
     *
     * @param element_type "triangle" or "quad"
     * @param resolution_factor Subdivision factor
     * @return Matrix of reference coordinates (n_points x 2)
     */
    [[nodiscard]] static DView2 get_reference_subdivision_points(std::string_view element_type,
                                                                 int resolution_factor);

    /**
     * @brief Get connectivity for subdivided reference element
     *
     * @param element_type "triangle" or "quad"
     * @param resolution_factor Subdivision factor
     * @return Matrix of connectivity indices (n_sub_elements x vertices_per_element)
     */
    [[nodiscard]] static IView2
    get_reference_subdivision_connectivity(std::string_view element_type, int resolution_factor);

    /**
     * @brief Write VTK header
     */
    static void write_vtk_header(std::ofstream& file, const std::string& title = "DGFEM Solution");

    /**
     * @brief Write VTK points section
     */
    static void write_vtk_points(std::ofstream& file, const DView2& vertices);

    /**
     * @brief Write VTK cells section
     */
    static void write_vtk_cells(std::ofstream& file, const IView2& connectivity,
                                const std::string& element_type);

    /**
     * @brief Write VTK point data section
     */
    static void
    write_vtk_point_data(std::ofstream& file,
                         const std::vector<std::pair<std::string, DView1>>& data_arrays);

    /**
     * @brief Get VTK cell type identifier
     */
    [[nodiscard]] static int get_vtk_cell_type(std::string_view element_type) noexcept;
};

}  // namespace dgfem
