/**
 * @file weak_formulation_base.cpp
 * @brief Implementation of base weak formulation classes
 */

#include "dgfem/weak_forms/weak_formulation_base.hpp"

#include "dgfem/core/mesh.hpp"
#include "dgfem/core/space.hpp"
#include "dgfem/reference/mapping.hpp"
#include "dgfem/solver/assembler.hpp"

#include <cmath>

#include <iostream>
#include <set>

namespace dgfem {

// ============================================================================
// TimeIndependentWeakFormulation implementation
// ============================================================================

void TimeIndependentWeakFormulation::assemble(DGAssembler& assembler,
                                              std::function<double(const Vec2&)> source_func,
                                              std::function<double(const Vec2&)> bc_func) const {
    auto mesh = assembler.get_mesh();
    auto dg_space = assembler.get_dg_space();

    assembler.clear_assembly_data();

    std::cout << "\nAssembling time-independent system..." << std::endl;

    // Volume integrals
    for (int elem_id = 0; elem_id < mesh->get_n_elements(); ++elem_id) {
        const auto& elem_data = mesh->get_element_data(elem_id);
        DView2 K_vol = compute_volume_integral(elem_data, dg_space);
        assembler.add_to_matrix(elem_id, elem_id, K_vol);

        // Source term
        if (source_func) {
            DView1 F_src = compute_source_integral(elem_id, source_func, mesh);
            assembler.add_to_rhs(elem_id, F_src);
        }
    }

    std::cout << "Volume integrals assembled" << std::endl;

    // Face integrals
    std::set<std::pair<std::pair<int, int>, std::pair<int, int>>> processed_faces;

    for (int elem_L = 0; elem_L < mesh->get_n_elements(); ++elem_L) {
        auto neighbors = mesh->get_element_neighbors(elem_L);

        for (int face_L = 0; face_L < static_cast<int>(neighbors.size()); ++face_L) {
            int elem_R = neighbors[face_L].first;
            int face_R = neighbors[face_L].second;

            if (elem_R >= 0) {
                // Interior face. Dedup by the (elem,face) pair on each side, not just
                // (elem_L, elem_R) alone: if two elements ever share more than one face, an
                // (elem_L, elem_R)-only key would treat the second shared face as already
                // processed and silently drop its contribution. Matches the finer key
                // TimeDependentWeakFormulation::assemble() below already uses.
                std::pair<int, int> pair_L(elem_L, face_L);
                std::pair<int, int> pair_R(elem_R, face_R);
                auto face_key = std::minmax(pair_L, pair_R);
                if (processed_faces.find(face_key) == processed_faces.end()) {
                    processed_faces.insert(face_key);

                    DView2 v_L = mesh->get_element_vertices(elem_L);
                    DView2 v_R = mesh->get_element_vertices(elem_R);

                    auto perm = dg_space->compute_face_permutation(face_L, v_L, face_R, v_R);

                    auto [K_LL, K_LR, K_RL, K_RR] =
                        compute_interior_face_integral(elem_L, face_L, elem_R, face_R, mesh, perm);

                    assembler.add_to_matrix(elem_L, elem_L, K_LL);
                    assembler.add_to_matrix(elem_L, elem_R, K_LR);
                    assembler.add_to_matrix(elem_R, elem_L, K_RL);
                    assembler.add_to_matrix(elem_R, elem_R, K_RR);
                }
            } else {
                // Boundary face (elem_R == -1). compute_boundary_face_integral/
                // compute_boundary_rhs_integral are pure virtual on this base, so every
                // concrete formulation already implements them -- no need for derived classes
                // to re-implement this whole loop just to fill in the boundary branch.
                auto bc = mesh->get_boundary_condition(elem_L, face_L);
                if (!bc) {
                    std::cerr << "Warning: No boundary condition set for element " << elem_L
                              << " face " << face_L << std::endl;
                    continue;
                }

                DView2 K_bnd = compute_boundary_face_integral(elem_L, face_L, mesh, bc);
                assembler.add_to_matrix(elem_L, elem_L, K_bnd);

                DView1 F_bnd = compute_boundary_rhs_integral(elem_L, face_L, mesh, bc);
                assembler.add_to_rhs(elem_L, F_bnd);
            }
        }
    }

    std::cout << "Face integrals assembled" << std::endl;

    assembler.finalize_assembly();
}

DView1 TimeIndependentWeakFormulation::compute_source_integral(
    int elem_id, std::function<double(const Vec2&)> source_func,
    std::shared_ptr<DGMesh> mesh) const {
    auto dg_space = mesh->get_dg_space();
    int n_basis = dg_space->get_basis()->get_n_basis();
    DView1 F("source_integral", n_basis);

    const auto& elem_data = mesh->get_element_data(elem_id);
    const DView2& phi = elem_data.at("phi");
    const DView2& xy = elem_data.at("xy");
    const DView2& weights = elem_data.at("weights");
    double jac_det = elem_data.at("jac_det")(0, 0);

    for (int q = 0; q < static_cast<int>(weights.extent(0)); ++q) {
        Vec2 pt = row2(xy, q);
        double f_val = source_func(pt);
        for (int i = 0; i < n_basis; ++i) {
            F(i) += weights(q, 0) * jac_det * f_val * phi(q, i);
        }
    }

    return F;
}

// ============================================================================
// TimeDependentWeakFormulation implementation
// ============================================================================

void TimeDependentWeakFormulation::assemble(DGAssembler& assembler,
                                            std::function<double(const Vec2&)> source_func,
                                            std::function<double(const Vec2&)> bc_func) const {
    auto mesh = assembler.get_mesh();
    auto dg_space = assembler.get_dg_space();

    assembler.clear_assembly_data();

    // Volume term (stiffness integral)
    for (int elem_id = 0; elem_id < mesh->get_n_elements(); ++elem_id) {
        const auto& elem_data = mesh->get_element_data(elem_id);
        DView2 L_vol = compute_volume_integral(elem_data, dg_space);
        assembler.add_to_matrix(elem_id, elem_id, L_vol);
    }

    // Interior face terms with upwind flux
    std::set<std::pair<std::pair<int, int>, std::pair<int, int>>> processed_faces;

    int n_faces_per_elem = (mesh->get_element_type() == "triangle") ? 3 : 4;

    for (int elem_L = 0; elem_L < mesh->get_n_elements(); ++elem_L) {
        const auto& neighbors = mesh->get_element_neighbors(elem_L);

        for (int face_L = 0; face_L < n_faces_per_elem; ++face_L) {
            int elem_R = neighbors[face_L].first;
            int face_R = neighbors[face_L].second;

            if (elem_R >= 0) {
                std::pair<int, int> pair_L = std::make_pair(elem_L, face_L);
                std::pair<int, int> pair_R = std::make_pair(elem_R, face_R);
                auto face_key = std::minmax(pair_L, pair_R);

                if (processed_faces.find(face_key) == processed_faces.end()) {
                    processed_faces.insert(face_key);

                    DView2 v_L = mesh->get_element_vertices(elem_L);
                    DView2 v_R = mesh->get_element_vertices(elem_R);

                    IView1 perm = dg_space->compute_face_permutation(face_L, v_L, face_R, v_R);

                    auto [L_LL, L_LR, L_RL, L_RR] =
                        compute_interior_face_integral(elem_L, face_L, elem_R, face_R, mesh, perm);

                    assembler.add_to_matrix(elem_L, elem_L, L_LL);
                    assembler.add_to_matrix(elem_L, elem_R, L_LR);
                    assembler.add_to_matrix(elem_R, elem_L, L_RL);
                    assembler.add_to_matrix(elem_R, elem_R, L_RR);
                }
            }
        }
    }

    // Boundary face terms
    for (const auto& [elem_id, face_id] : mesh->get_boundary_faces()) {
        auto [L_bc, F_bc] = compute_boundary_face_integral(elem_id, face_id, mesh, bc_func);
        assembler.add_to_matrix(elem_id, elem_id, L_bc);
        assembler.add_to_rhs(elem_id, F_bc);
    }

    assembler.finalize_assembly();
}

DView2
TimeDependentWeakFormulation::compute_mass_integral(const std::map<std::string, DView2>& elem_data,
                                                    std::shared_ptr<DGSpace> dg_space) const {
    int n_basis = dg_space->get_basis()->get_n_basis();
    DView2 M("mass_integral", n_basis, n_basis);

    const DView1& weights = dg_space->get_volume_quad()->weights;
    const DView2& phi = dg_space->get_volume_basis_values();
    const DView2& J_det = elem_data.at("J_det_vol");

    int n_quad = weights.extent(0);

    for (int q = 0; q < n_quad; ++q) {
        double w_q_phys = weights[q] * std::abs(J_det(q, 0));
        DView1 phi_q = row_of(phi, q);
        outer_add(M, w_q_phys, phi_q, phi_q);
    }

    return M;
}

}  // namespace dgfem
