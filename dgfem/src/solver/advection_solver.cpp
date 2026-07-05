#include "dgfem/reference/mapping.hpp"
#include "dgfem/solver/dg_solver.hpp"
#include "dgfem/solver/time_stepping.hpp"

#include <cmath>

#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace dgfem {

using Stage = DGSolverBase::Stage;

AdvectionDGSolver::AdvectionDGSolver(std::shared_ptr<DGMesh> mesh, const Vec2& advection_velocity)
    : DGSolverBase(std::move(mesh), "Advection"),
      weak_form_(std::make_shared<AdvectionWeakFormulation>(advection_velocity)),
      advection_velocity_(advection_velocity) {
    assembler_ = std::make_shared<DGAssembler>(mesh_, weak_form_);

    std::ostringstream oss;
    oss << "Velocity = (" << advection_velocity_[0] << ", " << advection_velocity_[1] << ")";
    log(Stage::Setup, oss.str());
}

Teuchos::RCP<const TpetraCrsMatrix> AdvectionDGSolver::get_system_matrix() const {
    return assembler_->get_system_matrix();
}

double AdvectionDGSolver::compute_cfl(double dt) const {
    double speed = std::sqrt(advection_velocity_[0] * advection_velocity_[0] +
                             advection_velocity_[1] * advection_velocity_[1]);

    double h_min = std::numeric_limits<double>::max();
    for (int elem_id = 0; elem_id < mesh_->get_n_elements(); ++elem_id) {
        DView2 vertices = mesh_->get_element_vertices(elem_id);
        int n_verts = static_cast<int>(vertices.extent(0));
        for (int i = 0; i < n_verts; ++i) {
            for (int j = i + 1; j < n_verts; ++j) {
                double dist = norm(row2(vertices, i) - row2(vertices, j));
                h_min = std::min(h_min, dist);
            }
        }
    }

    return speed * dt / h_min;
}

std::vector<Teuchos::RCP<TpetraMultiVector>>
AdvectionDGSolver::solve(std::function<double(const Vec2&)> initial_condition, double T_final,
                         double dt, std::shared_ptr<BoundaryCondition> boundary_condition,
                         int save_every) {
    if (dt <= 0.0) {
        throw std::invalid_argument("Time step dt must be positive");
    }
    if (T_final <= 0.0) {
        throw std::invalid_argument("T_final must be positive");
    }

    std::ostringstream setup_msg;
    setup_msg << "Config: T_final = " << T_final << ", dt = " << dt
              << ", save_every = " << save_every;
    log(Stage::Setup, setup_msg.str());

    double cfl = compute_cfl(dt);
    std::ostringstream cfl_msg;
    cfl_msg << "CFL number = " << std::fixed << std::setprecision(4) << cfl;
    log(Stage::Setup, cfl_msg.str());
    if (cfl > 1.0) {
        log(Stage::Setup, "WARNING: CFL exceeds 1.0; results may be unstable.");
    }

    begin_stage(Stage::Projection);
    compute_mass_matrix_inverse_blocks();
    auto u = project_initial_condition(initial_condition);
    end_stage(Stage::Projection);
    log(Stage::Projection, "Projected initial condition onto DG space.");

    auto mesh_solution = mesh_->get_solution();
    mesh_solution->set_global_coeffs(tpetra_to_view(*u));

    begin_stage(Stage::Assembly);
    auto bc_func = [boundary_condition](const Vec2& x) -> double {
        if (boundary_condition) {
            return boundary_condition->evaluate(x);
        }
        return 0.0;
    };

    assembler_->assemble(nullptr, bc_func);
    L_operator_ = assembler_->get_system_matrix();
    F_boundary_ = assembler_->get_rhs();
    end_stage(Stage::Assembly);
    log(Stage::Assembly, "Assembled advection operator and boundary flux.");

    begin_stage(Stage::TimeStep);

    std::vector<Teuchos::RCP<TpetraMultiVector>> solution_frames;
    double t = 0.0;
    int step = 0;
    int n_steps = static_cast<int>(std::ceil(T_final / dt));

    solution_frames.push_back(u);

    while (t < T_final) {
        if (step % save_every == 0 && step > 0) {
            solution_frames.push_back(u);
        }

        u = time_step_ssp_rk3(u, dt);
        t += dt;
        step++;
        increment_steps();
    }

    solution_frames.push_back(u);

    mesh_solution->set_global_coeffs(tpetra_to_view(*u));

    end_stage(Stage::TimeStep);

    increment_output_frames(static_cast<int>(solution_frames.size()));

    std::ostringstream timestep_msg;
    timestep_msg << "Completed time stepping in " << n_steps << " steps. Saved "
                 << solution_frames.size() << " frames.";
    log(Stage::TimeStep, timestep_msg.str());

    log(Stage::Output, "Updated mesh solution coefficients with final state.");

    return solution_frames;
}

void AdvectionDGSolver::compute_mass_matrix_inverse_blocks() {
    auto M = assembler_->assemble_mass_matrix();

    int n_basis = mesh_->get_dg_space()->get_basis()->get_n_basis();
    M_inv_blocks_.clear();
    M_inv_blocks_.reserve(mesh_->get_n_elements());

    // Mass integrals never couple different elements, so each element's block is exactly the
    // rows [start, start+n_basis) of the (block-diagonal) global mass matrix, restricted to
    // those same columns.
    for (int elem_id = 0; elem_id < mesh_->get_n_elements(); ++elem_id) {
        int start = elem_id * n_basis;
        DView2 M_block("M_block", n_basis, n_basis);
        for (int i = 0; i < n_basis; ++i) {
            typename TpetraCrsMatrix::local_inds_host_view_type indices;
            typename TpetraCrsMatrix::values_host_view_type values;
            M->getLocalRowView(start + i, indices, values);
            for (size_t k = 0; k < indices.extent(0); ++k) {
                int col = static_cast<int>(indices(k)) - start;
                if (col >= 0 && col < n_basis) {
                    M_block(i, col) = values(k);
                }
            }
        }
        M_inv_blocks_.push_back(invert_dense(M_block));
    }
}

Teuchos::RCP<TpetraMultiVector>
AdvectionDGSolver::apply_mass_inv(const Teuchos::RCP<const TpetraMultiVector>& vec) const {
    int n_basis = mesh_->get_dg_space()->get_basis()->get_n_basis();
    auto result = Teuchos::rcp(new TpetraMultiVector(vec->getMap(), 1));

    auto in_view = vec->getLocalViewHost(Tpetra::Access::ReadOnly);
    auto out_view = result->getLocalViewHost(Tpetra::Access::OverwriteAll);

    for (int elem_id = 0; elem_id < mesh_->get_n_elements(); ++elem_id) {
        int start = elem_id * n_basis;
        DView1 local_in("local_in", n_basis);
        for (int i = 0; i < n_basis; ++i) {
            local_in[i] = in_view(start + i, 0);
        }
        DView1 local_out("local_out", n_basis);
        gemv('N', 1.0, M_inv_blocks_[elem_id], local_in, 0.0, local_out);
        for (int i = 0; i < n_basis; ++i) {
            out_view(start + i, 0) = local_out[i];
        }
    }

    return result;
}

Teuchos::RCP<TpetraMultiVector>
AdvectionDGSolver::project_initial_condition(std::function<double(const Vec2&)> u0_func) {
    auto dg_space = mesh_->get_dg_space();
    auto mapping = dg_space->get_mapping();
    int n_basis = dg_space->get_basis()->get_n_basis();
    int n_dofs = mesh_->get_n_elements() * n_basis;

    auto map = make_serial_map(n_dofs);
    auto F_proj = Teuchos::rcp(new TpetraMultiVector(map, 1));
    auto view = F_proj->getLocalViewHost(Tpetra::Access::OverwriteAll);

    for (int elem_id = 0; elem_id < mesh_->get_n_elements(); ++elem_id) {
        DView2 vertices = mesh_->get_element_vertices(elem_id);

        const DView2& phi = dg_space->get_volume_basis_values();
        const DView1& weights = dg_space->get_volume_quad()->weights;
        const auto& elem_data = mesh_->get_element_data(elem_id);
        const DView2& J_det = elem_data.at("J_det_vol");

        int n_quad = weights.extent(0);
        DView1 F_local("F_local", n_basis);

        for (int q = 0; q < n_quad; ++q) {
            Vec2 xi_q = row2(dg_space->get_volume_quad()->points, q);
            Vec2 x_q = mapping->map_to_physical(vertices, xi_q);
            double u0_val = u0_func(x_q);
            double w_q = weights[q] * std::abs(J_det(q, 0));

            for (int i = 0; i < n_basis; ++i) {
                F_local[i] += w_q * u0_val * phi(q, i);
            }
        }

        int start = elem_id * n_basis;
        for (int i = 0; i < n_basis; ++i) {
            view(start + i, 0) = F_local[i];
        }
    }

    return apply_mass_inv(F_proj);
}

Teuchos::RCP<TpetraMultiVector>
AdvectionDGSolver::time_step_ssp_rk3(const Teuchos::RCP<TpetraMultiVector>& u_n, double dt) const {
    std::function<Teuchos::RCP<TpetraMultiVector>(const Teuchos::RCP<TpetraMultiVector>&)>
        rhs_func = [this](const Teuchos::RCP<TpetraMultiVector>& u) { return compute_rhs(u); };
    return SSP_RK::step_rk3<Teuchos::RCP<TpetraMultiVector>>(u_n, dt, rhs_func);
}

Teuchos::RCP<TpetraMultiVector>
AdvectionDGSolver::compute_rhs(const Teuchos::RCP<TpetraMultiVector>& u) const {
    increment_rhs_evaluations();

    auto Lu = Teuchos::rcp(new TpetraMultiVector(L_operator_->getRangeMap(), 1));
    L_operator_->apply(*u, *Lu);
    Lu->update(1.0, *F_boundary_, 1.0);  // Lu += F_boundary_

    return apply_mass_inv(Lu);
}

}  // namespace dgfem
