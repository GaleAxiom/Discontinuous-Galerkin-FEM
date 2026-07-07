/**
 * @file weno_reconstruction.cpp
 * @brief Implementation of the WENO-style reconstruction technique.
 */

#include "dgfem/solver/weno_reconstruction.hpp"

#include "dgfem/solver/euler_eigensystem.hpp"

#include <cmath>

namespace dgfem {

WenoReconstruction::WenoReconstruction(
    std::shared_ptr<const FluxEigensystemProvider> eigensystem_provider,
    std::shared_ptr<const BasisModeMap> mode_map, double gamma_own, double epsilon,
    double weno_power)
    : eigensystem_provider_(std::move(eigensystem_provider)), mode_map_(std::move(mode_map)),
      gamma_own_(gamma_own), epsilon_(epsilon), weno_power_(weno_power) {}

double WenoReconstruction::blend_slopes(double s_own, double s_left, double s_right, double dx,
                                        double gamma_own, double epsilon, double weno_power) {
    double gamma_lr = (1.0 - gamma_own) / 2.0;
    double beta_own = dx * dx * s_own * s_own;
    double beta_left = dx * dx * s_left * s_left;
    double beta_right = dx * dx * s_right * s_right;

    double w_own = gamma_own / std::pow(epsilon + beta_own, weno_power);
    double w_left = gamma_lr / std::pow(epsilon + beta_left, weno_power);
    double w_right = gamma_lr / std::pow(epsilon + beta_right, weno_power);
    double sum = w_own + w_left + w_right;

    return (w_own * s_own + w_left * s_left + w_right * s_right) / sum;
}

std::vector<DView2>
WenoReconstruction::apply(const std::vector<DView2>& u,
                         const std::vector<std::uint8_t>& troubled_cells,
                         const NeighborConnectivity& neighbor_connectivity,
                         const std::vector<double>& element_length, int n_vars,
                         double gamma) const {
    const CoupledFieldConfig& config = eigensystem_provider_->coupled_field_config();
    int n_coupled = static_cast<int>(config.coupled_indices.size());
    int n_elem = static_cast<int>(u.size());
    int x_linear_mode = mode_map_->linear_mode_index(Direction::X);

    std::vector<DView2> result(n_elem);
    for (int e = 0; e < n_elem; ++e) {
        int n_basis = static_cast<int>(u[e].extent(0));
        result[e] = DView2("limited_elem", u[e].extent(0), u[e].extent(1));
        for (int i = 0; i < n_basis; ++i) {
            for (int v = 0; v < n_vars; ++v) {
                result[e](i, v) = u[e](i, v);
            }
        }

        if (!troubled_cells[e]) {
            continue;
        }

        NeighborPair pair = get_axis_neighbors(neighbor_connectivity[e], FaceDirection::XMinus,
                                               FaceDirection::XPlus);
        if (pair.minus_id < 0 || pair.plus_id < 0) {
            continue;
        }
        int n_left = pair.minus_id, n_right = pair.plus_id;
        double dx_e = element_length[e];

        // Passive fields: WENO blend of own/left/right's OWN linear coefficients directly (no
        // characteristic transform needed -- these fields have no coupled wave structure).
        // Unlike MinmodReconstruction, the cross term is deliberately left untouched: minmod's
        // moment-limiter cascade (zero the higher mode once the lower one is deemed
        // untrustworthy) is the right call for a *discrete* trust/no-trust decision, but WENO's
        // blend is continuous -- a cell whose blended slope is 99.9% its own value (because its
        // neighbors were rougher) has not had its linear trend meaningfully discredited, so
        // discarding the cross term there is pure information loss with no matching benefit.
        // Confirmed empirically: unconditionally zeroing it (as an earlier version of this
        // function did) measurably hurt Sod shock tube accuracy even as gamma_own -> 1 (i.e.
        // even when the blend left the slope nearly unchanged from the unlimited value).
        for (int v : config.passive_indices) {
            double s_own = u[e](x_linear_mode, v) * (2.0 / dx_e);
            double s_left = u[n_left](x_linear_mode, v) * (2.0 / element_length[n_left]);
            double s_right = u[n_right](x_linear_mode, v) * (2.0 / element_length[n_right]);
            double blended =
                blend_slopes(s_own, s_left, s_right, dx_e, gamma_own_, epsilon_, weno_power_);
            result[e](x_linear_mode, v) = blended * (dx_e / 2.0);
        }

        // Coupled fields: project own/left/right's OWN linear coefficients into this
        // element's own characteristic space (frozen at e's cell average), blend per field,
        // transform back.
        std::vector<double> prim_bar = cell_average_primitives(u[e], n_vars, gamma);
        LocalEigensystem es = eigensystem_provider_->build({1.0, 0.0}, prim_bar, gamma);

        std::vector<double> slope_own(n_coupled), slope_left(n_coupled), slope_right(n_coupled);
        for (int k = 0; k < n_coupled; ++k) {
            int v = config.coupled_indices[k];
            slope_own[k] = u[e](x_linear_mode, v) * (2.0 / dx_e);
            slope_left[k] = u[n_left](x_linear_mode, v) * (2.0 / element_length[n_left]);
            slope_right[k] = u[n_right](x_linear_mode, v) * (2.0 / element_length[n_right]);
        }

        std::vector<double> char_own(n_coupled), char_left(n_coupled), char_right(n_coupled),
            char_blended(n_coupled);
        for (int k = 0; k < n_coupled; ++k) {
            char_own[k] = char_left[k] = char_right[k] = 0.0;
            for (int m = 0; m < n_coupled; ++m) {
                char_own[k] += es.L[k][m] * slope_own[m];
                char_left[k] += es.L[k][m] * slope_left[m];
                char_right[k] += es.L[k][m] * slope_right[m];
            }
        }
        for (int k = 0; k < n_coupled; ++k) {
            char_blended[k] = blend_slopes(char_own[k], char_left[k], char_right[k], dx_e,
                                           gamma_own_, epsilon_, weno_power_);
        }

        for (int k = 0; k < n_coupled; ++k) {
            int v = config.coupled_indices[k];
            double limited_slope_v = 0.0;
            for (int m = 0; m < n_coupled; ++m) {
                limited_slope_v += es.R[k][m] * char_blended[m];
            }
            result[e](x_linear_mode, v) = limited_slope_v * (dx_e / 2.0);
        }
    }
    return result;
}

}  // namespace dgfem
