/**
 * @file minmod_reconstruction.cpp
 * @brief Implementation of characteristic-variable TVB minmod reconstruction.
 */

#include "dgfem/solver/minmod_reconstruction.hpp"

#include "dgfem/solver/euler_eigensystem.hpp"

#include <cmath>

namespace dgfem {

namespace {
// Classic three-argument minmod: returns 0 unless a, b, c all share the same sign, in which
// case it returns the smallest-magnitude one.
double minmod3(double a, double b, double c) {
    if (a > 0.0 && b > 0.0 && c > 0.0) {
        return std::min({a, b, c});
    }
    if (a < 0.0 && b < 0.0 && c < 0.0) {
        return std::max({a, b, c});
    }
    return 0.0;
}
}  // namespace

MinmodReconstruction::MinmodReconstruction(
    std::shared_ptr<const FluxEigensystemProvider> eigensystem_provider,
    std::shared_ptr<const BasisModeMap> mode_map, double tvb_m)
    : eigensystem_provider_(std::move(eigensystem_provider)), mode_map_(std::move(mode_map)),
      tvb_m_(tvb_m) {}

std::vector<DView2>
MinmodReconstruction::apply(const std::vector<DView2>& u,
                            const std::vector<std::uint8_t>& troubled_cells,
                            const NeighborConnectivity& neighbor_connectivity,
                            const std::vector<double>& element_length, int n_vars,
                            double gamma) const {
    const CoupledFieldConfig& config = eigensystem_provider_->coupled_field_config();
    int n_coupled = static_cast<int>(config.coupled_indices.size());
    int n_elem = static_cast<int>(u.size());
    int x_linear_mode = mode_map_->linear_mode_index(Direction::X);
    std::vector<int> cross_terms = mode_map_->dependent_cross_term_indices(Direction::X);

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
        double tvb_threshold = tvb_m_ * dx_e;

        // Passive fields: componentwise TVB minmod on cell-average differences.
        for (int v : config.passive_indices) {
            double ubar_e = u[e](0, v);
            double ubar_left = u[n_left](0, v);
            double ubar_right = u[n_right](0, v);
            double slope_phys = u[e](x_linear_mode, v) * (2.0 / dx_e);
            if (std::abs(slope_phys) <= tvb_threshold) {
                continue;
            }
            double slope_fwd = (ubar_right - ubar_e) / dx_e;
            double slope_bwd = (ubar_e - ubar_left) / dx_e;
            double limited_slope = minmod3(slope_phys, slope_fwd, slope_bwd);
            if (std::abs(limited_slope - slope_phys) > 1e-13 * (std::abs(slope_phys) + 1.0)) {
                result[e](x_linear_mode, v) = limited_slope * (dx_e / 2.0);
                for (int ct : cross_terms) {
                    result[e](ct, v) = 0.0;
                }
            }
        }

        // Coupled fields: characteristic-variable limiting.
        std::vector<double> prim_bar = cell_average_primitives(u[e], n_vars, gamma);
        LocalEigensystem es = eigensystem_provider_->build({1.0, 0.0}, prim_bar, gamma);

        std::vector<double> slope_phys(n_coupled), slope_fwd(n_coupled), slope_bwd(n_coupled);
        for (int k = 0; k < n_coupled; ++k) {
            int v = config.coupled_indices[k];
            slope_phys[k] = u[e](x_linear_mode, v) * (2.0 / dx_e);
            slope_fwd[k] = (u[n_right](0, v) - u[e](0, v)) / dx_e;
            slope_bwd[k] = (u[e](0, v) - u[n_left](0, v)) / dx_e;
        }

        std::vector<double> char_phys(n_coupled), char_fwd(n_coupled), char_bwd(n_coupled),
            char_limited(n_coupled);
        for (int k = 0; k < n_coupled; ++k) {
            char_phys[k] = char_fwd[k] = char_bwd[k] = 0.0;
            for (int m = 0; m < n_coupled; ++m) {
                char_phys[k] += es.L[k][m] * slope_phys[m];
                char_fwd[k] += es.L[k][m] * slope_fwd[m];
                char_bwd[k] += es.L[k][m] * slope_bwd[m];
            }
        }

        bool any_limited = false;
        for (int k = 0; k < n_coupled; ++k) {
            if (std::abs(char_phys[k]) <= tvb_threshold) {
                char_limited[k] = char_phys[k];
                continue;
            }
            char_limited[k] = minmod3(char_phys[k], char_fwd[k], char_bwd[k]);
            if (std::abs(char_limited[k] - char_phys[k]) > 1e-13 * (std::abs(char_phys[k]) + 1.0)) {
                any_limited = true;
            }
        }

        if (any_limited) {
            for (int k = 0; k < n_coupled; ++k) {
                int v = config.coupled_indices[k];
                double limited_slope_v = 0.0;
                for (int m = 0; m < n_coupled; ++m) {
                    limited_slope_v += es.R[k][m] * char_limited[m];
                }
                result[e](x_linear_mode, v) = limited_slope_v * (dx_e / 2.0);
                for (int ct : cross_terms) {
                    result[e](ct, v) = 0.0;
                }
            }
        }
    }
    return result;
}

}  // namespace dgfem
