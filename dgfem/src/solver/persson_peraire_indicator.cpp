/**
 * @file persson_peraire_indicator.cpp
 * @brief Implementation of the Persson-Peraire modal-decay troubled-cell indicator.
 */

#include "dgfem/solver/persson_peraire_indicator.hpp"

#include "dgfem/solver/euler_eigensystem.hpp"

#include <cmath>
#include <stdexcept>

namespace dgfem {

PerssonPeraireIndicator::PerssonPeraireIndicator(
    std::shared_ptr<const FluxEigensystemProvider> eigensystem_provider,
    std::shared_ptr<const BasisModeMap> mode_map, double kappa)
    : eigensystem_provider_(std::move(eigensystem_provider)), mode_map_(std::move(mode_map)),
      kappa_(kappa) {}

std::vector<std::uint8_t> PerssonPeraireIndicator::detect_troubled_cells(
    const std::vector<DView2>& u, const NeighborConnectivity& neighbor_connectivity,
    const std::vector<double>& element_length, int n_vars, double gamma) const {
    (void)element_length;

    const CoupledFieldConfig& config = eigensystem_provider_->coupled_field_config();
    int n_coupled = static_cast<int>(config.coupled_indices.size());
    if (n_vars != static_cast<int>(config.coupled_indices.size() + config.passive_indices.size())) {
        throw std::invalid_argument(
            "PerssonPeraireIndicator::detect_troubled_cells: n_vars does not match the "
            "eigensystem provider's coupled+passive index count.");
    }

    int n_elem = static_cast<int>(u.size());
    std::vector<std::uint8_t> flags(n_elem, 0);

    int x_linear_mode = mode_map_->linear_mode_index(Direction::X);
    constexpr double kPEff = 2.0;  // order 1 -> "number of modes" convention: order + 1
    const double s0 = -4.0 * std::log10(kPEff);
    constexpr double kEps = 1e-12;

    auto se_log = [](double c0, double c1) {
        double energy0 = c0 * c0 * 2.0;
        double energy1 = c1 * c1 * (2.0 / 3.0);
        double s_e = energy1 / (energy0 + energy1 + kEps);
        return std::log10(s_e + kEps);
    };

    for (int e = 0; e < n_elem; ++e) {
        NeighborPair pair = get_axis_neighbors(neighbor_connectivity[e], FaceDirection::XMinus,
                                               FaceDirection::XPlus);
        if (pair.minus_id < 0 || pair.plus_id < 0) {
            continue;  // boundary element: leave unlimited rather than guess a ghost average.
        }

        bool troubled = false;

        std::vector<double> prim_bar = cell_average_primitives(u[e], n_vars, gamma);
        LocalEigensystem es = eigensystem_provider_->build({1.0, 0.0}, prim_bar, gamma);

        std::vector<double> slope0(n_coupled), slope1(n_coupled);
        for (int k = 0; k < n_coupled; ++k) {
            int v = config.coupled_indices[k];
            slope0[k] = u[e](0, v);
            slope1[k] = u[e](x_linear_mode, v);
        }
        for (int k = 0; k < n_coupled && !troubled; ++k) {
            double c0 = 0.0, c1 = 0.0;
            for (int m = 0; m < n_coupled; ++m) {
                c0 += es.L[k][m] * slope0[m];
                c1 += es.L[k][m] * slope1[m];
            }
            if (se_log(c0, c1) > (s0 - kappa_)) {
                troubled = true;
            }
        }

        for (int v : config.passive_indices) {
            if (troubled) {
                break;
            }
            double c0 = u[e](0, v);
            double c1 = u[e](x_linear_mode, v);
            if (se_log(c0, c1) > (s0 - kappa_)) {
                troubled = true;
            }
        }

        flags[e] = troubled ? 1 : 0;
    }

    return flags;
}

}  // namespace dgfem
