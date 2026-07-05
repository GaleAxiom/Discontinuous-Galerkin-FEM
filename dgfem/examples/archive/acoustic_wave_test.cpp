/**
 * @file acoustic_wave_test.cpp
 * @brief Test the Euler DG solver with acoustic wave propagation
 *
 * This test case implements plane acoustic waves, which are exact solutions to the
 * linearized Euler equations. By comparing numerical solutions to analytical ones,
 * we can measure:
 *
 * 1. DISPERSION ERROR: Does the wave travel at the correct speed?
 *    - Phase error = (numerical speed - exact speed) / exact speed
 *    - Affects wave position over time
 *
 * 2. DISSIPATION ERROR: Does the wave amplitude decay?
 *    - Amplitude error = (numerical amplitude - exact amplitude) / exact amplitude
 *    - Affects wave magnitude over time
 *
 * Analytical solution for small amplitude acoustic waves:
 *   ρ(x,t) = ρ₀ + A·sin(k·x - ω·t)
 *   u(x,t) = (A/ρ₀)·c·sin(k·x - ω·t)
 *   p(x,t) = p₀ + A·c²·sin(k·x - ω·t)
 *
 * where ω = k·c and c = √(γp₀/ρ₀) is the speed of sound.
 */

#include <Kokkos_Core.hpp>
#include <cmath>

#include <algorithm>
#include <chrono>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

// Include DGFEM headers
#include "dgfem/boundary/conditions.hpp"
#include "dgfem/config/config.hpp"
#include "dgfem/core/mesh.hpp"
#include "dgfem/core/space.hpp"
#include "dgfem/solver/dg_solver.hpp"
#include "dgfem/solver/weak_form.hpp"
#include "dgfem/utils/mesh_creation.hpp"
#include "dgfem/utils/vtk_writer.hpp"

// Acoustic wave analytical solution
struct AcousticWave {
    double rho0;       // Background density
    double p0;         // Background pressure
    double u0;         // Background velocity (0 for standing wave)
    double amplitude;  // Wave amplitude (small for linear regime)
    double k;          // Wave number (2π/λ)
    double omega;      // Angular frequency
    double c;          // Speed of sound
    double gamma;      // Ratio of specific heats

    AcousticWave(double rho0_, double p0_, double u0_, double A, double wavelength, double gamma_)
        : rho0(rho0_), p0(p0_), u0(u0_), amplitude(A), gamma(gamma_) {
        c = std::sqrt(gamma * p0 / rho0);  // Speed of sound
        k = 2.0 * M_PI / wavelength;       // Wave number
        omega = k * c;                     // Dispersion relation (exact for linear acoustics)

        // Check linear regime
        double M_acoustic = A * c / (rho0 * c);  // Acoustic Mach number
        if (M_acoustic > 0.1) {
            std::cout << "WARNING: Amplitude may be too large for linear acoustics (M_ac = "
                      << M_acoustic << ")" << std::endl;
        }
    }

    // Get primitive variables at time t
    dgfem::Vec4 primitive_variables(double x, double y, double t) const {
        // 1D plane wave in x-direction
        double phase = k * x - omega * t;

        double rho = rho0 + amplitude * std::sin(phase);
        double u = u0 + (amplitude / rho0) * c * std::sin(phase);
        double v = 0.0;  // No y-component for x-directed wave
        double p = p0 + amplitude * c * c * std::sin(phase);

        return dgfem::Vec4{rho, u, v, p};
    }

    // Get conserved variables at time t
    dgfem::Vec4 conserved_variables(double x, double y, double t) const {
        dgfem::Vec4 W = primitive_variables(x, y, t);
        return dgfem::primitive_to_conserved(W, gamma);
    }

    // Get exact amplitude at a given time (should be constant for linear acoustics)
    double exact_amplitude(double t) const {
        return amplitude;  // No decay in inviscid Euler
    }

    // Get exact phase at a given time
    double exact_phase(double x, double t) const { return k * x - omega * t; }
};

// Min/max of column `col` of an (n, m) view.
static std::pair<double, double> col_min_max(const dgfem::DView2& m, int col) {
    double lo = std::numeric_limits<double>::infinity();
    double hi = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < static_cast<int>(m.extent(0)); ++i) {
        lo = std::min(lo, m(i, col));
        hi = std::max(hi, m(i, col));
    }
    return {lo, hi};
}

// Measure wave amplitude using Fourier analysis (perturbation only)
double measure_amplitude(const std::shared_ptr<dgfem::DGMesh>& mesh, const dgfem::DView2& solution,
                         double k, double gamma,
                         double rho0) {  // Add background density parameter
    auto dg_space = mesh->get_dg_space();
    auto mapping = dg_space->get_mapping();
    int n_elem = mesh->get_n_elements();
    int n_basis = dg_space->get_basis()->get_n_basis();

    // Sample points along a horizontal line (y = constant)
    const int n_samples = 200;
    double y_sample = M_PI;  // Middle of domain

    std::vector<double> x_samples(n_samples);
    std::vector<double> rho_samples(n_samples);

    auto [xmin, xmax] = col_min_max(mesh->get_vertices(), 0);

    const auto& quad_points = dg_space->get_volume_quad()->points;
    const dgfem::DView2& phi = dg_space->get_volume_basis_values();

    for (int i = 0; i < n_samples; ++i) {
        x_samples[i] = xmin + (xmax - xmin) * i / (n_samples - 1);

        // Find element containing this point and evaluate solution
        for (int e = 0; e < n_elem; ++e) {
            dgfem::DView2 vertices = mesh->get_element_vertices(e);

            // Check if point is in this element (simple box test)
            auto [x_min, x_max] = col_min_max(vertices, 0);
            auto [y_min, y_max] = col_min_max(vertices, 1);

            if (x_samples[i] >= x_min && x_samples[i] <= x_max && y_sample >= y_min &&
                y_sample <= y_max) {
                // Use first quadrature point approximation for simplicity
                dgfem::Vec4 U{0.0, 0.0, 0.0, 0.0};
                for (int j = 0; j < n_basis; ++j) {
                    for (int var = 0; var < 4; ++var) {
                        U[var] += solution(e, j * 4 + var) * phi(0, j);
                    }
                }

                dgfem::Vec4 W = dgfem::conserved_to_primitive(U, gamma);
                rho_samples[i] = W[0] - rho0;  // Store PERTURBATION only
                break;
            }
        }
    }

    // Compute Fourier coefficient at wave number k using trapezoidal rule
    std::complex<double> fourier_coeff(0.0, 0.0);
    double L = xmax - xmin;
    double dx = L / (n_samples - 1);

    for (int i = 0; i < n_samples; ++i) {
        double phase = k * x_samples[i];
        double weight = (i == 0 || i == n_samples - 1) ? 0.5 : 1.0;  // Trapezoidal rule
        fourier_coeff += weight * std::complex<double>(rho_samples[i] * std::cos(phase),
                                                       -rho_samples[i] * std::sin(phase));
    }

    // Normalize: (2/L) * integral for Fourier coefficient
    fourier_coeff *= (2.0 * dx / L);
    return std::abs(fourier_coeff);
}

// Measure phase shift by comparing numerical to exact solution
double measure_phase_shift(const std::shared_ptr<dgfem::DGMesh>& mesh,
                           const dgfem::DView2& solution, const AcousticWave& exact, double t,
                           double gamma) {
    auto dg_space = mesh->get_dg_space();
    int n_elem = mesh->get_n_elements();
    int n_basis = dg_space->get_basis()->get_n_basis();

    // Sample along a line y = const
    const int n_samples = 200;
    double y_sample = M_PI;

    auto [xmin, xmax] = col_min_max(mesh->get_vertices(), 0);

    const auto& quad_points = dg_space->get_volume_quad()->points;
    const dgfem::DView2& phi = dg_space->get_volume_basis_values();

    // Compute cross-correlation to find phase shift
    std::vector<double> rho_num_samples(n_samples);
    std::vector<double> rho_exact_samples(n_samples);

    for (int i = 0; i < n_samples; ++i) {
        double x = xmin + (xmax - xmin) * i / (n_samples - 1);

        // Find element containing this point
        for (int e = 0; e < n_elem; ++e) {
            dgfem::DView2 vertices = mesh->get_element_vertices(e);

            auto [x_min, x_max] = col_min_max(vertices, 0);
            auto [y_min, y_max] = col_min_max(vertices, 1);

            if (x >= x_min && x <= x_max && y_sample >= y_min && y_sample <= y_max) {
                // Evaluate numerical solution
                dgfem::Vec4 U{0.0, 0.0, 0.0, 0.0};
                for (int j = 0; j < n_basis; ++j) {
                    for (int var = 0; var < 4; ++var) {
                        U[var] += solution(e, j * 4 + var) * phi(0, j);
                    }
                }

                dgfem::Vec4 W_num = dgfem::conserved_to_primitive(U, gamma);
                dgfem::Vec4 W_exact = exact.primitive_variables(x, y_sample, t);

                // Store density perturbations
                rho_num_samples[i] = W_num[0] - exact.rho0;
                rho_exact_samples[i] = W_exact[0] - exact.rho0;
                break;
            }
        }
    }

    // Compute phase difference using Fourier transform
    double k = exact.k;
    std::complex<double> num_fourier(0.0, 0.0);
    std::complex<double> exact_fourier(0.0, 0.0);
    double dx = (xmax - xmin) / (n_samples - 1);

    for (int i = 0; i < n_samples; ++i) {
        double x = xmin + dx * i;
        double weight = (i == 0 || i == n_samples - 1) ? 0.5 : 1.0;

        num_fourier += weight * std::complex<double>(rho_num_samples[i] * std::cos(k * x),
                                                     -rho_num_samples[i] * std::sin(k * x));

        exact_fourier += weight * std::complex<double>(rho_exact_samples[i] * std::cos(k * x),
                                                       -rho_exact_samples[i] * std::sin(k * x));
    }

    // Phase difference is angle between the two complex numbers
    double phase_num = std::arg(num_fourier);
    double phase_exact = std::arg(exact_fourier);
    double phase_error = phase_num - phase_exact;

    // Wrap to [-pi, pi]
    while (phase_error > M_PI)
        phase_error -= 2.0 * M_PI;
    while (phase_error < -M_PI)
        phase_error += 2.0 * M_PI;

    return phase_error;
}

// Compute L2 errors
std::map<std::string, double> compute_errors(const std::shared_ptr<dgfem::DGMesh>& mesh,
                                             const dgfem::DView2& numerical_solution,
                                             const AcousticWave& exact, double time, double gamma) {
    auto dg_space = mesh->get_dg_space();
    auto mapping = dg_space->get_mapping();
    int n_elem = mesh->get_n_elements();
    int n_basis = dg_space->get_basis()->get_n_basis();

    double rho_error_sq = 0.0, rho_norm_sq = 0.0;
    double u_error_sq = 0.0, u_norm_sq = 0.0;
    double p_error_sq = 0.0, p_norm_sq = 0.0;

    const auto& quad_points = dg_space->get_volume_quad()->points;
    const auto& quad_weights = dg_space->get_volume_quad()->weights;
    int n_quad = static_cast<int>(quad_weights.size());

    for (int e = 0; e < n_elem; ++e) {
        const auto& elem_data = mesh->get_element_data(e);
        const dgfem::DView2& J_det = elem_data.at("J_det_vol");

        dgfem::DView2 vertices = mesh->get_element_vertices(e);

        const dgfem::DView2& phi = dg_space->get_volume_basis_values();

        for (int q = 0; q < n_quad; ++q) {
            double w_q = quad_weights[q];
            double det_J = J_det(q, 0);

            dgfem::Vec2 xi_q = dgfem::row2(quad_points, q);
            dgfem::Vec2 x_phys = mapping->map_to_physical(vertices, xi_q);

            dgfem::Vec4 U_num{0.0, 0.0, 0.0, 0.0};
            for (int i = 0; i < n_basis; ++i) {
                for (int var = 0; var < 4; ++var) {
                    U_num[var] += numerical_solution(e, i * 4 + var) * phi(q, i);
                }
            }

            dgfem::Vec4 W_num = dgfem::conserved_to_primitive(U_num, gamma);
            dgfem::Vec4 W_exact = exact.primitive_variables(x_phys[0], x_phys[1], time);

            double dw = w_q * det_J;

            rho_error_sq += (W_num[0] - W_exact[0]) * (W_num[0] - W_exact[0]) * dw;
            rho_norm_sq += W_exact[0] * W_exact[0] * dw;

            u_error_sq += (W_num[1] - W_exact[1]) * (W_num[1] - W_exact[1]) * dw;
            u_norm_sq += W_exact[1] * W_exact[1] * dw;

            p_error_sq += (W_num[3] - W_exact[3]) * (W_num[3] - W_exact[3]) * dw;
            p_norm_sq += W_exact[3] * W_exact[3] * dw;
        }
    }

    std::map<std::string, double> errors;
    errors["rho"] = std::sqrt(rho_error_sq) / std::sqrt(rho_norm_sq);
    errors["u"] = std::sqrt(u_error_sq) / std::sqrt(u_norm_sq);
    errors["p"] = std::sqrt(p_error_sq) / std::sqrt(p_norm_sq);

    return errors;
}

int main(int argc, char** argv) {
    Kokkos::ScopeGuard kokkos_guard(argc, argv);
    try {
        std::cout << "=== DGFEM Acoustic Wave Propagation Test ===" << std::endl;
        std::cout << "============================================" << std::endl;
        std::cout << "\nThis test analyzes DISPERSION and DISSIPATION errors" << std::endl;
        std::cout << "by comparing numerical vs. analytical acoustic waves.\n" << std::endl;

        // Initialize GMSH
        dgfem::MeshCreator::initialize_gmsh();

        // Configuration
        auto& config = dgfem::Config::instance();
        config.use_triangles = true;
        config.order = 2;

        // Physics parameters
        const double gamma = 1.4;
        const double rho0 = 1.0;
        const double p0 = 1.0;
        const double c0 = std::sqrt(gamma * p0 / rho0);  // Speed of sound

        // Wave parameters - FIXED wavelength, varying resolution
        const double wavelength = 1.0;         // Keep wave properties constant
        const double amplitude = 0.01 * rho0;  // 1% perturbation (linear regime)

        // Domain size must be multiple of wavelength for periodic BC
        const int n_wavelengths = 4;  // Fit 6 wavelengths in domain
        const double L = n_wavelengths * wavelength;

        config.xmin = 0.0;
        config.xmax = L;
        config.ymin = 0.0;
        config.ymax = L;

        // Test multiple resolutions - vary mesh size only
        std::vector<double> ppw_values = {20.0, 15.0, 10.0};  // Points per wavelength
        std::vector<std::string> resolution_names = {"fine", "medium", "coarse"};

        std::cout << "--- Physics Parameters ---" << std::endl;
        std::cout << "  Gamma: " << gamma << std::endl;
        std::cout << "  Background density: " << rho0 << std::endl;
        std::cout << "  Background pressure: " << p0 << std::endl;
        std::cout << "  Speed of sound: " << c0 << std::endl;
        std::cout << "  Wavelength λ: " << wavelength << " (FIXED)" << std::endl;
        std::cout << "  Wave amplitude: " << amplitude << " (" << (100.0 * amplitude / rho0)
                  << "% of background)" << std::endl;
        std::cout << "  Domain: [0, " << L << "] × [0, " << L << "]" << std::endl;
        std::cout << "  Domain contains: " << n_wavelengths << " wavelengths (exact)" << std::endl;

        // Wave properties (same for all tests)
        AcousticWave exact_wave(rho0, p0, 0.0, amplitude, wavelength, gamma);
        double period = 2.0 * M_PI / exact_wave.omega;

        // FIXED timestep for all resolutions - choose based on finest mesh
        double finest_dx = wavelength / ppw_values[0];  // Finest resolution
        double CFL = 0.1;
        double dt_base = CFL * finest_dx / (c0 * (2.0 * config.order + 1.0));

        // Make dt an exact divisor of the period for exact periodic ending
        int steps_per_period = static_cast<int>(std::ceil(period / dt_base));
        double dt = period / steps_per_period;  // Adjusted to fit exactly

        // Simulation time: exactly 2 periods
        double T_final = 2.0 * period;
        int total_steps = static_cast<int>(T_final / dt);
        int save_every = steps_per_period / 10;  // ~10 frames per period

        std::cout << "\n--- Time Integration (SAME for all resolutions) ---" << std::endl;
        std::cout << "  Period T: " << period << std::endl;
        std::cout << "  Time step dt: " << dt << " (exactly " << steps_per_period
                  << " steps per period)" << std::endl;
        std::cout << "  Total time: " << T_final << " (exactly 2 periods)" << std::endl;
        std::cout << "  Total steps: " << total_steps << std::endl;

        // Results storage for analysis
        std::vector<std::map<std::string, std::vector<double>>> all_results;
        std::vector<double> actual_ppw;  // Store actual PPW for each test

        // Test each resolution
        for (size_t r_idx = 0; r_idx < ppw_values.size(); ++r_idx) {
            double target_ppw = ppw_values[r_idx];
            std::string resolution_name = resolution_names[r_idx];

            std::cout << "\n" << std::string(60, '=') << std::endl;
            std::cout << "TEST " << (r_idx + 1) << ": " << resolution_name
                      << " resolution (PPW = " << target_ppw << ")" << std::endl;
            std::cout << std::string(60, '=') << std::endl;

            // Set mesh size based on desired points per wavelength
            config.dx = wavelength / target_ppw;

            std::cout << "\n--- Resolution Details ---" << std::endl;
            std::cout << "  Target PPW: " << target_ppw << std::endl;
            std::cout << "  Mesh size h: " << config.dx << std::endl;
            std::cout << "  Actual PPW: " << wavelength / config.dx << std::endl;
            std::cout << "  CFL number: " << dt * c0 * (2.0 * config.order + 1.0) / config.dx
                      << std::endl;

            actual_ppw.push_back(wavelength / config.dx);

            // Create mesh
            auto mesh = dgfem::MeshCreator::create_rectangular_mesh(config.dx, config.use_triangles,
                                                                    config.xmin, config.xmax,
                                                                    config.ymin, config.ymax);

            std::cout << "  Elements: " << mesh->get_n_elements() << std::endl;

            // Create DG space
            auto dg_space =
                std::make_shared<dgfem::DGSpace>(mesh->get_element_type(), config.order);
            mesh->initialize_dg_space(dg_space, 4);

            std::cout << "  DOFs: "
                      << mesh->get_n_elements() * dg_space->get_basis()->get_n_basis() * 4
                      << std::endl;

            // Initial condition
            auto initial_condition = [&exact_wave](const dgfem::Vec2& x) -> dgfem::Vec4 {
                return exact_wave.conserved_variables(x[0], x[1], 0.0);
            };

            // Periodic boundaries
            mesh->set_periodic_boundaries("Left", "Right");
            mesh->set_periodic_boundaries("Bottom", "Top");

            // Solve
            dgfem::EulerDGSolver solver(mesh, gamma);

            std::cout << "\n--- Solving ---" << std::endl;
            auto start = std::chrono::high_resolution_clock::now();

            std::vector<dgfem::DView2> solutions =
                solver.solve(initial_condition, T_final, dt, save_every);

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = end - start;

            std::cout << "  Solve time: " << duration.count() << " s" << std::endl;
            std::cout << "  Frames saved: " << solutions.size() << std::endl;

            // Analyze dispersion and dissipation
            std::cout << "\n--- DISPERSION & DISSIPATION ANALYSIS ---" << std::endl;

            std::map<std::string, std::vector<double>> results;
            results["time"] = std::vector<double>();
            results["amplitude"] = std::vector<double>();
            results["phase_error"] = std::vector<double>();
            results["dissipation"] = std::vector<double>();
            results["l2_error"] = std::vector<double>();

            double initial_amplitude =
                measure_amplitude(mesh, solutions[0], exact_wave.k, gamma, rho0);

            for (size_t i = 0; i < solutions.size(); ++i) {
                double t = i * save_every * dt;
                results["time"].push_back(t);

                // Measure amplitude (dissipation)
                double amp = measure_amplitude(mesh, solutions[i], exact_wave.k, gamma, rho0);
                results["amplitude"].push_back(amp);
                results["dissipation"].push_back((initial_amplitude - amp) / initial_amplitude *
                                                 100.0);

                // Measure phase error (dispersion)
                double phase_shift = measure_phase_shift(mesh, solutions[i], exact_wave, t, gamma);
                results["phase_error"].push_back(phase_shift);

                // L2 error
                auto errors = compute_errors(mesh, solutions[i], exact_wave, t, gamma);
                results["l2_error"].push_back(errors["rho"]);
            }

            all_results.push_back(results);

            // Print summary
            std::cout << "\n  Time    | Amplitude | Dissipation | Phase Error | L2 Error"
                      << std::endl;
            std::cout << "  " << std::string(65, '-') << std::endl;
            for (size_t i = 0; i < results["time"].size(); ++i) {
                std::cout << "  " << std::setw(6) << std::fixed << std::setprecision(3)
                          << results["time"][i] << "  | " << std::setw(9) << std::scientific
                          << std::setprecision(3) << results["amplitude"][i] << " | "
                          << std::setw(10) << std::fixed << std::setprecision(2)
                          << results["dissipation"][i] << "%" << " | " << std::setw(11)
                          << std::scientific << std::setprecision(2) << results["phase_error"][i]
                          << " | " << std::setw(8) << std::setprecision(2) << results["l2_error"][i]
                          << std::endl;
            }

            // Export VTK
            for (size_t i = 0; i < solutions.size(); ++i) {
                std::string filename =
                    "../../output/acoustic_" + resolution_name + "_" + std::to_string(i);
                dgfem::VTKWriter::write_euler_solution(mesh, solutions[i], filename, gamma, 2);
            }

            std::cout << "\n  ✓ Exported " << solutions.size() << " VTK frames" << std::endl;
        }

        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "SUMMARY: Dispersion & Dissipation Analysis" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        std::cout << "\nNOTE: All tests use SAME wavelength with DIFFERENT mesh resolutions."
                  << std::endl;
        std::cout << "This isolates the effect of spatial discretization on accuracy.\n"
                  << std::endl;

        for (size_t r = 0; r < ppw_values.size(); ++r) {
            const auto& res = all_results[r];
            size_t n = res.at("time").size();

            double avg_dissipation = 0.0;
            double max_phase_error = 0.0;
            double avg_l2_error = 0.0;
            double max_l2_error = 0.0;
            double final_l2_error = res.at("l2_error").back();

            for (size_t i = 0; i < n; ++i) {
                avg_dissipation += res.at("dissipation")[i];
                max_phase_error = std::max(max_phase_error, std::abs(res.at("phase_error")[i]));
                avg_l2_error += res.at("l2_error")[i];
                max_l2_error = std::max(max_l2_error, res.at("l2_error")[i]);
            }
            avg_dissipation /= n;
            avg_l2_error /= n;

            std::cout << "\nResolution: " << resolution_names[r] << " (PPW = " << actual_ppw[r]
                      << "):" << std::endl;
            std::cout << "  Average dissipation: " << std::fixed << std::setprecision(2)
                      << avg_dissipation << "%" << std::endl;
            std::cout << "  Maximum phase error: " << std::scientific << std::setprecision(3)
                      << max_phase_error << " rad" << std::endl;
            std::cout << "  Average L2 error: " << std::setprecision(3) << avg_l2_error
                      << std::endl;
            std::cout << "  Maximum L2 error: " << std::setprecision(3) << max_l2_error
                      << std::endl;
            std::cout << "  Final L2 error: " << std::setprecision(3) << final_l2_error
                      << std::endl;

            // Interpretation
            if (avg_dissipation < 1.0) {
                std::cout << "  → EXCELLENT: Very low numerical dissipation!" << std::endl;
            } else if (avg_dissipation < 5.0) {
                std::cout << "  → GOOD: Low numerical dissipation" << std::endl;
            } else {
                std::cout << "  → WARNING: Significant numerical dissipation" << std::endl;
            }

            if (max_phase_error < 0.1) {
                std::cout << "  → EXCELLENT: Very low dispersion error!" << std::endl;
            } else if (max_phase_error < 0.5) {
                std::cout << "  → GOOD: Low dispersion error" << std::endl;
            } else {
                std::cout << "  → WARNING: Significant dispersion error" << std::endl;
            }
        }

        dgfem::MeshCreator::finalize_gmsh();

        std::cout << "\n=== Acoustic wave test COMPLETED! ===" << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        dgfem::MeshCreator::finalize_gmsh();
        return 1;
    }
}
