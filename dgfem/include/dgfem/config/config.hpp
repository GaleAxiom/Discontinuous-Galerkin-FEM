/**
 * @file config.hpp
 * @brief Configuration parameters for the DGFEM package
 */

#pragma once

#include <string>

namespace dgfem {

/**
 * @brief Global configuration parameters for DGFEM
 */
struct Config {
    // Testing and debugging
    bool run_tests = true;      ///< Set to true to run tests
    bool use_triangles = true;  ///< Set to true to use triangles instead of quads

    // Domain size
    double xmin = 0.0;  ///< Minimum x-coordinate
    double xmax = 1.0;  ///< Maximum x-coordinate
    double ymin = 0.0;  ///< Minimum y-coordinate
    double ymax = 1.0;  ///< Maximum y-coordinate

    // Element parameters
    int order = 1;        ///< Order of finite elements (1-7)
    double dx = 0.1;      ///< Target mesh size
    double sigma = 10.0;  ///< Penalty parameter for interior faces

    // Time-stepping parameters
    double dt = 0.02;        ///< Time step for time-dependent problems
    double T_final = 10.0;   ///< Final time
    int save_interval = 10;  ///< Interval for saving solution frames

    // Advection parameters
    double beta_x = 0.5;   ///< x-component of advection velocity
    double beta_y = 0.25;  ///< y-component of advection velocity

    // File paths
    std::string output_dir = "../output/";
    std::string fig_dir = "../fig/";

    // Singleton access
    static Config& instance() {
        static Config config;
        return config;
    }

private:
    Config() = default;
};

// Convenience macros for accessing configuration
#define DGFEM_CONFIG dgfem::Config::instance()

}  // namespace dgfem