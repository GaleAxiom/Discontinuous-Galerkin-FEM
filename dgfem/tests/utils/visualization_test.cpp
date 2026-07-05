#include <gmock/gmock.h>
#include <gtest/gtest.h>
// #include <dgfem/utils/visualization.hpp>
#include <cmath>
#include <dgfem/core/mesh.hpp>
#include <dgfem/core/space.hpp>

using namespace dgfem;
using namespace testing;

//? temporarly disabled

// // Helper function to create a test mesh
// std::shared_ptr<Mesh> create_test_mesh() {
//     Eigen::MatrixXd vertices(4, 2);
//     vertices << 0.0, 0.0,
//                 1.0, 0.0,
//                 1.0, 1.0,
//                 0.0, 1.0;

//     Eigen::MatrixXi elements(2, 3);
//     elements << 0, 1, 2,
//                 0, 2, 3;

//     return std::make_shared<Mesh>(vertices, elements);
// }

// #ifdef HAS_MATPLOTLIB
// TEST(VisualizationTest, VisualizeLaplaceSolution) {
//     auto mesh = create_test_mesh();

//     // Create a simple analytical solution
//     auto analytical_func = [](const Eigen::Vector2d& x) -> double {
//         return std::sin(M_PI * x(0)) * std::sin(M_PI * x(1));
//     };

//     EXPECT_NO_THROW(visualize_laplace_solution(*mesh, analytical_func));
// }

// TEST(VisualizationTest, VisualizeAdvectionSolution) {
//     auto mesh = create_test_mesh();

//     // Create some dummy solution frames
//     std::vector<Eigen::VectorXd> solution_frames;
//     int n_dofs = mesh->n_elements() * mesh->dg_space().basis().n_basis();

//     for (int i = 0; i < 5; ++i) {
//         Eigen::VectorXd frame = Eigen::VectorXd::Zero(n_dofs);
//         frame.array() = std::sin(i * 0.5);  // Different value for each frame
//         solution_frames.push_back(frame);
//     }

//     EXPECT_NO_THROW(visualize_advection_solution(*mesh, solution_frames, 0.1, 1));
// }

// TEST(VisualizationTest, VisualizeEulerSolution) {
//     auto mesh = create_test_mesh();

//     // Create some dummy solution frames for Euler equations
//     std::vector<Eigen::MatrixXd> solution_frames;
//     int n_basis = mesh->dg_space().basis().n_basis();

//     for (int i = 0; i < 5; ++i) {
//         // Each element has conserved variables [rho, rhou, rhov, E]
//         Eigen::MatrixXd frame(mesh->n_elements(), n_basis * 4);
//         frame.setZero();

//         // Set some values for density (first component)
//         frame.col(0).array() = 1.0 + 0.1 * std::sin(i * 0.5);

//         solution_frames.push_back(frame);
//     }

//     EXPECT_NO_THROW(visualize_euler_solution(*mesh, solution_frames, 0.1, 1, 0, "Density"));
// }
// #endif

// TEST(VisualizationTest, InterpolateDataTriangle) {
//     auto mesh = create_test_mesh();

//     // Test interpolation of scalar data on a triangle
//     Eigen::VectorXd data = Eigen::VectorXd::Ones(mesh->n_elements() *
//     mesh->dg_space().basis().n_basis());

//     // Test points
//     std::vector<Eigen::Vector2d> test_points = {
//         Eigen::Vector2d(0.5, 0.5),  // Inside
//         Eigen::Vector2d(0.25, 0.25), // Inside
//         Eigen::Vector2d(2.0, 2.0)    // Outside
//     };

//     for (const auto& point : test_points) {
//         auto result = interpolate_data_at_point(*mesh, data, point);
//         if (point.x() <= 1.0 && point.y() <= 1.0) {
//             EXPECT_TRUE(result.has_value());
//             if (result) {
//                 EXPECT_NEAR(*result, 1.0, 1e-10);
//             }
//         } else {
//             EXPECT_FALSE(result.has_value());
//         }
//     }
// }

// TEST(VisualizationTest, GenerateVisualizationPoints) {
//     auto mesh = create_test_mesh();

//     // Test generation of visualization points
//     auto [points, triangles] = generate_visualization_points(*mesh, 5);

//     EXPECT_GT(points.rows(), 0);
//     EXPECT_EQ(points.cols(), 2);

//     if (mesh->element_type() == "triangle") {
//         EXPECT_GT(triangles.rows(), 0);
//         EXPECT_EQ(triangles.cols(), 3);
//     } else {  // quad
//         EXPECT_GT(triangles.rows(), 0);
//         EXPECT_EQ(triangles.cols(), 3);
//         // For quads, each element should be split into two triangles
//         EXPECT_EQ(triangles.rows(), 2 * mesh->n_elements());
//     }
// }
