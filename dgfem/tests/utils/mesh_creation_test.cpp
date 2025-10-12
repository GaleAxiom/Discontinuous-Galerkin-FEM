#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <dgfem/utils/mesh_creation.hpp>
#include <dgfem/reference/elements.hpp>
#include <dgfem/reference/mapping.hpp>
#include <dgfem/core/mesh.hpp>
#include <Eigen/Dense>
#include <cmath>

using namespace dgfem;
using namespace testing;

class MeshCreationFixture : public ::testing::Test {
protected:
    void SetUp() override {
        MeshCreator::initialize_gmsh();
    }
    
    void TearDown() override {
        MeshCreator::finalize_gmsh();
    }
};

TEST_F(MeshCreationFixture, CreateRectangularMeshTriangles) {
    // Test triangle mesh with default domain [0,1]^2
    auto mesh_tri = MeshCreator::create_rectangular_mesh(0.5, true);
    
    ASSERT_NE(mesh_tri, nullptr) << "Mesh should not be null";
    
    // Verify mesh has elements
    EXPECT_GT(mesh_tri->get_n_elements(), 0) << "Should have at least one triangle element";
    
    // Verify mesh has vertices
    EXPECT_GT(mesh_tri->get_vertices().rows(), 0) << "Should have vertices";
    EXPECT_EQ(mesh_tri->get_vertices().cols(), 2) << "Vertices should be 2D";
    
    // Verify boundary faces exist
    EXPECT_GT(mesh_tri->get_boundary_faces().size(), 0) << "Should have boundary faces";
    
    // For a unit square with dx=0.5, expect roughly 4-16 triangles
    EXPECT_GE(mesh_tri->get_n_elements(), 4) << "Too few triangles for given mesh size";
    EXPECT_LE(mesh_tri->get_n_elements(), 50) << "Too many triangles for given mesh size";
    
    // Verify triangle elements have 3 vertices each
    EXPECT_EQ(mesh_tri->get_elements().cols(), 3) << "Triangle elements should have 3 vertices";
}

TEST_F(MeshCreationFixture, CreateRectangularMeshQuads) {
    // Test quad mesh with default domain [0,1]^2
    auto mesh_quad = MeshCreator::create_rectangular_mesh(0.5, false);
    
    ASSERT_NE(mesh_quad, nullptr) << "Mesh should not be null";
    
    // Verify mesh has elements
    EXPECT_GT(mesh_quad->get_n_elements(), 0) << "Should have at least one quad element";
    
    // Verify mesh has vertices  
    EXPECT_GT(mesh_quad->get_vertices().rows(), 0) << "Should have vertices";
    EXPECT_EQ(mesh_quad->get_vertices().cols(), 2) << "Vertices should be 2D";
    
    // Verify boundary faces exist
    EXPECT_GT(mesh_quad->get_boundary_faces().size(), 0) << "Should have boundary faces";
    
    // For a unit square with dx=0.5, expect 4 quads (2x2 grid)
    EXPECT_GE(mesh_quad->get_n_elements(), 4) << "Should have at least 4 quads";
    EXPECT_LE(mesh_quad->get_n_elements(), 16) << "Too many quads for given mesh size";
    
    // Verify quad elements have 4 vertices each
    EXPECT_EQ(mesh_quad->get_elements().cols(), 4) << "Quad elements should have 4 vertices";
    
    // Verify boundary - unit square should have 4 boundary edges (one per side)
    // With 4 quads in a 2x2 grid, there should be 8 boundary faces total (2 per side)
    EXPECT_EQ(mesh_quad->get_boundary_faces().size(), 8) << "Unit square with 2x2 quads should have 8 boundary faces";
}

TEST_F(MeshCreationFixture, CreateEulerMesh) {
    // Test with default parameters: domain [-5,5]^2 with dx=0.2
    auto mesh = MeshCreator::create_euler_mesh(-5, 5, -5, 5, 0.2, false);
    
    ASSERT_NE(mesh, nullptr) << "Euler mesh should not be null";
    EXPECT_GT(mesh->get_n_elements(), 0) << "Should have elements";
    
    // Domain is 10x10, with dx=0.2, expect roughly 50x50 = 2500 quads
    EXPECT_GE(mesh->get_n_elements(), 2000) << "Should have roughly 2500 quads for 10x10 domain with dx=0.2";
    EXPECT_LE(mesh->get_n_elements(), 3000) << "Shouldn't have too many quads";
    
    // Verify vertices are within domain
    const auto& vertices = mesh->get_vertices();
    for (int i = 0; i < vertices.rows(); ++i) {
        EXPECT_GE(vertices(i, 0), -5.0 - 0.3) << "X coordinate should be >= -5";
        EXPECT_LE(vertices(i, 0), 5.0 + 0.3) << "X coordinate should be <= 5";
        EXPECT_GE(vertices(i, 1), -5.0 - 0.3) << "Y coordinate should be >= -5";
        EXPECT_LE(vertices(i, 1), 5.0 + 0.3) << "Y coordinate should be <= 5";
    }
}

TEST_F(MeshCreationFixture, CreateCustomDomainMesh) {
    // Test mesh creation with custom domain
    auto mesh = MeshCreator::create_rectangular_mesh(0.25, true, -1.0, 2.0, 0.5, 3.0);
    
    ASSERT_NE(mesh, nullptr) << "Custom domain mesh should not be null";
    EXPECT_GT(mesh->get_n_elements(), 0) << "Should have elements";
    
    // Verify vertices are within custom domain [-1,2] x [0.5,3]
    const auto& vertices = mesh->get_vertices();
    for (int i = 0; i < vertices.rows(); ++i) {
        EXPECT_GE(vertices(i, 0), -1.0 - 0.3) << "X coordinate should be >= -1";
        EXPECT_LE(vertices(i, 0), 2.0 + 0.3) << "X coordinate should be <= 2";
        EXPECT_GE(vertices(i, 1), 0.5 - 0.3) << "Y coordinate should be >= 0.5";
        EXPECT_LE(vertices(i, 1), 3.0 + 0.3) << "Y coordinate should be <= 3";
    }
}

TEST_F(MeshCreationFixture, BoundaryTagsExist) {
    // Test that boundary tags are properly set
    auto mesh = MeshCreator::create_rectangular_mesh(0.5, true);
    
    ASSERT_NE(mesh, nullptr);
    
    const auto& boundary_faces = mesh->get_boundary_faces();
    EXPECT_GT(boundary_faces.size(), 0) << "Should have boundary faces";
    
    // Each boundary face should be recognized as a boundary face
    for (const auto& [elem_id, face_id] : boundary_faces) {
        EXPECT_TRUE(mesh->is_boundary_face(elem_id, face_id)) 
            << "Face should be recognized as boundary for elem " << elem_id << " face " << face_id;
    }
    
    // For a rectangular domain, we should have 4 boundary tags: Bottom, Right, Top, Left
    const auto& boundary_tags = mesh->get_boundary_tags();
    EXPECT_EQ(boundary_tags.size(), 4) << "Should have exactly 4 boundary tags for rectangular domain";
    
    // Verify standard tags exist
    std::set<std::string> expected_tags = {"Bottom", "Top", "Left", "Right"};
    for (const auto& tag_name : expected_tags) {
        EXPECT_TRUE(boundary_tags.find(tag_name) != boundary_tags.end()) 
            << "Should have boundary tag: " << tag_name;
    }
}

TEST(MeshCreationUtilsTest, FindReferenceCoordsQuadCenter) {
    // Test finding reference coords for center of unit square
    Eigen::MatrixXd vertices(4, 2);
    vertices << 0.0, 0.0,
                1.0, 0.0,
                1.0, 1.0,
                0.0, 1.0;
    
    Eigen::Vector2d x_phys(0.5, 0.5);
    auto xi = find_reference_coords(x_phys, vertices, "quad");
    
    ASSERT_TRUE(xi.has_value()) << "Should find reference coords for center point";
    EXPECT_NEAR((*xi)[0], 0.0, 1e-6) << "Center of physical element should map to (0,0) in reference";
    EXPECT_NEAR((*xi)[1], 0.0, 1e-6) << "Center of physical element should map to (0,0) in reference";
}

TEST(MeshCreationUtilsTest, FindReferenceCoordsQuadCorners) {
    // Test all four corners of unit square
    Eigen::MatrixXd vertices(4, 2);
    vertices << 0.0, 0.0,  // v0: bottom-left
                1.0, 0.0,  // v1: bottom-right
                1.0, 1.0,  // v2: top-right
                0.0, 1.0;  // v3: top-left
    
    struct TestCase {
        Eigen::Vector2d physical;
        Eigen::Vector2d reference;
        std::string name;
    };
    
    std::vector<TestCase> test_cases = {
        {Eigen::Vector2d(0.0, 0.0), Eigen::Vector2d(-1.0, -1.0), "bottom-left"},
        {Eigen::Vector2d(1.0, 0.0), Eigen::Vector2d(1.0, -1.0), "bottom-right"},
        {Eigen::Vector2d(1.0, 1.0), Eigen::Vector2d(1.0, 1.0), "top-right"},
        {Eigen::Vector2d(0.0, 1.0), Eigen::Vector2d(-1.0, 1.0), "top-left"}
    };
    
    for (const auto& tc : test_cases) {
        auto xi = find_reference_coords(tc.physical, vertices, "quad");
        ASSERT_TRUE(xi.has_value()) << "Should find reference coords for " << tc.name;
        EXPECT_NEAR((*xi)[0], tc.reference[0], 1e-6) << "Incorrect xi for " << tc.name;
        EXPECT_NEAR((*xi)[1], tc.reference[1], 1e-6) << "Incorrect eta for " << tc.name;
    }
}

TEST(MeshCreationUtilsTest, FindReferenceCoordsQuadEdges) {
    // Test edge midpoints
    Eigen::MatrixXd vertices(4, 2);
    vertices << 0.0, 0.0,
                1.0, 0.0,
                1.0, 1.0,
                0.0, 1.0;
    
    struct EdgeTest {
        Eigen::Vector2d physical;
        Eigen::Vector2d reference;
        std::string edge;
    };
    
    std::vector<EdgeTest> edge_tests = {
        {Eigen::Vector2d(0.5, 0.0), Eigen::Vector2d(0.0, -1.0), "bottom edge"},
        {Eigen::Vector2d(1.0, 0.5), Eigen::Vector2d(1.0, 0.0), "right edge"},
        {Eigen::Vector2d(0.5, 1.0), Eigen::Vector2d(0.0, 1.0), "top edge"},
        {Eigen::Vector2d(0.0, 0.5), Eigen::Vector2d(-1.0, 0.0), "left edge"}
    };
    
    for (const auto& et : edge_tests) {
        auto xi = find_reference_coords(et.physical, vertices, "quad");
        ASSERT_TRUE(xi.has_value()) << "Should find reference coords for " << et.edge;
        EXPECT_NEAR((*xi)[0], et.reference[0], 1e-6) << "Incorrect xi for " << et.edge;
        EXPECT_NEAR((*xi)[1], et.reference[1], 1e-6) << "Incorrect eta for " << et.edge;
    }
}

TEST(MeshCreationUtilsTest, FindReferenceCoordsOutsideElement) {
    // Test point clearly outside the element
    Eigen::MatrixXd vertices(4, 2);
    vertices << 0.0, 0.0,
                1.0, 0.0,
                1.0, 1.0,
                0.0, 1.0;
    
    Eigen::Vector2d x_phys(2.0, 2.0);  // Far outside
    
    // The function should throw when Newton iteration fails to converge
    EXPECT_THROW(find_reference_coords(x_phys, vertices, "quad"), std::runtime_error);
}

TEST(MeshCreationUtilsTest, FindReferenceCoordsTriangle) {
    // Test triangle mapping
    Eigen::MatrixXd vertices(3, 2);
    vertices << 0.0, 0.0,  // v0
                1.0, 0.0,  // v1
                0.0, 1.0;  // v2
    
    // Test center of triangle
    Eigen::Vector2d centroid(1.0/3.0, 1.0/3.0);
    auto xi = find_reference_coords(centroid, vertices, "triangle");
    
    ASSERT_TRUE(xi.has_value()) << "Should find reference coords for triangle centroid";
    
    // For reference triangle, centroid should be at (1/3, 1/3)
    EXPECT_NEAR((*xi)[0], 1.0/3.0, 1e-6) << "Incorrect xi for centroid";
    EXPECT_NEAR((*xi)[1], 1.0/3.0, 1e-6) << "Incorrect eta for centroid";
}

TEST(MeshCreationUtilsTest, FindReferenceCoordsNonStandardQuad) {
    // Test with a non-axis-aligned quad (rotated/scaled)
    Eigen::MatrixXd vertices(4, 2);
    vertices << -0.5, -0.5,
                0.5, -0.5,
                0.5, 0.5,
                -0.5, 0.5;
    
    // Center should still map to (0, 0) in reference
    Eigen::Vector2d center(0.0, 0.0);
    auto xi = find_reference_coords(center, vertices, "quad");
    
    ASSERT_TRUE(xi.has_value()) << "Should find reference coords for centered quad";
    EXPECT_NEAR((*xi)[0], 0.0, 1e-6) << "Center should map to xi=0";
    EXPECT_NEAR((*xi)[1], 0.0, 1e-6) << "Center should map to eta=0";
}
