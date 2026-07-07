/**
 * @file basis_mode_map_test.cpp
 * @brief Tests for the basis mode-index lookup used by the troubled-cell indicator /
 * reconstruction technique hierarchy.
 */

#include <dgfem/solver/basis_mode_map.hpp>

#include <gtest/gtest.h>

using namespace dgfem;
using namespace testing;

TEST(Order1QuadBasisModeMapTest, LinearModeIndices) {
    Order1QuadBasisModeMap mode_map;
    EXPECT_EQ(mode_map.linear_mode_index(Direction::X), 1);
    EXPECT_EQ(mode_map.linear_mode_index(Direction::Y), 2);
    EXPECT_EQ(mode_map.linear_mode_index(Direction::Z), -1);  // no z-mode on a 2D quad
}

TEST(Order1QuadBasisModeMapTest, DependentCrossTermIndices) {
    Order1QuadBasisModeMap mode_map;
    EXPECT_EQ(mode_map.dependent_cross_term_indices(Direction::X), (std::vector<int>{3}));
    EXPECT_EQ(mode_map.dependent_cross_term_indices(Direction::Y), (std::vector<int>{3}));
    EXPECT_EQ(mode_map.dependent_cross_term_indices(Direction::Z), (std::vector<int>{}));
}

TEST(Order1QuadBasisModeMapTest, GetTypeReportsOrder1Quad) {
    Order1QuadBasisModeMap mode_map;
    EXPECT_EQ(mode_map.get_type(), "Order1Quad");
}
