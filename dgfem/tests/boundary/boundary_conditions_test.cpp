#include <cmath>
#include <dgfem/boundary/conditions.hpp>
#include <dgfem/kokkos_math.hpp>

#include <gtest/gtest.h>

using namespace dgfem;

// ---------------------------------------------------------------------------
// BoundaryCondition (scalar): construction, evaluate(), getters
// ---------------------------------------------------------------------------

TEST(BoundaryConditionTest, DirichletConstantValue) {
    BoundaryCondition bc(BCType::DIRICHLET, 3.5);

    EXPECT_EQ(bc.get_type(), BCType::DIRICHLET);
    EXPECT_DOUBLE_EQ(bc.get_value(), 3.5);
    EXPECT_DOUBLE_EQ(bc.evaluate(Vec2{0.0, 0.0}), 3.5);
    EXPECT_DOUBLE_EQ(bc.evaluate(Vec2{1.0, -2.0}), 3.5);
}

TEST(BoundaryConditionTest, DirichletFunctionValue) {
    auto func = [](const Vec2& x) { return x[0] + 2.0 * x[1]; };
    BoundaryCondition bc(BCType::DIRICHLET, func);

    EXPECT_EQ(bc.get_type(), BCType::DIRICHLET);
    EXPECT_DOUBLE_EQ(bc.evaluate(Vec2{1.0, 2.0}), 5.0);
    EXPECT_DOUBLE_EQ(bc.evaluate(Vec2{0.0, 0.0}), 0.0);
}

TEST(BoundaryConditionTest, NeumannConstantValue) {
    BoundaryCondition bc(BCType::NEUMANN, -1.25);

    EXPECT_EQ(bc.get_type(), BCType::NEUMANN);
    EXPECT_DOUBLE_EQ(bc.evaluate(Vec2{0.5, 0.5}), -1.25);
}

TEST(BoundaryConditionTest, NeumannFunctionValue) {
    auto func = [](const Vec2& x) { return std::sin(x[0]); };
    BoundaryCondition bc(BCType::NEUMANN, func);

    EXPECT_EQ(bc.get_type(), BCType::NEUMANN);
    EXPECT_DOUBLE_EQ(bc.evaluate(Vec2{0.0, 1.0}), 0.0);
}

TEST(BoundaryConditionTest, RobinConstantValueWithAlpha) {
    BoundaryCondition bc(BCType::ROBIN, 2.0, 0.5);

    EXPECT_EQ(bc.get_type(), BCType::ROBIN);
    EXPECT_DOUBLE_EQ(bc.get_value(), 2.0);
    EXPECT_DOUBLE_EQ(bc.get_robin_alpha(), 0.5);
    EXPECT_DOUBLE_EQ(bc.evaluate(Vec2{0.0, 0.0}), 2.0);
}

TEST(BoundaryConditionTest, RobinFunctionValueWithAlpha) {
    auto func = [](const Vec2& x) { return x[0] * x[1]; };
    BoundaryCondition bc(BCType::ROBIN, func, 1.5);

    EXPECT_EQ(bc.get_type(), BCType::ROBIN);
    EXPECT_DOUBLE_EQ(bc.get_robin_alpha(), 1.5);
    EXPECT_DOUBLE_EQ(bc.evaluate(Vec2{2.0, 3.0}), 6.0);
}

TEST(BoundaryConditionTest, NonRobinTypeWithAlphaThrowsConstantOverload) {
    EXPECT_THROW(BoundaryCondition(BCType::DIRICHLET, 1.0, 0.5), std::invalid_argument);
    EXPECT_THROW(BoundaryCondition(BCType::NEUMANN, 1.0, 0.5), std::invalid_argument);
}

TEST(BoundaryConditionTest, NonRobinTypeWithAlphaThrowsFunctionOverload) {
    auto func = [](const Vec2& x) { return x[0]; };
    EXPECT_THROW(BoundaryCondition(BCType::DIRICHLET, func, 0.5), std::invalid_argument);
    EXPECT_THROW(BoundaryCondition(BCType::NEUMANN, func, 0.5), std::invalid_argument);
}

TEST(BoundaryConditionTest, NonRobinConstructorsDefaultAlphaToZero) {
    BoundaryCondition dirichlet(BCType::DIRICHLET, 1.0);
    BoundaryCondition neumann(BCType::NEUMANN, 1.0);

    EXPECT_DOUBLE_EQ(dirichlet.get_robin_alpha(), 0.0);
    EXPECT_DOUBLE_EQ(neumann.get_robin_alpha(), 0.0);
}

// ---------------------------------------------------------------------------
// Factory functions
// ---------------------------------------------------------------------------

TEST(BoundaryConditionFactoryTest, MakeDirichletBcConstant) {
    auto bc = make_dirichlet_bc(4.0);
    ASSERT_NE(bc, nullptr);
    EXPECT_EQ(bc->get_type(), BCType::DIRICHLET);
    EXPECT_DOUBLE_EQ(bc->evaluate(Vec2{0.0, 0.0}), 4.0);
}

TEST(BoundaryConditionFactoryTest, MakeDirichletBcFunction) {
    auto bc = make_dirichlet_bc([](const Vec2& x) { return x[1]; });
    ASSERT_NE(bc, nullptr);
    EXPECT_EQ(bc->get_type(), BCType::DIRICHLET);
    EXPECT_DOUBLE_EQ(bc->evaluate(Vec2{0.0, 7.0}), 7.0);
}

TEST(BoundaryConditionFactoryTest, MakeNeumannBcConstant) {
    auto bc = make_neumann_bc(-2.0);
    ASSERT_NE(bc, nullptr);
    EXPECT_EQ(bc->get_type(), BCType::NEUMANN);
    EXPECT_DOUBLE_EQ(bc->evaluate(Vec2{0.0, 0.0}), -2.0);
}

TEST(BoundaryConditionFactoryTest, MakeNeumannBcFunction) {
    auto bc = make_neumann_bc([](const Vec2& x) { return x[0]; });
    ASSERT_NE(bc, nullptr);
    EXPECT_EQ(bc->get_type(), BCType::NEUMANN);
    EXPECT_DOUBLE_EQ(bc->evaluate(Vec2{3.0, 0.0}), 3.0);
}

TEST(BoundaryConditionFactoryTest, MakeRobinBcConstant) {
    auto bc = make_robin_bc(1.0, 0.25);
    ASSERT_NE(bc, nullptr);
    EXPECT_EQ(bc->get_type(), BCType::ROBIN);
    EXPECT_DOUBLE_EQ(bc->get_robin_alpha(), 0.25);
    EXPECT_DOUBLE_EQ(bc->evaluate(Vec2{0.0, 0.0}), 1.0);
}

TEST(BoundaryConditionFactoryTest, MakeRobinBcFunction) {
    auto bc = make_robin_bc([](const Vec2& x) { return x[0] + x[1]; }, 0.75);
    ASSERT_NE(bc, nullptr);
    EXPECT_EQ(bc->get_type(), BCType::ROBIN);
    EXPECT_DOUBLE_EQ(bc->get_robin_alpha(), 0.75);
    EXPECT_DOUBLE_EQ(bc->evaluate(Vec2{1.0, 2.0}), 3.0);
}

// ---------------------------------------------------------------------------
// BoundaryConditionEuler
// ---------------------------------------------------------------------------

TEST(BoundaryConditionEulerTest, FarFieldConstantValue) {
    Vec4 state{1.0, 0.2, 0.0, 2.5};
    BoundaryConditionEuler bc(BCTypeEuler::FAR_FIELD, state);

    EXPECT_EQ(bc.get_type(), BCTypeEuler::FAR_FIELD);
    Vec4 result = bc.evaluate(Vec2{0.0, 0.0});
    EXPECT_DOUBLE_EQ(result[0], state[0]);
    EXPECT_DOUBLE_EQ(result[1], state[1]);
    EXPECT_DOUBLE_EQ(result[2], state[2]);
    EXPECT_DOUBLE_EQ(result[3], state[3]);
}

TEST(BoundaryConditionEulerTest, SlipWallConstantValue) {
    Vec4 state{1.2, 0.0, 0.0, 3.0};
    BoundaryConditionEuler bc(BCTypeEuler::SLIP_WALL, state);

    EXPECT_EQ(bc.get_type(), BCTypeEuler::SLIP_WALL);
    Vec4 result = bc.evaluate(Vec2{1.0, 1.0});
    EXPECT_DOUBLE_EQ(result[0], state[0]);
    EXPECT_DOUBLE_EQ(result[3], state[3]);
}

TEST(BoundaryConditionEulerTest, NoSlipWallConstantValue) {
    Vec4 state{1.0, 0.0, 0.0, 2.0};
    BoundaryConditionEuler bc(BCTypeEuler::NO_SLIP_WALL, state);

    EXPECT_EQ(bc.get_type(), BCTypeEuler::NO_SLIP_WALL);
    Vec4 result = bc.evaluate(Vec2{0.0, 0.0});
    EXPECT_DOUBLE_EQ(result[1], 0.0);
    EXPECT_DOUBLE_EQ(result[2], 0.0);
}

TEST(BoundaryConditionEulerTest, PeriodicConstantValue) {
    Vec4 state{1.0, 0.1, 0.1, 2.0};
    BoundaryConditionEuler bc(BCTypeEuler::PERIODIC, state);

    EXPECT_EQ(bc.get_type(), BCTypeEuler::PERIODIC);
    Vec4 result = bc.evaluate(Vec2{0.0, 0.0});
    EXPECT_DOUBLE_EQ(result[0], state[0]);
}

TEST(BoundaryConditionEulerTest, FunctionValue) {
    auto func = [](const Vec2& x) -> Vec4 { return Vec4{1.0 + x[0], x[1], 0.0, 2.0}; };
    BoundaryConditionEuler bc(BCTypeEuler::FAR_FIELD, func);

    EXPECT_EQ(bc.get_type(), BCTypeEuler::FAR_FIELD);
    Vec4 result = bc.evaluate(Vec2{0.5, 1.5});
    EXPECT_DOUBLE_EQ(result[0], 1.5);
    EXPECT_DOUBLE_EQ(result[1], 1.5);
    EXPECT_DOUBLE_EQ(result[2], 0.0);
    EXPECT_DOUBLE_EQ(result[3], 2.0);
}
