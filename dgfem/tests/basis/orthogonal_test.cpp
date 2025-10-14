#include <Eigen/Dense>
#include <dgfem/basis/orthogonal.hpp>
#include <dgfem/reference/elements.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace dgfem;
using namespace testing;

// Mock class for testing the abstract OrthogonalBasis
class MockOrthogonalBasis : public OrthogonalBasis {
public:
    MockOrthogonalBasis(std::shared_ptr<ReferenceElement> ref_element, int order)
        : OrthogonalBasis(ref_element, order) {}

    MOCK_METHOD(Eigen::VectorXd, evaluate, (const Eigen::Vector2d& xi), (const, override));
    MOCK_METHOD(Eigen::MatrixXd, evaluate_gradient, (const Eigen::Vector2d& xi), (const, override));

    int compute_n_basis() const override {
        return (get_order() + 1) * (get_order() + 1);  // Simple square for testing
    }
};

TEST(OrthogonalBasisTest, ConstructorAndGetters) {
    auto ref_quad = std::make_shared<ReferenceQuad>();
    int test_order = 2;
    MockOrthogonalBasis basis(ref_quad, test_order);

    EXPECT_EQ(basis.get_order(), test_order);
    EXPECT_EQ(basis.compute_n_basis(), (test_order + 1) * (test_order + 1));
    EXPECT_EQ(basis.get_ref_element().get(), ref_quad.get());
}

TEST(OrthogonalBasisTest, EvaluateCalled) {
    auto ref_quad = std::make_shared<ReferenceQuad>();
    MockOrthogonalBasis basis(ref_quad, 1);

    Eigen::Vector2d point(0.5, -0.5);
    EXPECT_CALL(basis, evaluate(point)).Times(1).WillOnce(Return(Eigen::VectorXd::Zero(4)));

    basis.evaluate(point);
}

TEST(OrthogonalBasisTest, EvaluateGradientCalled) {
    auto ref_quad = std::make_shared<ReferenceQuad>();
    MockOrthogonalBasis basis(ref_quad, 1);

    Eigen::Vector2d point(0.5, -0.5);
    EXPECT_CALL(basis, evaluate_gradient(point))
        .Times(1)
        .WillOnce(Return(Eigen::MatrixXd::Zero(4, 2)));

    basis.evaluate_gradient(point);
}
