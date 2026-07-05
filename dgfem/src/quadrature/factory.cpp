/**
 * @file factory.cpp
 * @brief Implementation of quadrature rules factory
 */

#include "dgfem/quadrature/factory.hpp"

#include <cmath>
#include <numbers>

#include <sstream>
#include <stdexcept>

namespace dgfem {

std::unique_ptr<QuadratureRule> QuadratureFactory::gauss_legendre_1d(int n_points) {
    if (!is_valid_gl_order(n_points)) {
        std::ostringstream oss;
        oss << "Invalid Gauss-Legendre order: " << n_points << ". Must be between 1 and "
            << MAX_GL_POINTS;
        throw std::invalid_argument(oss.str());
    }

    DView1 points, weights;
    compute_gauss_legendre(n_points, points, weights);

    DView2 pts("gl_1d_points", n_points, 1);
    for (int i = 0; i < n_points; ++i) {
        pts(i, 0) = points[i];
    }

    return std::make_unique<QuadratureRule>(std::move(pts), std::move(weights));
}

std::unique_ptr<QuadratureRule> QuadratureFactory::gauss_legendre_quad(int n_points_1d) {
    if (!is_valid_gl_order(n_points_1d)) {
        std::ostringstream oss;
        oss << "Invalid Gauss-Legendre order: " << n_points_1d << ". Must be between 1 and "
            << MAX_GL_POINTS;
        throw std::invalid_argument(oss.str());
    }

    DView1 points_1d, weights_1d;
    compute_gauss_legendre(n_points_1d, points_1d, weights_1d);

    const int total_points = n_points_1d * n_points_1d;
    DView2 points("gl_quad_points", total_points, 2);
    DView1 weights("gl_quad_weights", total_points);

    int idx = 0;
    for (int j = 0; j < n_points_1d; ++j) {
        for (int i = 0; i < n_points_1d; ++i) {
            points(idx, 0) = points_1d[i];
            points(idx, 1) = points_1d[j];
            weights[idx] = weights_1d[i] * weights_1d[j];
            ++idx;
        }
    }

    return std::make_unique<QuadratureRule>(std::move(points), std::move(weights));
}

std::unique_ptr<QuadratureRule> QuadratureFactory::dunavant_triangle(int order) {
    if (!is_valid_dunavant_order(order)) {
        std::ostringstream oss;
        oss << "Invalid Dunavant order: " << order << ". Must be between 0 and "
            << MAX_DUNAVANT_ORDER;
        throw std::invalid_argument(oss.str());
    }

    DView2 points;
    DView1 weights;
    get_dunavant_rule(order, points, weights);
    return std::make_unique<QuadratureRule>(std::move(points), std::move(weights));
}

void QuadratureFactory::compute_gauss_legendre(int n, DView1& points, DView1& weights) {
    points = DView1("gl_points", n);
    weights = DView1("gl_weights", n);

    // Use compile-time constants where possible
    constexpr double sqrt_3_inv =
        1.0 / 1.732050807568877293527446341505872366942805253810380628055;  // 1/sqrt(3)
    constexpr double sqrt_3_5 =
        0.774596669241483377035853079956479922166584341058318165320;  // sqrt(3/5)
    constexpr double w_5_9 = 5.0 / 9.0;
    constexpr double w_8_9 = 8.0 / 9.0;

    switch (n) {
    case 1:
        points[0] = 0.0;
        weights[0] = 2.0;
        break;

    case 2:
        points[0] = -sqrt_3_inv;
        points[1] = sqrt_3_inv;
        weights[0] = weights[1] = 1.0;
        break;

    case 3:
        points[0] = -sqrt_3_5;
        points[1] = 0.0;
        points[2] = sqrt_3_5;
        weights[0] = weights[2] = w_5_9;
        weights[1] = w_8_9;
        break;

    case 4: {
        constexpr double sqrt_6_5 =
            1.095445115010332226913939565601604642850062938142840310385;  // sqrt(6/5)
        const double a = std::sqrt(3.0 / 7.0 - 2.0 / 7.0 * sqrt_6_5);
        const double b = std::sqrt(3.0 / 7.0 + 2.0 / 7.0 * sqrt_6_5);
        constexpr double sqrt_30 = 5.477225575051661134569697828008021321125314690714201551925;
        constexpr double w1 = (18.0 + sqrt_30) / 36.0;
        constexpr double w2 = (18.0 - sqrt_30) / 36.0;

        points[0] = -b;
        points[1] = -a;
        points[2] = a;
        points[3] = b;
        weights[0] = w2;
        weights[1] = w1;
        weights[2] = w1;
        weights[3] = w2;
        break;
    }

    case 5: {
        // 5-point Gauss-Legendre rule (exact for polynomials up to degree 9)
        constexpr double a = 0.906179845938663992797626878299581;
        constexpr double b = 0.538469310105683091036314420700208;
        constexpr double w1 = 0.236926885056189087514264040719917;
        constexpr double w2 = 0.478628670499366468041291514835638;
        constexpr double w3 = 0.568888888888888888888888888888889;

        points[0] = -a;
        points[1] = -b;
        points[2] = 0.0;
        points[3] = b;
        points[4] = a;
        weights[0] = w1;
        weights[1] = w2;
        weights[2] = w3;
        weights[3] = w2;
        weights[4] = w1;
        break;
    }

    case 6: {
        // 6-point Gauss-Legendre rule (exact for polynomials up to degree 11)
        constexpr double a = 0.932469514203152027812301554493995;
        constexpr double b = 0.661209386466264513661399595019906;
        constexpr double c = 0.238619186083196908630501721680711;
        constexpr double w1 = 0.171324492379170345040296142172733;
        constexpr double w2 = 0.360761573048138607569833513837717;
        constexpr double w3 = 0.467913934572691047389870343989551;

        points[0] = -a;
        points[1] = -b;
        points[2] = -c;
        points[3] = c;
        points[4] = b;
        points[5] = a;
        weights[0] = w1;
        weights[1] = w2;
        weights[2] = w3;
        weights[3] = w3;
        weights[4] = w2;
        weights[5] = w1;
        break;
    }

    case 7: {
        // 7-point Gauss-Legendre rule (exact for polynomials up to degree 13)
        constexpr double a = 0.949107912342758524526189684047851;
        constexpr double b = 0.741531185599394439863864773280788;
        constexpr double c = 0.405845151377397166906606412076962;
        constexpr double w1 = 0.129484966168869693270611432679082;
        constexpr double w2 = 0.279705391489276667901467771423780;
        constexpr double w3 = 0.381830050505118944950369775488975;
        constexpr double w4 = 0.417959183673469387755102040816327;

        points[0] = -a;
        points[1] = -b;
        points[2] = -c;
        points[3] = 0.0;
        points[4] = c;
        points[5] = b;
        points[6] = a;
        weights[0] = w1;
        weights[1] = w2;
        weights[2] = w3;
        weights[3] = w4;
        weights[4] = w3;
        weights[5] = w2;
        weights[6] = w1;
        break;
    }

    case 8: {
        // 8-point Gauss-Legendre rule (exact for polynomials up to degree 15)
        constexpr double a = 0.960289856497536231683560868569473;
        constexpr double b = 0.796666477413626739591553936475830;
        constexpr double c = 0.525532409916328985817739049189246;
        constexpr double d = 0.183434642495649804939476142360183;
        constexpr double w1 = 0.101228536290376259152531354309962;
        constexpr double w2 = 0.222381034453374470544355994426240;
        constexpr double w3 = 0.313706645877887287337962201986602;
        constexpr double w4 = 0.362683783378361982965150449277196;

        points[0] = -a;
        points[1] = -b;
        points[2] = -c;
        points[3] = -d;
        points[4] = d;
        points[5] = c;
        points[6] = b;
        points[7] = a;
        weights[0] = w1;
        weights[1] = w2;
        weights[2] = w3;
        weights[3] = w4;
        weights[4] = w4;
        weights[5] = w3;
        weights[6] = w2;
        weights[7] = w1;
        break;
    }

    default: {
        std::ostringstream oss;
        oss << "Gauss-Legendre quadrature with " << n << " points not implemented";
        throw std::invalid_argument(oss.str());
    }
    }
}

void QuadratureFactory::get_dunavant_rule(int order, DView2& points, DView1& weights) {
    // Intrepid2 cubature rules for triangles
    // Reference triangle: {(0,0), (1,0), (0,1)}
    // Data from: Intrepid2_CubatureDirectTriDefaultDef.hpp
    constexpr double one_third = 1.0 / 3.0;
    constexpr double one_half = 0.5;
    constexpr double one_sixth = 1.0 / 6.0;

    switch (order) {
    case 0:
    case 1: {
        // 1-point rule (degree 1)
        points = DView2("dunavant_points", 1, 2);
        weights = DView1("dunavant_weights", 1);
        points(0, 0) = one_third;
        points(0, 1) = one_third;
        weights[0] = one_half;
        break;
    }

    case 2: {
        // 3-point rule (degree 2)
        points = DView2("dunavant_points", 3, 2);
        weights = DView1("dunavant_weights", 3);
        points(0, 0) = 1.0 / 6.0;
        points(0, 1) = 1.0 / 6.0;
        points(1, 0) = 1.0 / 6.0;
        points(1, 1) = 2.0 / 3.0;
        points(2, 0) = 2.0 / 3.0;
        points(2, 1) = 1.0 / 6.0;
        weights[0] = weights[1] = weights[2] = one_sixth;
        break;
    }

    case 3: {
        // 4-point rule (degree 3)
        points = DView2("dunavant_points", 4, 2);
        weights = DView1("dunavant_weights", 4);
        points(0, 0) = one_third;
        points(0, 1) = one_third;
        points(1, 0) = 0.2;
        points(1, 1) = 0.2;
        points(2, 0) = 0.2;
        points(2, 1) = 0.6;
        points(3, 0) = 0.6;
        points(3, 1) = 0.2;

        weights[0] = -9.0 / 32.0;
        weights[1] = 25.0 / 96.0;
        weights[2] = 25.0 / 96.0;
        weights[3] = 25.0 / 96.0;
        break;
    }

    case 4: {
        // 6-point rule (degree 4)
        points = DView2("dunavant_points", 6, 2);
        weights = DView1("dunavant_weights", 6);

        points(0, 0) = 4.4594849091596487577332043252695176084800e-1;
        points(0, 1) = 4.4594849091596487577332043252695176084800e-1;
        points(1, 0) = 4.4594849091596487577332043252695176084800e-1;
        points(1, 1) = 1.0810301816807024845335913494609647830400e-1;
        points(2, 0) = 1.0810301816807024845335913494609647830400e-1;
        points(2, 1) = 4.4594849091596487577332043252695176084800e-1;
        points(3, 0) = 9.1576213509770745141706146463816673424873e-2;
        points(3, 1) = 9.1576213509770745141706146463816673424873e-2;
        points(4, 0) = 9.1576213509770745141706146463816673424873e-2;
        points(4, 1) = 8.1684757298045850971658770707236665315025e-1;
        points(5, 0) = 8.1684757298045850971658770707236665315025e-1;
        points(5, 1) = 9.1576213509770745141706146463816673424873e-2;

        weights[0] = 1.1169079483900573053953942803038657817028e-1;
        weights[1] = 1.1169079483900573053953942803038657817028e-1;
        weights[2] = 1.1169079483900573053953942803038657817028e-1;
        weights[3] = 5.4975871827660936395761590801109078322738e-2;
        weights[4] = 5.4975871827660936395761590801109078322738e-2;
        weights[5] = 5.4975871827660936395761590801109078322738e-2;
        break;
    }

    case 5: {
        // 7-point rule (degree 5)
        points = DView2("dunavant_points", 7, 2);
        weights = DView1("dunavant_weights", 7);

        points(0, 0) = 3.3333333333333333333333333333333333333333e-1;
        points(0, 1) = 3.3333333333333333333333333333333333333333e-1;
        points(1, 0) = 1.0128650732345633880098736191512382805558e-1;
        points(1, 1) = 1.0128650732345633880098736191512382805558e-1;
        points(2, 0) = 7.9742698535308732239802527616975234388885e-1;
        points(2, 1) = 1.0128650732345633880098736191512382805558e-1;
        points(3, 0) = 1.0128650732345633880098736191512382805558e-1;
        points(3, 1) = 7.9742698535308732239802527616975234388885e-1;
        points(4, 0) = 4.7014206410511508977044120951344760051585e-1;
        points(4, 1) = 4.7014206410511508977044120951344760051585e-1;
        points(5, 0) = 5.9715871789769820459117580973104798968293e-2;
        points(5, 1) = 4.7014206410511508977044120951344760051585e-1;
        points(6, 0) = 4.7014206410511508977044120951344760051585e-1;
        points(6, 1) = 5.9715871789769820459117580973104798968293e-2;

        weights[0] = 1.1250000000000000000000000000000000000000e-1;
        weights[1] = 6.2969590272413576297841972750090666828820e-2;
        weights[2] = 6.2969590272413576297841972750090666828820e-2;
        weights[3] = 6.2969590272413576297841972750090666828820e-2;
        weights[4] = 6.6197076394253090368824693916575999837847e-2;
        weights[5] = 6.6197076394253090368824693916575999837847e-2;
        weights[6] = 6.6197076394253090368824693916575999837847e-2;
        break;
    }

    case 6: {
        // 12-point rule (degree 6)
        points = DView2("dunavant_points", 12, 2);
        weights = DView1("dunavant_weights", 12);

        points(0, 0) = 6.3089014491502228340331602870819157341003e-2;
        points(0, 1) = 6.3089014491502228340331602870819157341003e-2;
        points(1, 0) = 6.3089014491502228340331602870819157341003e-2;
        points(1, 1) = 8.7382197101699554331933679425836168531799e-1;
        points(2, 0) = 8.7382197101699554331933679425836168531799e-1;
        points(2, 1) = 6.3089014491502228340331602870819157341003e-2;
        points(3, 0) = 2.4928674517091042129163855310701907608796e-1;
        points(3, 1) = 2.4928674517091042129163855310701907608796e-1;
        points(4, 0) = 2.4928674517091042129163855310701907608796e-1;
        points(4, 1) = 5.0142650965817915741672289378596184782407e-1;
        points(5, 0) = 5.0142650965817915741672289378596184782407e-1;
        points(5, 1) = 2.4928674517091042129163855310701907608796e-1;
        points(6, 0) = 3.1035245103378441286759566723265117140082e-1;
        points(6, 1) = 5.3145049844816939902261738355299128796183e-2;
        points(7, 0) = 6.3650249912139865667831578657006182134850e-1;
        points(7, 1) = 3.1035245103378439596843454179854003165774e-1;
        points(8, 0) = 5.3145049844816930454088546197287007250682e-2;
        points(8, 1) = 6.3650249912139866412930371984616083954608e-1;
        points(9, 0) = 5.3145049844816939902261738355299128796183e-2;
        points(9, 1) = 3.1035245103378441286759566723265117140082e-1;
        points(10, 0) = 6.3650249912139866412930371984616083954608e-1;
        points(10, 1) = 5.3145049844816930454088546197287007250682e-2;
        points(11, 0) = 3.1035245103378439596843454179854003165774e-1;
        points(11, 1) = 6.3650249912139865667831578657006182134850e-1;

        weights[0] = 2.5422453185103408460468404553434492023395e-2;
        weights[1] = 2.5422453185103408460468404553434492023395e-2;
        weights[2] = 2.5422453185103408460468404553434492023395e-2;
        weights[3] = 5.8393137863189683012644805692789720663043e-2;
        weights[4] = 5.8393137863189683012644805692789720663043e-2;
        weights[5] = 5.8393137863189683012644805692789720663043e-2;
        weights[6] = 4.1425537809186787596776728210221226990114e-2;
        weights[7] = 4.1425537809186787596776728210221226990114e-2;
        weights[8] = 4.1425537809186787596776728210221226990114e-2;
        weights[9] = 4.1425537809186787596776728210221226990114e-2;
        weights[10] = 4.1425537809186787596776728210221226990114e-2;
        weights[11] = 4.1425537809186787596776728210221226990114e-2;
        break;
    }

    case 7: {
        // 13-point rule (degree 7)
        points = DView2("dunavant_points", 13, 2);
        weights = DView1("dunavant_weights", 13);

        points(0, 0) = 3.33333333333333e-1;
        points(0, 1) = 3.33333333333333e-1;
        points(1, 0) = 2.60345966079040e-1;
        points(1, 1) = 2.60345966079040e-1;
        points(2, 0) = 2.60345966079040e-1;
        points(2, 1) = 4.79308067841920e-1;
        points(3, 0) = 4.79308067841920e-1;
        points(3, 1) = 2.60345966079040e-1;
        points(4, 0) = 6.51301029022160e-2;
        points(4, 1) = 6.51301029022160e-2;
        points(5, 0) = 6.51301029022160e-2;
        points(5, 1) = 8.69739794195568e-1;
        points(6, 0) = 8.69739794195568e-1;
        points(6, 1) = 6.51301029022160e-2;
        points(7, 0) = 3.12865496004874e-1;
        points(7, 1) = 6.38444188569810e-1;
        points(8, 0) = 6.38444188569810e-1;
        points(8, 1) = 4.86903154253160e-2;
        points(9, 0) = 4.86903154253160e-2;
        points(9, 1) = 3.12865496004874e-1;
        points(10, 0) = 3.12865496004874e-1;
        points(10, 1) = 4.86903154253160e-2;
        points(11, 0) = 6.38444188569810e-1;
        points(11, 1) = 3.12865496004874e-1;
        points(12, 0) = 4.86903154253160e-2;
        points(12, 1) = 6.38444188569810e-1;

        weights[0] = -7.47850222338410e-2;
        weights[1] = 8.78076287166040e-2;
        weights[2] = 8.78076287166040e-2;
        weights[3] = 8.78076287166040e-2;
        weights[4] = 2.66736178044190e-2;
        weights[5] = 2.66736178044190e-2;
        weights[6] = 2.66736178044190e-2;
        weights[7] = 3.85568804451285e-2;
        weights[8] = 3.85568804451285e-2;
        weights[9] = 3.85568804451285e-2;
        weights[10] = 3.85568804451285e-2;
        weights[11] = 3.85568804451285e-2;
        weights[12] = 3.85568804451285e-2;
        break;
    }

    case 8: {
        // 16-point rule (degree 8)
        points = DView2("dunavant_points", 16, 2);
        weights = DView1("dunavant_weights", 16);

        points(0, 0) = 3.33333333333333e-1;
        points(0, 1) = 3.33333333333333e-1;
        points(1, 0) = 4.59292588292723e-1;
        points(1, 1) = 4.59292588292723e-1;
        points(2, 0) = 4.59292588292723e-1;
        points(2, 1) = 8.14148234145540e-2;
        points(3, 0) = 8.14148234145540e-2;
        points(3, 1) = 4.59292588292723e-1;
        points(4, 0) = 1.70569307751760e-1;
        points(4, 1) = 1.70569307751760e-1;
        points(5, 0) = 1.70569307751760e-1;
        points(5, 1) = 6.58861384496480e-1;
        points(6, 0) = 6.58861384496480e-1;
        points(6, 1) = 1.70569307751760e-1;
        points(7, 0) = 5.05472283170310e-2;
        points(7, 1) = 5.05472283170310e-2;
        points(8, 0) = 5.05472283170310e-2;
        points(8, 1) = 8.98905543365938e-1;
        points(9, 0) = 8.98905543365938e-1;
        points(9, 1) = 5.05472283170310e-2;
        points(10, 0) = 2.63112829634638e-1;
        points(10, 1) = 7.28492392955404e-1;
        points(11, 0) = 7.28492392955404e-1;
        points(11, 1) = 8.39477740995798e-3;
        points(12, 0) = 8.39477740995798e-3;
        points(12, 1) = 2.63112829634638e-1;
        points(13, 0) = 2.63112829634638e-1;
        points(13, 1) = 8.39477740995798e-3;
        points(14, 0) = 7.28492392955404e-1;
        points(14, 1) = 2.63112829634638e-1;
        points(15, 0) = 8.39477740995798e-3;
        points(15, 1) = 7.28492392955404e-1;

        weights[0] = 7.21578038388935e-2;
        weights[1] = 4.75458171336425e-2;
        weights[2] = 4.75458171336425e-2;
        weights[3] = 4.75458171336425e-2;
        weights[4] = 5.16086852673590e-2;
        weights[5] = 5.16086852673590e-2;
        weights[6] = 5.16086852673590e-2;
        weights[7] = 1.62292488115990e-2;
        weights[8] = 1.62292488115988e-2;
        weights[9] = 1.62292488115990e-2;
        weights[10] = 1.36151570872175e-2;
        weights[11] = 1.36151570872175e-2;
        weights[12] = 1.36151570872175e-2;
        weights[13] = 1.36151570872175e-2;
        weights[14] = 1.36151570872175e-2;
        weights[15] = 1.36151570872175e-2;
        break;
    }

    case 9:
    case 10: {
        // 19-point rule (degree 9)
        points = DView2("dunavant_points", 19, 2);
        weights = DView1("dunavant_weights", 19);

        points(0, 0) = 3.33333333333333e-1;
        points(0, 1) = 3.33333333333333e-1;
        points(1, 0) = 4.89682519198738e-1;
        points(1, 1) = 4.89682519198738e-1;
        points(2, 0) = 4.89682519198738e-1;
        points(2, 1) = 2.06349616025250e-2;
        points(3, 0) = 2.06349616025250e-2;
        points(3, 1) = 4.89682519198738e-1;
        points(4, 0) = 4.37089591492937e-1;
        points(4, 1) = 4.37089591492937e-1;
        points(5, 0) = 4.37089591492937e-1;
        points(5, 1) = 1.25820817014127e-1;
        points(6, 0) = 1.25820817014127e-1;
        points(6, 1) = 4.37089591492937e-1;
        points(7, 0) = 1.88203535619033e-1;
        points(7, 1) = 1.88203535619033e-1;
        points(8, 0) = 1.88203535619033e-1;
        points(8, 1) = 6.23592928761935e-1;
        points(9, 0) = 6.23592928761935e-1;
        points(9, 1) = 1.88203535619033e-1;
        points(10, 0) = 4.47295133944530e-2;
        points(10, 1) = 4.47295133944530e-2;
        points(11, 0) = 4.47295133944530e-2;
        points(11, 1) = 9.10540973211095e-1;
        points(12, 0) = 9.10540973211095e-1;
        points(12, 1) = 4.47295133944530e-2;
        points(13, 0) = 2.21962989160766e-1;
        points(13, 1) = 7.41198598784498e-1;
        points(14, 0) = 7.41198598784498e-1;
        points(14, 1) = 3.68384120547360e-2;
        points(15, 0) = 3.68384120547360e-2;
        points(15, 1) = 2.21962989160766e-1;
        points(16, 0) = 2.21962989160766e-1;
        points(16, 1) = 3.68384120547360e-2;
        points(17, 0) = 7.41198598784498e-1;
        points(17, 1) = 2.21962989160766e-1;
        points(18, 0) = 3.68384120547360e-2;
        points(18, 1) = 7.41198598784498e-1;

        weights[0] = 4.85678981413995e-2;
        weights[1] = 1.56673501135695e-2;
        weights[2] = 1.56673501135695e-2;
        weights[3] = 1.56673501135695e-2;
        weights[4] = 3.89137705023870e-2;
        weights[5] = 3.89137705023870e-2;
        weights[6] = 3.89137705023870e-2;
        weights[7] = 3.98238694636050e-2;
        weights[8] = 3.98238694636050e-2;
        weights[9] = 3.98238694636050e-2;
        weights[10] = 1.27888378293490e-2;
        weights[11] = 1.27888378293490e-2;
        weights[12] = 1.27888378293490e-2;
        weights[13] = 2.16417696886445e-2;
        weights[14] = 2.16417696886445e-2;
        weights[15] = 2.16417696886445e-2;
        weights[16] = 2.16417696886445e-2;
        weights[17] = 2.16417696886445e-2;
        weights[18] = 2.16417696886445e-2;
        break;
    }

    case 11:
    case 12: {
        // 42-point rule (degree 14) from Intrepid2
        points = DView2("dunavant_points", 42, 2);
        weights = DView1("dunavant_weights", 42);

        points(0, 0) = 4.88963910362179e-1;
        points(0, 1) = 4.88963910362179e-1;
        points(1, 0) = 4.88963910362179e-1;
        points(1, 1) = 2.20721792756430e-2;
        points(2, 0) = 2.20721792756430e-2;
        points(2, 1) = 4.88963910362179e-1;
        points(3, 0) = 4.17644719340454e-1;
        points(3, 1) = 4.17644719340454e-1;
        points(4, 0) = 4.17644719340454e-1;
        points(4, 1) = 1.64710561319092e-1;
        points(5, 0) = 1.64710561319092e-1;
        points(5, 1) = 4.17644719340454e-1;
        points(6, 0) = 2.73477528308839e-1;
        points(6, 1) = 2.73477528308839e-1;
        points(7, 0) = 2.73477528308839e-1;
        points(7, 1) = 4.53044943382323e-1;
        points(8, 0) = 4.53044943382323e-1;
        points(8, 1) = 2.73477528308839e-1;
        points(9, 0) = 1.77205532412543e-1;
        points(9, 1) = 1.77205532412543e-1;
        points(10, 0) = 1.77205532412543e-1;
        points(10, 1) = 6.45588935174913e-1;
        points(11, 0) = 6.45588935174913e-1;
        points(11, 1) = 1.77205532412543e-1;
        points(12, 0) = 6.17998830908730e-2;
        points(12, 1) = 6.17998830908730e-2;
        points(13, 0) = 6.17998830908730e-2;
        points(13, 1) = 8.76400233818255e-1;
        points(14, 0) = 8.76400233818255e-1;
        points(14, 1) = 6.17998830908730e-2;
        points(15, 0) = 1.93909612487010e-2;
        points(15, 1) = 1.93909612487010e-2;
        points(16, 0) = 1.93909612487010e-2;
        points(16, 1) = 9.61218077502598e-1;
        points(17, 0) = 9.61218077502598e-1;
        points(17, 1) = 1.93909612487010e-2;
        points(18, 0) = 1.72266687821356e-1;
        points(18, 1) = 7.70608554774996e-1;
        points(19, 0) = 7.70608554774996e-1;
        points(19, 1) = 5.71247574036480e-2;
        points(20, 0) = 5.71247574036480e-2;
        points(20, 1) = 1.72266687821356e-1;
        points(21, 0) = 1.72266687821356e-1;
        points(21, 1) = 5.71247574036480e-2;
        points(22, 0) = 7.70608554774996e-1;
        points(22, 1) = 1.72266687821356e-1;
        points(23, 0) = 5.71247574036480e-2;
        points(23, 1) = 7.70608554774996e-1;
        points(24, 0) = 3.36861459796345e-1;
        points(24, 1) = 5.70222290846683e-1;
        points(25, 0) = 5.70222290846683e-1;
        points(25, 1) = 9.29162493569720e-2;
        points(26, 0) = 9.29162493569720e-2;
        points(26, 1) = 3.36861459796345e-1;
        points(27, 0) = 3.36861459796345e-1;
        points(27, 1) = 9.29162493569720e-2;
        points(28, 0) = 5.70222290846683e-1;
        points(28, 1) = 3.36861459796345e-1;
        points(29, 0) = 9.29162493569720e-2;
        points(29, 1) = 5.70222290846683e-1;
        points(30, 0) = 2.98372882136258e-1;
        points(30, 1) = 6.86980167808088e-1;
        points(31, 0) = 6.86980167808088e-1;
        points(31, 1) = 1.46469500556540e-2;
        points(32, 0) = 1.46469500556540e-2;
        points(32, 1) = 2.98372882136258e-1;
        points(33, 0) = 2.98372882136258e-1;
        points(33, 1) = 1.46469500556540e-2;
        points(34, 0) = 6.86980167808088e-1;
        points(34, 1) = 2.98372882136258e-1;
        points(35, 0) = 1.46469500556540e-2;
        points(35, 1) = 6.86980167808088e-1;
        points(36, 0) = 1.18974497696957e-1;
        points(36, 1) = 8.79757171370171e-1;
        points(37, 0) = 8.79757171370171e-1;
        points(37, 1) = 1.26833093287199e-3;
        points(38, 0) = 1.26833093287199e-3;
        points(38, 1) = 1.18974497696957e-1;
        points(39, 0) = 1.18974497696957e-1;
        points(39, 1) = 1.26833093287199e-3;
        points(40, 0) = 8.79757171370171e-1;
        points(40, 1) = 1.18974497696957e-1;
        points(41, 0) = 1.26833093287199e-3;
        points(41, 1) = 8.79757171370171e-1;

        weights[0] = 1.09417906847145e-2;
        weights[1] = 1.09417906847145e-2;
        weights[2] = 1.09417906847145e-2;
        weights[3] = 1.63941767720625e-2;
        weights[4] = 1.63941767720625e-2;
        weights[5] = 1.63941767720625e-2;
        weights[6] = 2.58870522536460e-2;
        weights[7] = 2.58870522536460e-2;
        weights[8] = 2.58870522536460e-2;
        weights[9] = 2.10812943684965e-2;
        weights[10] = 2.10812943684965e-2;
        weights[11] = 2.10812943684965e-2;
        weights[12] = 7.21684983488850e-3;
        weights[13] = 7.21684983488850e-3;
        weights[14] = 7.21684983488850e-3;
        weights[15] = 2.46170180120000e-3;
        weights[16] = 2.46170180120000e-3;
        weights[17] = 2.46170180120000e-3;
        weights[18] = 1.23328766062820e-2;
        weights[19] = 1.23328766062820e-2;
        weights[20] = 1.23328766062820e-2;
        weights[21] = 1.23328766062820e-2;
        weights[22] = 1.23328766062820e-2;
        weights[23] = 1.23328766062820e-2;
        weights[24] = 1.92857553935305e-2;
        weights[25] = 1.92857553935305e-2;
        weights[26] = 1.92857553935305e-2;
        weights[27] = 1.92857553935305e-2;
        weights[28] = 1.92857553935305e-2;
        weights[29] = 1.92857553935305e-2;
        weights[30] = 7.21815405676700e-3;
        weights[31] = 7.21815405676700e-3;
        weights[32] = 7.21815405676700e-3;
        weights[33] = 7.21815405676700e-3;
        weights[34] = 7.21815405676700e-3;
        weights[35] = 7.21815405676700e-3;
        weights[36] = 2.50511441925050e-3;
        weights[37] = 2.50511441925050e-3;
        weights[38] = 2.50511441925050e-3;
        weights[39] = 2.50511441925050e-3;
        weights[40] = 2.50511441925050e-3;
        weights[41] = 2.50511441925050e-3;
        break;
    }

    case 13:
    case 14: {
        // Use the same 42-point rule as cases 11-12 (degree 14)
        // Redirect to avoid code duplication
        get_dunavant_rule(12, points, weights);
        break;
    }

    default: {
        std::ostringstream oss;
        oss << "Dunavant triangle quadrature for order " << order << " not implemented";
        throw std::invalid_argument(oss.str());
    }
    }
}

}  // namespace dgfem