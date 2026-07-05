/**
 * @file kokkos_math.hpp
 * @brief Small local (per-element) dense math on Kokkos types, replacing what Eigen's
 * expression templates used to provide for 2D points/vectors and 2x2 Jacobians. Kokkos
 * itself has no built-in small-matrix linear algebra -- these are the hand-written
 * replacements the migration plan called for.
 *
 * Convention: a 2D point/vector is Kokkos::Array<double, 2> ("Vec2"); a dynamically-sized
 * vector is Kokkos::View<double*>; a dynamically-sized (n_rows x n_cols) matrix is
 * Kokkos::View<double**> (row-major access via operator(i,j)).
 */
#pragma once

#include <KokkosBatched_InverseLU_Decl.hpp>
#include <KokkosBlas2_gemv.hpp>
#include <KokkosBlas3_gemm.hpp>
#include <KokkosBlas_util.hpp>
#include <Kokkos_Core.hpp>
#include <cmath>

#include <stdexcept>

namespace dgfem {

using Vec2 = Kokkos::Array<double, 2>;
using Vec4 = Kokkos::Array<double, 4>;
using DView1 = Kokkos::View<double*, Kokkos::HostSpace>;
using DView2 = Kokkos::View<double**, Kokkos::HostSpace>;
using IView1 = Kokkos::View<int*, Kokkos::HostSpace>;
using IView2 = Kokkos::View<int**, Kokkos::HostSpace>;

// --- Vec2 helpers -----------------------------------------------------------------------

[[nodiscard]] inline Vec2 operator+(const Vec2& a, const Vec2& b) {
    return {a[0] + b[0], a[1] + b[1]};
}
[[nodiscard]] inline Vec2 operator-(const Vec2& a, const Vec2& b) {
    return {a[0] - b[0], a[1] - b[1]};
}
[[nodiscard]] inline Vec2 operator*(double s, const Vec2& a) {
    return {s * a[0], s * a[1]};
}
[[nodiscard]] inline Vec2 operator*(const Vec2& a, double s) {
    return s * a;
}
[[nodiscard]] inline Vec2 operator-(const Vec2& a) {
    return {-a[0], -a[1]};
}
inline Vec2& operator+=(Vec2& a, const Vec2& b) {
    a[0] += b[0];
    a[1] += b[1];
    return a;
}
inline Vec2& operator/=(Vec2& a, double s) {
    a[0] /= s;
    a[1] /= s;
    return a;
}

[[nodiscard]] inline double dot(const Vec2& a, const Vec2& b) {
    return a[0] * b[0] + a[1] * b[1];
}
[[nodiscard]] inline double norm(const Vec2& a) {
    return std::sqrt(dot(a, a));
}
[[nodiscard]] inline Vec2 normalized(const Vec2& a) {
    const double n = norm(a);
    return {a[0] / n, a[1] / n};
}

// --- Vec4 helpers (conserved/primitive-variable tuples, e.g. [rho, rho*u, rho*v, E]) --

[[nodiscard]] inline Vec4 operator+(const Vec4& a, const Vec4& b) {
    return {a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]};
}
[[nodiscard]] inline Vec4 operator-(const Vec4& a, const Vec4& b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]};
}
[[nodiscard]] inline Vec4 operator*(double s, const Vec4& a) {
    return {s * a[0], s * a[1], s * a[2], s * a[3]};
}
[[nodiscard]] inline Vec4 operator*(const Vec4& a, double s) {
    return s * a;
}
[[nodiscard]] inline Vec4 operator-(const Vec4& a) {
    return {-a[0], -a[1], -a[2], -a[3]};
}
inline Vec4& operator+=(Vec4& a, const Vec4& b) {
    a[0] += b[0];
    a[1] += b[1];
    a[2] += b[2];
    a[3] += b[3];
    return a;
}
inline Vec4& operator-=(Vec4& a, const Vec4& b) {
    a[0] -= b[0];
    a[1] -= b[1];
    a[2] -= b[2];
    a[3] -= b[3];
    return a;
}

[[nodiscard]] inline double dot(const Vec4& a, const Vec4& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}
[[nodiscard]] inline double norm(const Vec4& a) {
    return std::sqrt(dot(a, a));
}

// --- Mat2: a 2x2 matrix stored row-major as {m00, m01, m10, m11} -----------------------

struct Mat2 {
    double m00, m01, m10, m11;

    [[nodiscard]] double det() const { return m00 * m11 - m01 * m10; }

    [[nodiscard]] Mat2 transpose() const { return {m00, m10, m01, m11}; }

    [[nodiscard]] Mat2 inverse() const {
        const double d = det();
        if (std::abs(d) < 1e-14) {
            throw std::runtime_error("Mat2::inverse() called on a singular (or near-singular) "
                                     "matrix -- would silently produce Inf/NaN");
        }
        return {m11 / d, -m01 / d, -m10 / d, m00 / d};
    }

    [[nodiscard]] Vec2 apply(const Vec2& v) const {
        return {m00 * v[0] + m01 * v[1], m10 * v[0] + m11 * v[1]};
    }

    [[nodiscard]] double operator()(int i, int j) const {
        if (i == 0)
            return j == 0 ? m00 : m01;
        return j == 0 ? m10 : m11;
    }

    [[nodiscard]] static Mat2 identity() { return {1.0, 0.0, 0.0, 1.0}; }
};

[[nodiscard]] inline Mat2 operator*(const Mat2& a, const Mat2& b) {
    return {a.m00 * b.m00 + a.m01 * b.m10, a.m00 * b.m01 + a.m01 * b.m11,
            a.m10 * b.m00 + a.m11 * b.m10, a.m10 * b.m01 + a.m11 * b.m11};
}
[[nodiscard]] inline Mat2 operator-(const Mat2& a, const Mat2& b) {
    return {a.m00 - b.m00, a.m01 - b.m01, a.m10 - b.m10, a.m11 - b.m11};
}
/// Frobenius norm of a 2x2 matrix (replaces Eigen's `.norm()` on a Matrix2d).
[[nodiscard]] inline double norm(const Mat2& m) {
    return std::sqrt(m.m00 * m.m00 + m.m01 * m.m01 + m.m10 * m.m10 + m.m11 * m.m11);
}

// --- DView2 row helpers (Kokkos::subview replaces Eigen's .row()) ---------------------

/// Copy row `i` of a (n_rows x 2) view into a Vec2.
[[nodiscard]] inline Vec2 row2(const DView2& m, int i) {
    return {m(i, 0), m(i, 1)};
}

/// Read a (2,1)-shaped view (e.g. a stored "normal" element/face-data entry) as a Vec2.
[[nodiscard]] inline Vec2 to_vec2(const DView2& col) {
    return {col(0, 0), col(1, 0)};
}

/// Read a (1,1)-shaped view (e.g. a stored scalar element/face-data entry) as a double.
[[nodiscard]] inline double scalar_of(const DView2& m) {
    return m(0, 0);
}

/// Write a Vec2 into row `i` of a (n_rows x 2) view.
inline void set_row2(DView2& m, int i, const Vec2& v) {
    m(i, 0) = v[0];
    m(i, 1) = v[1];
}

/// L2 norm of an arbitrary-length vector view (distinct overload from the fixed-size Vec2/Vec4
/// norm() above).
[[nodiscard]] inline double norm(const DView1& v) {
    double s = 0.0;
    for (int i = 0; i < static_cast<int>(v.extent(0)); ++i) {
        s += v[i] * v[i];
    }
    return std::sqrt(s);
}

/// Dot product of two arbitrary-length vector views (replaces Eigen's `a.dot(b)`).
[[nodiscard]] inline double dot(const DView1& a, const DView1& b) {
    double s = 0.0;
    for (int i = 0; i < static_cast<int>(a.extent(0)); ++i) {
        s += a[i] * b[i];
    }
    return s;
}

/// True if every entry is finite (replaces Eigen's `.allFinite()`).
[[nodiscard]] inline bool all_finite(const DView1& v) {
    for (int i = 0; i < static_cast<int>(v.extent(0)); ++i) {
        if (!std::isfinite(v[i]))
            return false;
    }
    return true;
}

/// True if every entry is finite (replaces Eigen's `.allFinite()`), matrix overload.
[[nodiscard]] inline bool all_finite(const DView2& m) {
    for (int i = 0; i < static_cast<int>(m.extent(0)); ++i) {
        for (int j = 0; j < static_cast<int>(m.extent(1)); ++j) {
            if (!std::isfinite(m(i, j)))
                return false;
        }
    }
    return true;
}

/// Extract row `i` of an (n_rows x n_cols) view as a fresh DView1 (arbitrary width; row2()
/// is the 2-wide specialization for physical/reference points).
[[nodiscard]] inline DView1 row_of(const DView2& m, int i) {
    DView1 r("row", m.extent(1));
    for (int j = 0; j < static_cast<int>(m.extent(1)); ++j) {
        r[j] = m(i, j);
    }
    return r;
}

/// Extract column `j` of an (n_rows x n_cols) view as a fresh DView1.
[[nodiscard]] inline DView1 col_of(const DView2& m, int j) {
    DView1 c("col", m.extent(0));
    for (int i = 0; i < static_cast<int>(m.extent(0)); ++i) {
        c[i] = m(i, j);
    }
    return c;
}

/// Element-wise M += alpha * X (replaces Eigen's `M += X` on same-shaped matrices).
inline void axpy(DView2& M, double alpha, const DView2& X) {
    for (int i = 0; i < static_cast<int>(M.extent(0)); ++i) {
        for (int j = 0; j < static_cast<int>(M.extent(1)); ++j) {
            M(i, j) += alpha * X(i, j);
        }
    }
}

/// Trace of a square matrix view (replaces Eigen's `.trace()`).
[[nodiscard]] inline double trace(const DView2& m) {
    double s = 0.0;
    for (int i = 0; i < static_cast<int>(m.extent(0)); ++i) {
        s += m(i, i);
    }
    return s;
}

/// Frobenius norm of an (n x n) matrix view (replaces Eigen's `.norm()` on a MatrixXd).
[[nodiscard]] inline double frobenius_norm(const DView2& m) {
    double s = 0.0;
    for (int i = 0; i < static_cast<int>(m.extent(0)); ++i) {
        for (int j = 0; j < static_cast<int>(m.extent(1)); ++j) {
            s += m(i, j) * m(i, j);
        }
    }
    return std::sqrt(s);
}

/// Frobenius norm of (M - M^T), i.e. how far a square matrix is from symmetric.
[[nodiscard]] inline double asymmetry_norm(const DView2& m) {
    double s = 0.0;
    for (int i = 0; i < static_cast<int>(m.extent(0)); ++i) {
        for (int j = 0; j < static_cast<int>(m.extent(1)); ++j) {
            const double d = m(i, j) - m(j, i);
            s += d * d;
        }
    }
    return std::sqrt(s);
}

/// Rank-1 update M += alpha * v * u^T (outer product accumulate). Genuine matrix-matrix and
/// matrix-vector products use KokkosBlas::gemm/gemv instead (see below) -- this one is just
/// element-wise indexing, not real linear algebra, so it isn't worth a BLAS call.
inline void outer_add(DView2& M, double alpha, const DView1& v, const DView1& u) {
    if (static_cast<int>(M.extent(0)) != static_cast<int>(v.extent(0)) ||
        static_cast<int>(M.extent(1)) != static_cast<int>(u.extent(0))) {
        throw std::invalid_argument("outer_add: M's shape must be (v.extent(0), u.extent(0))");
    }
    for (int i = 0; i < static_cast<int>(v.extent(0)); ++i) {
        for (int j = 0; j < static_cast<int>(u.extent(0)); ++j) {
            M(i, j) += alpha * v[i] * u[j];
        }
    }
}

/// C = alpha * A * B + beta * C, via KokkosBlas::gemm (Trilinos-provided, not hand-rolled).
inline void gemm(char transA, char transB, double alpha, const DView2& A, const DView2& B,
                 double beta, DView2& C) {
    const int a_rows =
        (transA == 'N') ? static_cast<int>(A.extent(0)) : static_cast<int>(A.extent(1));
    const int a_cols =
        (transA == 'N') ? static_cast<int>(A.extent(1)) : static_cast<int>(A.extent(0));
    const int b_rows =
        (transB == 'N') ? static_cast<int>(B.extent(0)) : static_cast<int>(B.extent(1));
    const int b_cols =
        (transB == 'N') ? static_cast<int>(B.extent(1)) : static_cast<int>(B.extent(0));
    if (a_cols != b_rows || a_rows != static_cast<int>(C.extent(0)) ||
        b_cols != static_cast<int>(C.extent(1))) {
        throw std::invalid_argument("gemm: incompatible shapes for C = alpha*op(A)*op(B) + beta*C");
    }
    const char ta[2] = {transA, '\0'};
    const char tb[2] = {transB, '\0'};
    KokkosBlas::gemm(ta, tb, alpha, A, B, beta, C);
}

/// y = alpha * A * x + beta * y, via KokkosBlas::gemv (Trilinos-provided, not hand-rolled).
inline void gemv(char trans, double alpha, const DView2& A, const DView1& x, double beta,
                 DView1& y) {
    const int a_rows =
        (trans == 'N') ? static_cast<int>(A.extent(0)) : static_cast<int>(A.extent(1));
    const int a_cols =
        (trans == 'N') ? static_cast<int>(A.extent(1)) : static_cast<int>(A.extent(0));
    if (a_cols != static_cast<int>(x.extent(0)) || a_rows != static_cast<int>(y.extent(0))) {
        throw std::invalid_argument("gemv: incompatible shapes for y = alpha*op(A)*x + beta*y");
    }
    const char t[2] = {trans, '\0'};
    KokkosBlas::gemv(t, alpha, A, x, beta, y);
}

/// Dense inverse of a small square matrix, via KokkosBatched::SerialInverseLU (Trilinos-
/// provided LU-based inverse, not hand-rolled). Used for mass-matrix blocks, which are
/// n_basis x n_basis (not always 2x2, so Mat2 doesn't apply here).
[[nodiscard]] inline DView2 invert_dense(const DView2& A) {
    const int n = static_cast<int>(A.extent(0));
    if (static_cast<int>(A.extent(1)) != n) {
        throw std::invalid_argument("invert_dense: A must be square");
    }
    DView2 A_copy("invert_dense_A", n, n);
    Kokkos::deep_copy(A_copy, A);
    DView1 workspace("invert_dense_ws", n * n);
    KokkosBatched::SerialInverseLU<KokkosBlas::Algo::LU::Unblocked>::invoke(A_copy, workspace);
    return A_copy;
}

}  // namespace dgfem
