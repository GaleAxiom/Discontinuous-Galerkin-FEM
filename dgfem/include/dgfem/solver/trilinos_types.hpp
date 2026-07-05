/**
 * @file trilinos_types.hpp
 * @brief Shared Tpetra type aliases and Eigen<->Tpetra conversion helpers for the global
 * (distributed) sparse linear algebra layer. Local per-element dense math stays Eigen/Kokkos
 * (see the migration plan) -- these types are only for assembled system matrices/vectors.
 *
 * No MPI: every DOF lives on a single serial map, since this solver runs single-machine.
 */
#pragma once

#include <Eigen/Dense>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_MultiVector.hpp>

namespace dgfem {

using TpetraScalar = double;
using TpetraLocalOrdinal = int;
using TpetraGlobalOrdinal = long long;

using TpetraMap = Tpetra::Map<TpetraLocalOrdinal, TpetraGlobalOrdinal>;
using TpetraCrsMatrix = Tpetra::CrsMatrix<TpetraScalar, TpetraLocalOrdinal, TpetraGlobalOrdinal>;
// Amesos2's adapters aren't specialized for Tpetra::Vector; MultiVector (1 column) is the
// portable choice for anything that may need a direct/iterative solve.
using TpetraMultiVector =
    Tpetra::MultiVector<TpetraScalar, TpetraLocalOrdinal, TpetraGlobalOrdinal>;

/// A single-rank Tpetra map over n_dofs global indices [0, n_dofs).
[[nodiscard]] Teuchos::RCP<const TpetraMap> make_serial_map(TpetraGlobalOrdinal n_dofs);

/// Copy an Eigen vector into a freshly-allocated single-column Tpetra::MultiVector.
[[nodiscard]] Teuchos::RCP<TpetraMultiVector>
eigen_to_tpetra(const Eigen::VectorXd& v, const Teuchos::RCP<const TpetraMap>& map);

/// Copy a single-column Tpetra::MultiVector into an Eigen vector.
[[nodiscard]] Eigen::VectorXd tpetra_to_eigen(const TpetraMultiVector& v);

/// Densify a (small!) Tpetra::CrsMatrix into an Eigen::MatrixXd, for tests/debugging.
[[nodiscard]] Eigen::MatrixXd tpetra_to_dense(const TpetraCrsMatrix& m);

}  // namespace dgfem
