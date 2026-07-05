/**
 * @file trilinos_types.hpp
 * @brief Shared Tpetra type aliases and Kokkos-View<->Tpetra conversion helpers for the
 * global (distributed) sparse linear algebra layer. Local per-element dense math uses the
 * Kokkos types in kokkos_math.hpp -- these types are only for assembled system matrices and
 * whole-mesh (global) vectors.
 *
 * No MPI: every DOF lives on a single serial map, since this solver runs single-machine.
 */
#pragma once

#include "dgfem/kokkos_math.hpp"

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

/// Copy a (whole-mesh, global) DView1 into a freshly-allocated single-column
/// Tpetra::MultiVector.
[[nodiscard]] Teuchos::RCP<TpetraMultiVector>
view_to_tpetra(const DView1& v, const Teuchos::RCP<const TpetraMap>& map);

/// Copy a single-column Tpetra::MultiVector into a (whole-mesh, global) DView1.
[[nodiscard]] DView1 tpetra_to_view(const TpetraMultiVector& v);

/// Densify a (small!) Tpetra::CrsMatrix into a DView2, for tests/debugging.
[[nodiscard]] DView2 tpetra_to_dense(const TpetraCrsMatrix& m);

}  // namespace dgfem
