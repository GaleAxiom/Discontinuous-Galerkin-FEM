#include "dgfem/solver/trilinos_types.hpp"

#include <Tpetra_Core.hpp>

namespace dgfem {

Teuchos::RCP<const TpetraMap> make_serial_map(TpetraGlobalOrdinal n_dofs) {
    return Teuchos::rcp(new TpetraMap(n_dofs, 0, Tpetra::getDefaultComm()));
}

Teuchos::RCP<TpetraMultiVector> view_to_tpetra(const DView1& v,
                                               const Teuchos::RCP<const TpetraMap>& map) {
    auto result = Teuchos::rcp(new TpetraMultiVector(map, 1));
    auto view = result->getLocalViewHost(Tpetra::Access::OverwriteAll);
    for (TpetraLocalOrdinal i = 0; i < static_cast<TpetraLocalOrdinal>(map->getLocalNumElements());
         ++i) {
        view(i, 0) = v[map->getGlobalElement(i)];
    }
    return result;
}

DView2 tpetra_to_dense(const TpetraCrsMatrix& m) {
    DView2 dense("dense", m.getGlobalNumRows(), m.getGlobalNumCols());
    auto row_map = m.getRowMap();
    for (TpetraLocalOrdinal local_row = 0;
         local_row < static_cast<TpetraLocalOrdinal>(row_map->getLocalNumElements()); ++local_row) {
        TpetraGlobalOrdinal global_row = row_map->getGlobalElement(local_row);
        size_t n_entries = m.getNumEntriesInGlobalRow(global_row);
        typename TpetraCrsMatrix::nonconst_global_inds_host_view_type indices("indices", n_entries);
        typename TpetraCrsMatrix::nonconst_values_host_view_type values("values", n_entries);
        size_t n_copied = 0;
        m.getGlobalRowCopy(global_row, indices, values, n_copied);
        for (size_t k = 0; k < n_copied; ++k) {
            dense(global_row, indices(k)) = values(k);
        }
    }
    return dense;
}

DView1 tpetra_to_view(const TpetraMultiVector& v) {
    auto map = v.getMap();
    DView1 result("tpetra_to_view", v.getGlobalLength());
    auto view = v.getLocalViewHost(Tpetra::Access::ReadOnly);
    for (TpetraLocalOrdinal i = 0; i < static_cast<TpetraLocalOrdinal>(map->getLocalNumElements());
         ++i) {
        result[map->getGlobalElement(i)] = view(i, 0);
    }
    return result;
}

}  // namespace dgfem
