// Milestone 1 smoke test: proves the Trilinos toolchain (Kokkos init, Tpetra assembly,
// Amesos2 direct solve) builds, links, and runs correctly before any real library code
// is touched. Not wired into the dgfem library or test suite -- standalone by design.
#include <Amesos2.hpp>
#include <Kokkos_Core.hpp>
#include <Tpetra_Core.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_MultiVector.hpp>

#include <iostream>

using Scalar = double;
using LO = int;
using GO = long long;
using Node = Tpetra::KokkosClassic::DefaultNode::DefaultNodeType;

using Map = Tpetra::Map<LO, GO, Node>;
using CrsMatrix = Tpetra::CrsMatrix<Scalar, LO, GO, Node>;
using Vector = Tpetra::MultiVector<Scalar, LO, GO, Node>;

int main(int argc, char* argv[]) {
    Kokkos::ScopeGuard kokkos_guard(argc, argv);

    // 3x3 system: diag(2,3,4) * x = [2,6,12] => x = [1,2,3]
    const GO n_global = 3;
    Teuchos::RCP<const Map> map = Teuchos::rcp(new Map(n_global, 0, Tpetra::getDefaultComm()));

    Teuchos::RCP<CrsMatrix> A = Teuchos::rcp(new CrsMatrix(map, 1));
    for (LO i = 0; i < static_cast<LO>(map->getLocalNumElements()); ++i) {
        GO gid = map->getGlobalElement(i);
        double diag_value = 2.0 + static_cast<double>(gid);
        A->insertGlobalValues(gid, Teuchos::tuple(gid), Teuchos::tuple(diag_value));
    }
    A->fillComplete();

    Teuchos::RCP<Vector> b = Teuchos::rcp(new Vector(map, 1));
    {
        auto b_view = b->getLocalViewHost(Tpetra::Access::OverwriteAll);
        for (LO i = 0; i < static_cast<LO>(map->getLocalNumElements()); ++i) {
            GO gid = map->getGlobalElement(i);
            b_view(i, 0) = (2.0 + static_cast<double>(gid)) * static_cast<double>(gid + 1);
        }
    }

    Teuchos::RCP<Vector> x = Teuchos::rcp(new Vector(map, 1));
    x->putScalar(0.0);

    auto solver = Amesos2::create<CrsMatrix, Vector>("klu2", A, x, b);
    solver->symbolicFactorization().numericFactorization().solve();

    auto x_view = x->getLocalViewHost(Tpetra::Access::ReadOnly);
    std::cout << "Trilinos smoke test solution (expect 1, 2, 3):" << std::endl;
    bool ok = true;
    for (LO i = 0; i < static_cast<LO>(map->getLocalNumElements()); ++i) {
        double expected = static_cast<double>(map->getGlobalElement(i) + 1);
        double got = x_view(i, 0);
        std::cout << "  x[" << map->getGlobalElement(i) << "] = " << got << " (expected "
                  << expected << ")" << std::endl;
        if (std::abs(got - expected) > 1e-10) {
            ok = false;
        }
    }

    if (!ok) {
        std::cerr << "Trilinos smoke test FAILED" << std::endl;
        return 1;
    }
    std::cout << "Trilinos smoke test PASSED" << std::endl;
    return 0;
}
