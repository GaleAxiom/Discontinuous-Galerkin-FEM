/**
 * @file basis_mode_map.hpp
 * @brief Injectable lookup for "which modal coefficient index is linear in direction d, and
 * which other modes are cross terms that depend on it."
 *
 * Encapsulates basis-specific knowledge that used to be hardcoded as literal mode indices
 * (mode 1 = x-linear, mode 3 = xy cross term) directly in MinmodReconstruction/
 * WenoReconstruction/PerssonPeraireIndicator. A future 3D tensor-product/hex basis has an
 * entirely different set of modes and cross terms; injecting this lookup means those classes
 * never need to change when that basis exists, only a new concrete BasisModeMap is added.
 */

#pragma once

#include <string>
#include <vector>

namespace dgfem {

// Purely an axis label for basis-modal questions -- deliberately not FaceDirection
// (neighbor_connectivity.hpp), which also encodes minus/plus sidedness that has no meaning
// for "which mode is linear in x."
enum class Direction { X, Y, Z };

class BasisModeMap {
public:
    virtual ~BasisModeMap() = default;
    [[nodiscard]] virtual std::string get_type() const = 0;

    // -1 if this basis has no mode isolated to that direction at its current order (shouldn't
    // happen for any basis in the current roadmap, but keeps the contract honest rather than
    // asserting).
    [[nodiscard]] virtual int linear_mode_index(Direction direction) const = 0;

    // Mode indices that must be zeroed alongside the linear mode once it's deemed
    // untrustworthy (the standard moment-limiter cascade: once a lower-order trend is
    // discarded, higher-variation modes that multiply it are discarded too).
    [[nodiscard]] virtual std::vector<int>
    dependent_cross_term_indices(Direction direction) const = 0;
};

// Order-1 tensor Legendre quad basis: modes [0]=const, [1]=x-linear, [2]=y-linear, [3]=xy.
class Order1QuadBasisModeMap : public BasisModeMap {
public:
    [[nodiscard]] std::string get_type() const override { return "Order1Quad"; }

    [[nodiscard]] int linear_mode_index(Direction direction) const override {
        switch (direction) {
        case Direction::X:
            return 1;
        case Direction::Y:
            return 2;
        default:
            return -1;
        }
    }

    [[nodiscard]] std::vector<int>
    dependent_cross_term_indices(Direction direction) const override {
        switch (direction) {
        case Direction::X:
        case Direction::Y:
            return {3};
        default:
            return {};
        }
    }
};

}  // namespace dgfem
