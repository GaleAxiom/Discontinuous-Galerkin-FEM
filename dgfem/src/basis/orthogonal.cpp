/**
 * @file orthogonal.cpp
 * @brief Implementation of base orthogonal basis class with modern C++ features
 */

#include "dgfem/basis/orthogonal.hpp"

#include <sstream>
#include <stdexcept>

namespace dgfem {

OrthogonalBasis::OrthogonalBasis(std::shared_ptr<ReferenceElement> ref_element, int order)
    : ref_element_(std::move(ref_element)), order_(order), n_basis_(0) {
    if (!is_order_valid(order)) {
        std::ostringstream oss;
        oss << "Order " << order << " is invalid. Must be between 1 and 10";
        throw std::invalid_argument(oss.str());
    }

    if (!ref_element_) {
        throw std::invalid_argument("Reference element cannot be null");
    }

    // Note: n_basis_ will be set by derived classes after construction
}

}  // namespace dgfem