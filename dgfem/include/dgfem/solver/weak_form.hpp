/**
 * @file weak_form.hpp
 * @brief Weak formulation for the DG method (backward compatibility header)
 *
 * This file includes all weak formulation headers for backward compatibility.
 * New code should include the specific weak formulation headers directly from dgfem/weak_forms/
 */

#pragma once

// Include all weak formulation headers from new location
#include "dgfem/weak_forms/advection_weak_formulation.hpp"
#include "dgfem/weak_forms/euler_weak_formulation.hpp"
#include "dgfem/weak_forms/laplace_weak_formulation.hpp"
#include "dgfem/weak_forms/weak_formulation_base.hpp"
