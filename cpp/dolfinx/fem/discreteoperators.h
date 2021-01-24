// Copyright (C) 2015 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <petscmat.h>

namespace dolfinx::fem
{
class FunctionSpace;

/// @todo Improve documentation
/// This function class computes discrete gradient operators (matrices)
/// that map derivatives of finite element functions into other finite
/// element spaces. An example of where discrete gradient operators are
/// required is the creation of algebraic multigrid solvers for H(curl)
/// and H(div) problems.
///
/// @warning This function is highly experimental and likely to change
/// or be replaced or be removed
///
/// Build the discrete gradient operator A that takes a
/// \f$w \in H^1\f$ (P1, nodal Lagrange) to \f$v \in H(curl)\f$
/// (lowest order Nedelec), i.e. v = Aw. V0 is the H(curl) space,
/// and V1 is the P1 Lagrange space.
///
/// @param[in] V0 A H(curl) space
/// @param[in] V1 A P1 Lagrange space
/// @return The discrete operator matrix. The caller is responsible for
/// destroying the  Mat.
Mat create_discrete_gradient(const fem::FunctionSpace& V0,
                             const fem::FunctionSpace& V1);
} // namespace dolfinx::fem
