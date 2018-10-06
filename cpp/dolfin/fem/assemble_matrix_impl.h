// Copyright (C) 2018 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <vector>

namespace dolfin
{
namespace la
{
class PETScMatrix;
} // namespace la

namespace fem
{
class Form;

/// Assemble matrix, with Dirichlet rows/columns zeroed. The matrix A
/// must already be initialised. The matrix may be a proxy, i.e. a view
/// into a larger matrix, and assembly is performed using local indices.
/// Matrix is not finalised.
void assemble_matrix(la::PETScMatrix& A, const Form& a,
                     const std::vector<std::int32_t>& bc_dofs0,
                     const std::vector<std::int32_t>& bc_dofs1);

} // namespace fem
} // namespace dolfin
