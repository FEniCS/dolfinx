// Copyright (C) 2018-2019 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <petscmat.h>
#include <vector>

namespace dolfin
{

namespace fem
{
class Form;

namespace impl
{

/// The matrix A must already be initialised. The matrix may be a proxy,
/// i.e. a view into a larger matrix, and assembly is performed using
/// local indices. Rows (bc0) and columns (bc1) with Dirichlet
/// conditions are zeroed. Markers (bc0 and bc1) can be empty if not bcs
/// are applied. Matrix is not finalised.
void assemble_matrix(Mat A, const Form& a, const std::vector<bool>& bc0,
                     const std::vector<bool>& bc1);

} // namespace impl
} // namespace fem
} // namespace dolfin