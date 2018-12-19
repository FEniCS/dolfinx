// Copyright (C) 2018 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <petscmat.h>
#include <petscsys.h>
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

// FIXME: Add comment on zero Dirichlet rows/cols
/// The matrix A must already be initialised. The matrix may be a proxy,
/// i.e. a view into a larger matrix, and assembly is performed using
/// local indices. Matrix is not finalised.
void assemble_matrix(la::PETScMatrix& A, const Form& a,
                     const std::vector<bool>& bc0,
                     const std::vector<bool>& bc1);

// FIXME: Add comment on zero Dirichlet rows/cols
/// The matrix A must already be initialised. The matrix may be a proxy,
/// i.e. a view into a larger matrix, and assembly is performed using
/// local indices. Matrix is not finalised.
void assemble_matrix(Mat A, const Form& a, const std::vector<bool>& bc0,
                     const std::vector<bool>& bc1);

} // namespace fem
} // namespace dolfin
