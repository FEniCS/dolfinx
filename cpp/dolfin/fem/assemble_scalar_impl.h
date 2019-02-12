// Copyright (C) 2019 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <dolfin/common/types.h>
#include <memory>
#include <petscsys.h>
#include <vector>

namespace dolfin
{

namespace function
{
class Function;
}

namespace mesh
{
class Mesh;
}

namespace fem
{
class DirichletBC;
class Form;
class GenericDofMap;

namespace impl
{

/// Assemble functional into an scalar
PetscScalar assemble_scalar(const fem::Form& M);

/// Assemble functional over cells
PetscScalar
assemble_cells(const mesh::Mesh& mesh,
               const std::function<void(PetscScalar*, const PetscScalar*,
                                        const double*, int)>& fn,
               std::vector<const function::Function*> coefficients,
               const std::vector<int>& offsets);

/// Execute kernel over exterior facets and accumulate result
PetscScalar assemble_exterior_facets(
    const mesh::Mesh& mesh,
    const std::function<void(PetscScalar*, const PetscScalar*, const double*,
                             int, int)>& fn,
    std::vector<const function::Function*> coefficients,
    const std::vector<int>& offsets);

/// Assemble functional over interior facets
PetscScalar assemble_interior_facets(const Form& M);

} // namespace impl
} // namespace fem
} // namespace dolfin
