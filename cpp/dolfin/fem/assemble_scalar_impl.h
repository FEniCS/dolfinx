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
PetscScalar assemble(const fem::Form& M);

/// Assemble functional over cells
PetscScalar
assemble_cells(const fem::Form& M, const mesh::Mesh& mesh,
               const std::function<void(PetscScalar*, const PetscScalar*,
                                        const double*, int)>& fn);

/// Assemble functional over exterior facets
PetscScalar assemble_exterior_facets(
    const fem::Form& M, const mesh::Mesh& mesh,
    const std::function<void(PetscScalar*, const PetscScalar*, const double*,
                             int, int)>& fn);

/// Assemble functional over interior facets
PetscScalar assemble_interior_facets(const Form& M);

} // namespace impl
} // namespace fem
} // namespace dolfin