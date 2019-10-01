// Copyright (C) 2019 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfin/common/types.h>
#include <Eigen/Dense>
#include <dolfin/common/types.h>
#include <memory>
#include <petscsys.h>
#include "ufc.h"
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
class DofMap;

namespace impl
{

/// Assemble functional into an scalar
PetscScalar assemble_scalar(const fem::Form& M);

/// Assemble functional over cells
PetscScalar
assemble_cells(const mesh::Mesh& mesh,
               const std::vector<std::int32_t>& active_cells,
               const std::function<ufc_tabulate_tensor>& fn,
               const std::vector<const function::Function*>& coefficients,
               const std::vector<int>& offsets,
               const std::vector<PetscScalar> constant_values);

/// Execute kernel over exterior facets and accumulate result
PetscScalar assemble_exterior_facets(
    const mesh::Mesh& mesh, const std::vector<std::int32_t>& active_cells,
    const std::function<ufc_tabulate_tensor>& fn,
    const std::vector<const function::Function*>& coefficients,
    const std::vector<int>& offsets,
    const std::vector<PetscScalar> constant_values);

/// Assemble functional over interior facets
PetscScalar assemble_interior_facets(
    const mesh::Mesh& mesh, const std::vector<std::int32_t>& active_cells,
    const std::function<ufc_tabulate_tensor>& fn,
    const std::vector<const function::Function*>& coefficients,
    const std::vector<int>& offsets,
    const std::vector<PetscScalar> constant_values);

} // namespace impl
} // namespace fem
} // namespace dolfin
