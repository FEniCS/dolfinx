// Copyright (C) 2019 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <dolfinx/common/types.h>
#include <memory>
#include <petscsys.h>
#include <vector>

namespace dolfinx
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
PetscScalar assemble_cells(
    const mesh::Mesh& mesh, const std::vector<std::int32_t>& active_cells,
    const std::function<void(PetscScalar*, const PetscScalar*,
                             const PetscScalar*, const double*, const int*,
                             const std::uint8_t*, const bool*, const bool*,
                             const std::uint8_t*)>& fn,
    const Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic,
                       Eigen::RowMajor>& coeffs,
    const std::vector<PetscScalar>& constant_values);

/// Execute kernel over exterior facets and accumulate result
PetscScalar assemble_exterior_facets(
    const mesh::Mesh& mesh, const std::vector<std::int32_t>& active_cells,
    const std::function<void(PetscScalar*, const PetscScalar*,
                             const PetscScalar*, const double*, const int*,
                             const std::uint8_t*, const bool*, const bool*,
                             const std::uint8_t*)>& fn,
    const Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic,
                       Eigen::RowMajor>& coeffs,
    const std::vector<PetscScalar>& constant_values);

/// Assemble functional over interior facets
PetscScalar assemble_interior_facets(
    const mesh::Mesh& mesh, const std::vector<std::int32_t>& active_cells,
    const std::function<void(PetscScalar*, const PetscScalar*,
                             const PetscScalar*, const double*, const int*,
                             const std::uint8_t*, const bool*, const bool*,
                             const std::uint8_t*)>& fn,
    const Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic,
                       Eigen::RowMajor>& coeffs,
    const std::vector<int>& offsets,
    const std::vector<PetscScalar>& constant_values);

} // namespace impl
} // namespace fem
} // namespace dolfinx
