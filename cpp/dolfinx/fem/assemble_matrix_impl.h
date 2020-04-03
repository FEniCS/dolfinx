// Copyright (C) 2018-2019 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <functional>
#include <petscmat.h>
#include <petscsys.h>
#include <vector>

namespace dolfinx
{

namespace function
{
class Function;
}

namespace graph
{
template <typename T>
class AdjacencyList;
}

namespace mesh
{
class Mesh;
}

namespace fem
{
class Form;
class DofMap;

namespace impl
{

/// The matrix A must already be initialised. The matrix may be a proxy,
/// i.e. a view into a larger matrix, and assembly is performed using
/// local indices. Rows (bc0) and columns (bc1) with Dirichlet
/// conditions are zeroed. Markers (bc0 and bc1) can be empty if not bcs
/// are applied. Matrix is not finalised.

template <typename ScalarType>
void assemble_matrix(
    const std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                            const std::int32_t*, const ScalarType*)>&
        mat_set_values_local,
    const Form& a, const std::vector<bool>& bc0, const std::vector<bool>& bc1);

/// Execute kernel over cells and accumulate result in Mat
template <typename ScalarType>
void assemble_cells(
    const std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                            const std::int32_t*, const ScalarType*)>&
        mat_set_values_local,
    const mesh::Mesh& mesh, const std::vector<std::int32_t>& active_cells,
    const graph::AdjacencyList<std::int32_t>& dofmap0, int num_dofs_per_cell0,
    const graph::AdjacencyList<std::int32_t>& dofmap1, int num_dofs_per_cell1,
    const std::vector<bool>& bc0, const std::vector<bool>& bc1,
    const std::function<void(ScalarType*, const ScalarType*, const ScalarType*,
                             const double*, const int*, const std::uint8_t*,
                             const std::uint32_t)>& kernel,
    const Eigen::Array<ScalarType, Eigen::Dynamic, Eigen::Dynamic,
                       Eigen::RowMajor>& coeffs,
    const Eigen::Array<ScalarType, Eigen::Dynamic, 1>& constant_values);

/// Execute kernel over exterior facets and  accumulate result in Mat
template <typename ScalarType>
void assemble_exterior_facets(
    const std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                            const std::int32_t*, const ScalarType*)>&
        mat_set_values_local,
    const mesh::Mesh& mesh, const std::vector<std::int32_t>& active_facets,
    const DofMap& dofmap0, const DofMap& dofmap1, const std::vector<bool>& bc0,
    const std::vector<bool>& bc1,
    const std::function<void(ScalarType*, const ScalarType*, const ScalarType*,
                             const double*, const int*, const std::uint8_t*,
                             const std::uint32_t)>& fn,
    const Eigen::Array<ScalarType, Eigen::Dynamic, Eigen::Dynamic,
                       Eigen::RowMajor>& coeffs,
    const Eigen::Array<ScalarType, Eigen::Dynamic, 1> constant_values);

/// Execute kernel over interior facets and  accumulate result in Mat
template <typename ScalarType>
void assemble_interior_facets(
    const std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                            const std::int32_t*, const ScalarType*)>&
        mat_set_values_local,
    const mesh::Mesh& mesh, const std::vector<std::int32_t>& active_facets,
    const DofMap& dofmap0, const DofMap& dofmap1, const std::vector<bool>& bc0,
    const std::vector<bool>& bc1,
    const std::function<void(ScalarType*, const ScalarType*, const ScalarType*,
                             const double*, const int*, const std::uint8_t*,
                             const std::uint32_t)>& kernel,
    const Eigen::Array<ScalarType, Eigen::Dynamic, Eigen::Dynamic,
                       Eigen::RowMajor>& coeffs,
    const std::vector<int>& offsets,
    const Eigen::Array<ScalarType, Eigen::Dynamic, 1>& constant_values);

} // namespace impl
} // namespace fem
} // namespace dolfinx
