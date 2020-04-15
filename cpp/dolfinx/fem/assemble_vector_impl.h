// Copyright (C) 2018-2019 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <dolfinx/common/types.h>
#include <functional>
#include <memory>
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
class DirichletBC;
class Form;
class DofMap;

/// Implementation of assembly
namespace impl
{

/// Assemble linear form into an Eigen vector
/// @param[in,out] b The vector to be assembled. It will not be zeroed
///                  before assembly.
/// @param[in] L The linear forms to assemble into b
void assemble_vector(
    Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> b, const Form& L);

/// Execute kernel over cells and accumulate result in vector
void assemble_cells(
    Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> b,
    const mesh::Mesh& mesh, const std::vector<std::int32_t>& active_cells,
    const graph::AdjacencyList<std::int32_t>& dofmap, int num_dofs_per_cell,
    const std::function<void(PetscScalar*, const PetscScalar*,
                             const PetscScalar*, const double*, const int*,
                             const std::uint8_t*, const std::uint32_t)>& kernel,
    const Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic,
                       Eigen::RowMajor>& coeffs,
    const Eigen::Array<PetscScalar, Eigen::Dynamic, 1>& constant_values);

/// Execute kernel over cells and accumulate result in vector
void assemble_exterior_facets(
    Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> b,
    const mesh::Mesh& mesh, const std::vector<std::int32_t>& active_facets,
    const fem::DofMap& dofmap,
    const std::function<void(PetscScalar*, const PetscScalar*,
                             const PetscScalar*, const double*, const int*,
                             const std::uint8_t*, const std::uint32_t)>& fn,
    const Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic,
                       Eigen::RowMajor>& coeffs,
    const Eigen::Array<PetscScalar, Eigen::Dynamic, 1>& constant_values);

/// Assemble linear form interior facet integrals into an Eigen vector
void assemble_interior_facets(
    Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> b,
    const mesh::Mesh& mesh, const std::vector<std::int32_t>& active_facets,
    const fem::DofMap& dofmap,
    const std::function<void(PetscScalar*, const PetscScalar*,
                             const PetscScalar*, const double*, const int*,
                             const std::uint8_t*, const std::uint32_t)>& fn,
    const Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic,
                       Eigen::RowMajor>& coeffs,
    const std::vector<int>& offsets,
    const Eigen::Array<PetscScalar, Eigen::Dynamic, 1>& constant_values);

/// Modify b such that:
///
///   b <- b - scale * A_j (g_j - x0_j)
///
/// where j is a block (nest) row index. For a non-blocked problem j = 0.
/// The boundary conditions bc1 are on the trial spaces V_j. The forms
/// in [a] must have the same test space as L (from which b was built),
/// but the trial space may differ. If x0 is not supplied, then it is
/// treated as zero.
/// @param[in,out] b The vector to be modified
/// @param[in] a The bilinear forms, where a[j] is the form that
///              generates A_j
/// @param[in] bcs1 List of boundary conditions for each block, i.e.
///                 bcs1[2] are the boundary conditions applied to the
///                 columns of a[2] / x0[2] block
/// @param[in] x0 The vectors used in the lifting
/// @param[in] scale Scaling to apply
void apply_lifting(
    Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> b,
    const std::vector<std::shared_ptr<const Form>> a,
    const std::vector<std::vector<std::shared_ptr<const DirichletBC>>>& bcs1,
    const std::vector<
        Eigen::Ref<const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>>>& x0,
    double scale);

/// Modify RHS vector to account for boundary condition
///
///    b <- b - scale * A x_bc
////
/// @param[in,out] b The vector to be modified
/// @param[in] a The bilinear form that generates A
/// @param[in] bc_values1 The boundary condition 'values'
/// @param[in] bc_markers1 The indices (columns of A, rows of x) to
///                        which bcs belong
/// @param[in] scale Scaling to apply
void lift_bc(
    Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> b, const Form& a,
    const Eigen::Ref<const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>>&
        bc_values1,
    const std::vector<bool>& bc_markers1, double scale);

/// Modify RHS vector to account for boundary condition such that: b <-
////
///     b - scale * A (x_bc - x0)
////
/// @param[in,out] b The vector to be modified
/// @param[in] a The bilinear form that generates A
/// @param[in] bc_values1 The boundary condition 'values'
/// @param[in] bc_markers1 The indices (columns of A, rows of x) to
///                        which bcs belong
/// @param[in] x0 The array used in the lifting, typically a 'current
///               solution' in a Newton method
/// @param[in] scale Scaling to apply
void lift_bc(
    Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> b, const Form& a,
    const Eigen::Ref<const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>>&
        bc_values1,
    const std::vector<bool>& bc_markers1,
    const Eigen::Ref<const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>>& x0,
    double scale);

} // namespace impl
} // namespace fem
} // namespace dolfinx
