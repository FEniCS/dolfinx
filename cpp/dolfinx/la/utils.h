// Copyright (C) 2018-2019 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <array>
#include <dolfinx/common/MPI.h>
#include <petscis.h>
#include <petscmat.h>
#include <petscvec.h>
#include <string>
#include <vector>

namespace dolfinx
{
namespace common
{
class IndexMap;
}
namespace fem
{
class Form;
}
namespace la
{
class VectorSpaceBasis;
class SparsityPattern;

/// Norm types
enum class Norm
{
  l1,
  l2,
  linf,
  frobenius
};

/// Create a ghosted PETSc Vec. Caller is responsible for destroying the
/// returned object.
Vec create_petsc_vector(const common::IndexMap& map);

/// Create a ghosted PETSc Vec. Caller is responsible for destroying the
/// returned object.
Vec create_petsc_vector(
    MPI_Comm comm, std::array<std::int64_t, 2> range,
    const Eigen::Ref<const Eigen::Array<std::int64_t, Eigen::Dynamic, 1>>&
        ghost_indices,
    int block_size);

/// Create a PETSc Mat. Caller is responsible for destroying the
/// returned object.
Mat create_petsc_matrix(MPI_Comm comm, const SparsityPattern& sparsity_pattern);

/// Create PETSc MatNullSpace. Caller is responsible for destruction
/// returned object.
MatNullSpace create_petsc_nullspace(MPI_Comm comm,
                                    const VectorSpaceBasis& nullspace);

/// @todo This function could take just the local sizes
///
/// Compute IndexSets (IS) for stacked index maps.  E.g., if map[0] =
/// {0, 1, 2, 3, 4, 5, 6} and map[1] = {0, 1, 2, 4} (in local indices),
/// IS[0] = {0, 1, 2, 3, 4, 5, 6} and IS[1] = {7, 8, 9, 10}. Caller is
/// responsible for destruction of each IS.
/// @param[in] maps Vector of IndexMaps
/// @returns Vector of PETSc Index Sets, created on PETSc_COMM_SELF
std::vector<IS>
create_petsc_index_sets(const std::vector<const common::IndexMap*>& maps);

/// Print error message for PETSc calls that return an error
void petsc_error(int error_code, std::string filename,
                 std::string petsc_function);

/// Copy blocks from Vec into Eigen vectors
std::vector<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>>
get_local_vectors(const Vec x,
                  const std::vector<const common::IndexMap*>& maps);

/// Scatter local Eigen vectors to Vec
void scatter_local_vectors(
    Vec x,
    const std::vector<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>>& x_b,
    const std::vector<const common::IndexMap*>& maps);

} // namespace la
} // namespace dolfinx
