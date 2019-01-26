// Copyright (C) 2018-2019 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <array>
#include <dolfin/common/MPI.h>
#include <petscis.h>
#include <petscmat.h>
#include <petscvec.h>
#include <string>
#include <vector>

namespace dolfin
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
/// Vec object.
Vec create_vector(const common::IndexMap& map);

/// Create a ghosted PETSc Vec. Caller is responsible for destroying the
/// Vec object.
Vec create_vector(
    MPI_Comm comm, std::array<std::int64_t, 2> range,
    const Eigen::Array<PetscInt, Eigen::Dynamic, 1>& ghost_indices,
    int block_size);

/// Create a PETSc Mat. Caller is responsible for destroying the object.
Mat create_matrix(MPI_Comm comm, const SparsityPattern& sparsity_pattern);

/// Create PETSc MatNullSpace. Caller is responsible for destruction.
MatNullSpace create_petsc_nullspace(MPI_Comm comm,
                                    const VectorSpaceBasis& nullspace);

/// Compute IndexSets (IS) for stacked index maps
std::vector<IS> compute_index_sets(std::vector<const common::IndexMap*> maps);

/// Print error message for PETSc calls that return an error
void petsc_error(int error_code, std::string filename,
                 std::string petsc_function);

class VecWrapper
{
public:
  VecWrapper(Vec y, bool ghosted = true);
  VecWrapper(const VecWrapper& x) = delete;
  VecWrapper(VecWrapper&& x) = default;
  VecWrapper& operator=(const VecWrapper& x) = delete;
  VecWrapper& operator=(VecWrapper&& x) = default;
  ~VecWrapper() = default;
  void restore();
  Eigen::Map<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> x;

private:
  PetscScalar* array = nullptr;
  Vec _y;
  Vec _y_local = nullptr;
};

class VecReadWrapper
{
public:
  VecReadWrapper(const Vec y, bool ghosted = true);
  VecReadWrapper(const VecReadWrapper& x) = delete;
  VecReadWrapper(VecReadWrapper&& x) = default;
  VecReadWrapper& operator=(const VecReadWrapper& x) = delete;
  VecReadWrapper& operator=(VecReadWrapper&& x) = default;
  ~VecReadWrapper() = default;
  void restore();
  Eigen::Map<const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> x;

private:
  PetscScalar const* array = nullptr;
  const Vec _y;
  Vec _y_local = nullptr;
};

} // namespace la
} // namespace dolfin
