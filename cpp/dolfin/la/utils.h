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
/// returned object.
Vec create_petsc_vector(const common::IndexMap& map);

/// Create a ghosted PETSc Vec. Caller is responsible for destroying the
/// returned object.
Vec create_petsc_vector(
    MPI_Comm comm, std::array<std::int64_t, 2> range,
    const Eigen::Array<PetscInt, Eigen::Dynamic, 1>& ghost_indices,
    int block_size);

/// Create a PETSc Mat. Caller is responsible for destroying the
/// returned object.
Mat create_petsc_matrix(MPI_Comm comm, const SparsityPattern& sparsity_pattern);

/// Create PETSc MatNullSpace. Caller is responsible for destruction
/// returned object.
MatNullSpace create_petsc_nullspace(MPI_Comm comm,
                                    const VectorSpaceBasis& nullspace);

/// Compute IndexSets (IS) for stacked index maps. Caller is responsible
/// for destruction of each IS.
std::vector<IS>
compute_petsc_index_sets(std::vector<const dolfin::common::IndexMap*> maps);

/// Print error message for PETSc calls that return an error
void petsc_error(int error_code, std::string filename,
                 std::string petsc_function);

/// Wrapper around a PETSc Vec object, to simplify direct access to data.
class VecWrapper
{
public:
  /// Wrap PETSc Vec y
  VecWrapper(Vec y, bool ghosted = true);
  VecWrapper(const VecWrapper& w) = delete;
  /// Move constructor
  VecWrapper(VecWrapper&& w);
  VecWrapper& operator=(const VecWrapper& w) = delete;
  /// Move assignment
  VecWrapper& operator=(VecWrapper&& w);
  ~VecWrapper();
  /// Restore PETSc Vec object
  void restore();
  /// Eigen Map into PETSc Vec
  Eigen::Map<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> x;

private:
  PetscScalar* array = nullptr;
  Vec _y;
  Vec _y_local = nullptr;
  bool _ghosted;
};

/// Read-only wrapper around a PETSc Vec object, to simplify direct access to
/// data.
class VecReadWrapper
{
public:
  /// Wrap PETSc Vec y
  VecReadWrapper(const Vec y, bool ghosted = true);
  VecReadWrapper(const VecReadWrapper& w) = delete;
  /// Move constructor
  VecReadWrapper(VecReadWrapper&& w);
  VecReadWrapper& operator=(const VecReadWrapper& w) = delete;
  /// Move assignment
  VecReadWrapper& operator=(VecReadWrapper&& w);
  ~VecReadWrapper();
  /// Restore PETSc Vec
  void restore();
  /// Eigen Map into PETSc Vec
  Eigen::Map<const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> x;

private:
  PetscScalar const* array = nullptr;
  Vec _y;
  Vec _y_local = nullptr;
  bool _ghosted;
};

// /// Get sub-matrix. Sub-matrix is local to the process
// Mat get_local_submatrix(const Mat A, const IS row, const IS col);

// /// Restore local submatrix
// void restore_local_submatrix(const Mat A, const IS row, const IS col, Mat* Asub);


/// Copy blocks from Vec into Eigen vectors
std::vector<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>>
get_local_vectors(const Vec x,
                  const std::vector<const common::IndexMap*>& maps);

/// Copy blocks from Vec into local Vec objects
// std::vector<Vec>
// get_local_petsc_vectors(const Vec x,
//                         const std::vector<const common::IndexMap*>& maps);

/// Scatter local Eigen vectors to Vec
void scatter_local_vectors(
    Vec x,
    const std::vector<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>>& x_b,
    const std::vector<const common::IndexMap*>& maps);

} // namespace la
} // namespace dolfin
