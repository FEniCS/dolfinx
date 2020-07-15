// Copyright (C) 2004-2018 Johan Hoffman, Johan Jansson, Anders Logg and Garth
// N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "utils.h"
#include <Eigen/Dense>
#include <array>
#include <cstdint>
#include <petscvec.h>

namespace dolfinx
{
namespace common
{
class IndexMap;
}
namespace la
{

/// Create a PETSc Vec that wraps the data in x
/// @param[in] map The index map that described the parallel layout of
///   the distributed vector
/// @param[in] x The local part of the vector, including ghost entries
/// @return A PETSc Vec object that share the x data. The caller is
///   responsible for destroying the Vec.
Vec create_ghosted_vector(
    const common::IndexMap& map,
    const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>& x);

/// Print error message for PETSc calls that return an error
void petsc_error(int error_code, std::string filename,
                 std::string petsc_function);

/// @todo This function could take just the local sizes
///
/// Compute IndexSets (IS) for stacked index maps.  E.g., if map[0] =
/// {0, 1, 2, 3, 4, 5, 6} and map[1] = {0, 1, 2, 4} (in local indices),
/// IS[0] = {0, 1, 2, 3, 4, 5, 6} and IS[1] = {7, 8, 9, 10}. Caller is
/// responsible for destruction of each IS.
///
/// @param[in] maps Vector of IndexMaps
/// @returns Vector of PETSc Index Sets, created on PETSc_COMM_SELF
std::vector<IS>
create_petsc_index_sets(const std::vector<const common::IndexMap*>& maps);

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

/// Copy blocks from Vec into Eigen vectors
std::vector<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>>
get_local_vectors(const Vec x,
                  const std::vector<const common::IndexMap*>& maps);

/// Scatter local Eigen vectors to Vec
void scatter_local_vectors(
    Vec x,
    const std::vector<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>>& x_b,
    const std::vector<const common::IndexMap*>& maps);

/// It is a simple wrapper for a PETSc vector pointer (Vec). Its main
/// purpose is to assist memory management of PETSc Vec objects.
///
/// For advanced usage, access the PETSc Vec pointer using the function
/// vec() and use the standard PETSc interface.

class PETScVector
{
public:
  /// Create vector
  PETScVector(const common::IndexMap& map);

  /// Create vector
  PETScVector(
      MPI_Comm comm, std::array<std::int64_t, 2> range,
      const Eigen::Array<std::int64_t, Eigen::Dynamic, 1>& ghost_indices,
      int block_size);

  // Delete copy constructor to avoid accidental copying of 'heavy' data
  PETScVector(const PETScVector& x) = delete;

  /// Move constructor
  PETScVector(PETScVector&& x) noexcept;

  /// Create holder of a PETSc Vec object/pointer. The Vec x object
  /// should already be created. If inc_ref_count is true, the reference
  /// counter of the Vec object will be increased. The Vec reference
  /// count will always be decreased upon destruction of the the
  /// PETScVector.
  PETScVector(Vec x, bool inc_ref_count);

  /// Destructor
  virtual ~PETScVector();

  // Assignment operator (disabled)
  PETScVector& operator=(const PETScVector& x) = delete;

  /// Move Assignment operator
  PETScVector& operator=(PETScVector&& x);

  /// Copy vector
  PETScVector copy() const;

  /// Return global size of vector
  std::int64_t size() const;

  /// Return local size of vector (belonging to this process)
  std::int32_t local_size() const;

  /// Return ownership range for process
  std::array<std::int64_t, 2> local_range() const;

  /// Update owned entries owned by this process and which are ghosts on
  /// other processes, i.e., have been added to by a remote process.
  /// This is more efficient that apply() when processes only add/set
  /// their owned entries and the pre-defined ghosts.
  void apply_ghosts();

  /// Update ghost values (gathers ghost values from the owning
  /// processes)
  void update_ghosts();

  /// Return MPI communicator
  MPI_Comm mpi_comm() const;

  /// Return norm of vector
  PetscReal norm(la::Norm norm_type) const;

  /// Sets the prefix used by PETSc when searching the options database
  void set_options_prefix(std::string options_prefix);

  /// Returns the prefix used by PETSc when searching the options
  /// database
  std::string get_options_prefix() const;

  /// Call PETSc function VecSetFromOptions on the underlying Vec object
  void set_from_options();

  /// Return pointer to PETSc Vec object
  Vec vec() const;

private:
  // PETSc Vec pointer
  Vec _x;
};
} // namespace la
} // namespace dolfinx
