// Copyright (C) 2004-2018 Johan Hoffman, Johan Jansson, Anders Logg and Garth
// N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "utils.h"
#include <array>
#include <cstdint>
#include <functional>
#include <petscvec.h>
#include <vector>
#include <xtl/xspan.hpp>

namespace dolfinx::common
{
class IndexMap;
}
namespace dolfinx::la
{

namespace petsc
{

/// Print error message for PETSc calls that return an error
void petsc_error(int error_code, std::string filename,
                 std::string petsc_function);

/// Create PETsc vectors from the local data. The data is copied into
/// the PETSc vectors and is not shared.
/// @param[in] comm The MPI communicator
/// @param[in] x The vector data owned by the calling rank. All
/// components must have the same length.
/// @return Array of PETSc vectors
std::vector<Vec>
create_vectors(MPI_Comm comm,
               const std::vector<xtl::span<const PetscScalar>>& x);

/// Create a ghosted PETSc Vec
/// @note Caller is responsible for destroying the returned object
/// @param[in] map The index map describing the parallel layout (by block)
/// @param[in] bs The block size
/// @returns A PETSc Vec
Vec create_vector(const common::IndexMap& map, int bs);

/// Create a ghosted PETSc Vec from a local range and ghost indices
/// @note Caller is responsible for destroying the returned object
/// @param[in] comm The MPI communicator
/// @param[in] range The local ownership range (by blocks)
/// @param[in] ghosts Ghost blocks
/// @param[in] bs The block size. The total number of local entries is
/// `bs * (range[1] - range[0])`.
/// @returns A PETSc Vec
Vec create_vector(MPI_Comm comm, std::array<std::int64_t, 2> range,
                  const std::vector<std::int64_t>& ghosts, int bs);

/// Create a PETSc Vec that wraps the data in an array
/// @param[in] map The index map that describes the parallel layout of
/// the distributed vector (by block)
/// @param[in] bs Block size
/// @param[in] x The local part of the vector, including ghost entries
/// @return A PETSc Vec object that shares the data in @p x. The caller
/// is responsible for destroying the Vec.
Vec create_vector_wrap(const common::IndexMap& map, int bs,
                       const xtl::span<PetscScalar>& x);

/// @todo This function could take just the local sizes
///
/// Compute PETSc IndexSets (IS) for a stack of index maps. E.g., if
/// `map[0] = {0, 1, 2, 3, 4, 5, 6}` and `map[1] = {0, 1, 2, 4}` (in
/// local indices) then `IS[0] = {0, 1, 2, 3, 4, 5, 6}` and `IS[1] = {7, 8,
/// 9, 10}`.
///
/// The caller is responsible for destruction of each IS.
///
/// @param[in] maps Vector of IndexMaps and corresponding block sizes
/// @returns Vector of PETSc Index Sets, created on` PETSC_COMM_SELF`
std::vector<IS> create_index_sets(
    const std::vector<
        std::pair<std::reference_wrapper<const common::IndexMap>, int>>& maps);

/// Copy blocks from Vec into local vectors
std::vector<std::vector<PetscScalar>> get_local_vectors(
    const Vec x,
    const std::vector<
        std::pair<std::reference_wrapper<const common::IndexMap>, int>>& maps);

/// Scatter local vectors to Vec
void scatter_local_vectors(
    Vec x, const std::vector<xtl::span<const PetscScalar>>& x_b,
    const std::vector<
        std::pair<std::reference_wrapper<const common::IndexMap>, int>>& maps);

} // namespace petsc

/// A simple wrapper for a PETSc vector pointer (Vec). Its main purpose
/// is to assist with memory/lifetime management of PETSc Vec objects.
///
/// Access the underlying PETSc Vec pointer using the function vec() and
/// use the full PETSc interface.
class PETScVector
{
public:
  /// Create vector
  ///
  /// Collective
  ///
  /// @param[in] map Index map describing the parallel layout
  /// @param[in] bs the block size
  PETScVector(const common::IndexMap& map, int bs);

  // Delete copy constructor to avoid accidental copying of 'heavy' data
  PETScVector(const PETScVector& x) = delete;

  /// Move constructor
  PETScVector(PETScVector&& x);

  /// Create holder of a PETSc Vec object/pointer. The Vec x object
  /// should already be created. If inc_ref_count is true, the reference
  /// counter of the Vec object will be increased. The Vec reference
  /// count will always be decreased upon destruction of the the
  /// PETScVector.
  ///
  /// Collective
  ///
  /// @param[in] x The PETSc Vec
  /// @param[in] inc_ref_count True if the reference count of `x` should
  /// be incremented
  PETScVector(Vec x, bool inc_ref_count);

  /// Destructor
  virtual ~PETScVector();

  // Assignment operator (disabled)
  PETScVector& operator=(const PETScVector& x) = delete;

  /// Move Assignment operator
  PETScVector& operator=(PETScVector&& x);

  /// Create a copy of the vector
  ///
  /// Collective
  PETScVector copy() const;

  /// Return global size of vector
  std::int64_t size() const;

  /// Return local size of vector (belonging to this process)
  std::int32_t local_size() const;

  /// Return ownership range for calling rank
  std::array<std::int64_t, 2> local_range() const;

  /// Return MPI communicator
  MPI_Comm comm() const;

  /// Compute norm of vector
  ///
  /// Collective
  ///
  /// @param[in] type The norm type
  /// @returns The norm of the vector
  PetscReal norm(la::Norm type) const;

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
} // namespace dolfinx::la
