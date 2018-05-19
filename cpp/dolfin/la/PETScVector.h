// Copyright (C) 2004-2016 Johan Hoffman, Johan Jansson, Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>
#include <cstdint>
#include <dolfin/common/ArrayView.h>
#include <dolfin/common/IndexMap.h>
#include <dolfin/common/types.h>
#include <memory>
#include <petscsys.h>
#include <petscvec.h>
#include <string>

namespace dolfin
{
namespace la
{

/// It is a simple wrapper for a PETSc vector pointer (Vec).
///
/// The interface is intentionally simple. For advanced usage,
/// access the PETSc Vec pointer using the function vec() and
/// use the standard PETSc interface.

class PETScVector
{
public:
  /// Create vector
  PETScVector(const common::IndexMap& map);

  /// Create vector
  PETScVector(MPI_Comm comm, std::array<std::int64_t, 2> range,
              const Eigen::Array<la_index_t, Eigen::Dynamic, 1>& ghost_indices,
              int block_size);

  // FIXME: Try to remove
  /// Create empty vector
  PETScVector();

  /// Copy constructor
  PETScVector(const PETScVector& x);

  /// Move constructor
  PETScVector(PETScVector&& x);

  /// Create vector wrapper of PETSc Vec pointer. The reference counter
  /// of the Vec will be increased, and decreased upon destruction of
  /// this object.
  explicit PETScVector(Vec x);

  /// Destructor
  virtual ~PETScVector();

  // Assignment operator (disabled)
  PETScVector& operator=(const PETScVector& x) = delete;

  /// Move Assignment operator
  PETScVector& operator=(PETScVector&& x);

  /// Return size of vector
  std::int64_t size() const;

  /// Return local size of vector
  std::size_t local_size() const;

  /// Return ownership range of a vector
  std::array<std::int64_t, 2> local_range() const;

  /// Set all entries to a
  void set(PetscScalar a);

  /// Finalize assembly of vector. Communicates off-process entries
  /// added or set on this process to the owner, and receives from other
  /// processes changes to locally owned entries.
  void apply();

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

  /// Return informal string representation (pretty-print)
  std::string str(bool verbose) const;

  /// Return true if vector is empty
  bool empty() const;

  /// Get block of values using global indices (all values must be
  /// owned by local process, ghosts cannot be accessed)
  void get(PetscScalar* block, std::size_t m,
           const dolfin::la_index_t* rows) const;

  /// Get block of values using local indices
  void get_local(PetscScalar* block, std::size_t m,
                 const dolfin::la_index_t* rows) const;

  /// Set block of values using global indices
  void set(const PetscScalar* block, std::size_t m,
           const dolfin::la_index_t* rows);

  /// Set block of values using local indices
  void set_local(const PetscScalar* block, std::size_t m,
                 const dolfin::la_index_t* rows);

  /// Add block of values using global indices
  void add(const PetscScalar* block, std::size_t m,
           const dolfin::la_index_t* rows);

  /// Add block of values using local indices
  void add_local(const PetscScalar* block, std::size_t m,
                 const dolfin::la_index_t* rows);

  /// Get all values on local process
  void get_local(std::vector<PetscScalar>& values) const;

  /// Set all values on local process
  void set_local(const std::vector<PetscScalar>& values);

  /// Add values to each entry on local process
  void add_local(const std::vector<PetscScalar>& values);

  /// Gather entries (given by global indices) into local
  /// (MPI_COMM_SELF) vector x. Provided x must be empty or of correct
  /// dimension (same as provided indices). This operation is
  /// collective.
  void gather(PETScVector& y,
              const std::vector<dolfin::la_index_t>& indices) const;

  /// Gather entries (given by global indices) into x. This operation is
  /// collective.
  void gather(std::vector<PetscScalar>& x,
              const std::vector<dolfin::la_index_t>& indices) const;

  /// Add multiple of given vector (AXPY operation)
  void axpy(PetscScalar a, const PETScVector& x);

  /// Replace all entries in the vector by their absolute values
  void abs();

  /// Return dot product with given vector. For complex vectors, the
  /// argument v gets complex conjugate.
  PetscScalar dot(const PETScVector& v) const;

  /// Return norm of vector
  double norm(std::string norm_type) const;

  /// Return minimum value of vector
  /// For complex vectors - return the minimum real part
  double min() const;

  /// Return maximum value of vector
  /// For complex vectors - return the maximum real part
  double max() const;

  /// Return sum of values of vector
  PetscScalar sum() const;

  /// Multiply vector by given number
  PETScVector& operator*=(PetscScalar a);

  /// Multiply vector by another vector pointwise
  PETScVector& operator*=(const PETScVector& x);

  /// Add given vector
  PETScVector& operator+=(const PETScVector& x);

  /// Add number to all components of a vector
  PETScVector& operator+=(PetscScalar a);

  /// Subtract given vector
  PETScVector& operator-=(const PETScVector& x);

  /// Subtract number from all components of a vector
  PETScVector& operator-=(PetscScalar a);

  /// Sets the prefix used by PETSc when searching the options database
  void set_options_prefix(std::string options_prefix);

  /// Returns the prefix used by PETSc when searching the options
  /// database
  std::string get_options_prefix() const;

  /// Call PETSc function VecSetFromOptions on the underlying Vec
  /// object
  void set_from_options();

  /// Return pointer to PETSc Vec object
  Vec vec() const;

private:
  // PETSc Vec pointer
  Vec _x;
};
} // namespace la
} // namespace dolfin
