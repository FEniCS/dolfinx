// Copyright (C) 2004-2018 Johan Hoffman, Johan Jansson, Anders Logg and Garth
// N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "utils.h"
#include <Eigen/Dense>
#include <array>
#include <cstdint>
#include <petscsys.h>
#include <petscvec.h>
#include <vector>

namespace dolfin
{
namespace la
{
class IndexMap;
}
namespace la
{

/// It is a simple wrapper for a PETSc vector pointer (Vec).
///
/// For advanced usage, access the PETSc Vec pointer using the function
/// vec() and use the standard PETSc interface.

class PETScVector
{
public:
  /// Create vector
  PETScVector(const common::IndexMap& map);

  /// Create vector
  PETScVector(MPI_Comm comm, std::array<std::int64_t, 2> range,
              const Eigen::Array<PetscInt, Eigen::Dynamic, 1>& ghost_indices,
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

  /// Return global size of vector
  std::int64_t size() const;

  /// Return local size of vector (belonging to this process)
  std::size_t local_size() const;

  /// Return ownership range for process
  std::array<std::int64_t, 2> local_range() const;

  /// Set all entries to 'a' using VecSet. This is local and does not
  /// update ghost entries.
  void set(PetscScalar a);

  /// A scalar 'a' using VecSet. This is local and does not update ghost
  /// entries.
  void shift(PetscScalar a);

  /// Multiply by scala a
  void scale(PetscScalar a);

  /// Multiply vector by vector x pointwise
  void mult(const PETScVector& x);

  /// Finalize assembly of vector. Communicates off-process entries
  /// added or set on this process to the owner, and receives from other
  /// processes changes to owned entries.
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

  /// Get block of values using local indices
  void get_local(PetscScalar* block, std::size_t m, const PetscInt* rows) const;

  /// Set block of values using global indices
  void set(const PetscScalar* block, std::size_t m, const PetscInt* rows);

  /// Set block of values using local indices
  void set_local(const PetscScalar* block, std::size_t m, const PetscInt* rows);

  /// Add block of values using global indices
  void add(const PetscScalar* block, std::size_t m, const PetscInt* rows);

  /// Add block of values using local indices
  void add_local(const PetscScalar* block, std::size_t m, const PetscInt* rows);

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
  void gather(PETScVector& y, const std::vector<PetscInt>& indices) const;

  /// Add multiple of given vector (AXPY operation, this = a*x + this)
  void axpy(PetscScalar a, const PETScVector& x);

  /// Replace all entries in the vector by their absolute values
  void abs();

  /// Return dot product with given vector. For complex vectors, the
  /// argument v gets complex conjugate.
  PetscScalar dot(const PETScVector& v) const;

  /// Return norm of vector
  PetscReal norm(la::Norm norm_type) const;

  /// Normalize vector with respect to the l2 norm. Returns the norm
  /// before normalization.
  PetscReal normalize();

  /// Return minimum value of vector, and location of entry. For complex
  /// vectors returns the minimum real part.
  std::pair<double, PetscInt> min() const;

  /// Return maximum value of vector, and location of entry. For complex
  /// vectors returns the maximum real part.
  std::pair<double, PetscInt> max() const;

  /// Return sum of entries
  PetscScalar sum() const;

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
} // namespace dolfin
