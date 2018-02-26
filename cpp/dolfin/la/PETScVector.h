// Copyright (C) 2004-2016 Johan Hoffman, Johan Jansson, Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#ifdef HAS_PETSC

#include "PETScObject.h"
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

/// A simple vector class based on PETSc.
///
/// It is a simple wrapper for a PETSc vector pointer (Vec)
/// implementing the GenericTensor interface.
///
/// The interface is intentionally simple. For advanced usage,
/// access the PETSc Vec pointer using the function vec() and
/// use the standard PETSc interface.

class PETScVector : public PETScObject
{
public:
  /// Create empty vector on an MPI communicator
  explicit PETScVector(MPI_Comm comm);

  /// Copy constructor
  PETScVector(const PETScVector& x);

  /// Create vector wrapper of PETSc Vec pointer. The reference
  /// counter of the Vec will be increased, and decreased upon
  /// destruction of this object.
  explicit PETScVector(Vec x);

  /// Destructor
  virtual ~PETScVector();

  /// Initialize vector to global size N
  void init(std::size_t N);

  /// Initialize vector with given ownership range
  void init(std::array<std::int64_t, 2> range);

  /// Initialize vector with given ownership range and with ghost
  /// values
  void init(std::array<std::int64_t, 2> range,
            const std::vector<la_index_t>& local_to_global_map,
            const std::vector<la_index_t>& ghost_indices, int block_size);

  /// Return size of vector
  std::int64_t size() const;

  /// Return local size of vector
  std::size_t local_size() const;

  /// Return ownership range of a vector
  std::array<std::int64_t, 2> local_range() const;

  /// Set all entries to zero and keep any sparse structure
  void zero();

  /// Finalize assembly of tensor
  void apply();

  /// Return MPI communicator
  MPI_Comm mpi_comm() const;

  /// Return informal string representation (pretty-print)
  std::string str(bool verbose) const;

  /// Return true if vector is empty
  bool empty() const;

  /// Determine whether global vector index is owned by this process
  bool owns_index(std::size_t i) const;

  /// Get block of values using global indices (all values must be
  /// owned by local process, ghosts cannot be accessed)
  void get(double* block, std::size_t m, const dolfin::la_index_t* rows) const;

  /// Get block of values using local indices
  void get_local(double* block, std::size_t m,
                 const dolfin::la_index_t* rows) const;

  /// Set block of values using global indices
  void set(const double* block, std::size_t m, const dolfin::la_index_t* rows);

  /// Set block of values using local indices
  void set_local(const double* block, std::size_t m,
                 const dolfin::la_index_t* rows);

  /// Add block of values using global indices
  void add(const double* block, std::size_t m, const dolfin::la_index_t* rows);

  /// Add block of values using local indices
  void add_local(const double* block, std::size_t m,
                 const dolfin::la_index_t* rows);

  /// Get all values on local process
  void get_local(std::vector<double>& values) const;

  /// Set all values on local process
  void set_local(const std::vector<double>& values);

  /// Add values to each entry on local process
  void add_local(const std::vector<double>& values);

  /// Gather entries (given by global indices) into local
  /// (MPI_COMM_SELF) vector x. Provided x must be empty or of
  /// correct dimension (same as provided indices).  This operation
  /// is collective.
  void gather(PETScVector& y,
              const std::vector<dolfin::la_index_t>& indices) const;

  /// Gather entries (given by global indices) into x.  This
  /// operation is collective
  void gather(std::vector<double>& x,
              const std::vector<dolfin::la_index_t>& indices) const;

  /// Gather all entries into x on process 0.
  /// This operation is collective
  void gather_on_zero(std::vector<double>& x) const;

  /// Add multiple of given vector (AXPY operation)
  void axpy(double a, const PETScVector& x);

  /// Replace all entries in the vector by their absolute values
  void abs();

  /// Return dot product with given vector
  double dot(const PETScVector& v) const;

  /// Return norm of vector
  double norm(std::string norm_type) const;

  /// Return minimum value of vector
  double min() const;

  /// Return maximum value of vector
  double max() const;

  /// Return sum of values of vector
  double sum() const;

  /// Multiply vector by given number
  const PETScVector& operator*=(double a);

  /// Multiply vector by another vector pointwise
  const PETScVector& operator*=(const PETScVector& x);

  /// Divide vector by given number
  const PETScVector& operator/=(double a);

  /// Add given vector
  const PETScVector& operator+=(const PETScVector& x);

  /// Add number to all components of a vector
  const PETScVector& operator+=(double a);

  /// Subtract given vector
  const PETScVector& operator-=(const PETScVector& x);

  /// Subtract number from all components of a vector
  const PETScVector& operator-=(double a);

  /// Assignment operator
  const PETScVector& operator=(const PETScVector& x);

  /// Assignment operator
  const PETScVector& operator=(double a);

  /// Update values shared from remote processes
  void update_ghost_values();

  /// Sets the prefix used by PETSc when searching the options
  /// database
  void set_options_prefix(std::string options_prefix);

  /// Returns the prefix used by PETSc when searching the options
  /// database
  std::string get_options_prefix() const;

  /// Call PETSc function VecSetFromOptions on the underlying Vec
  /// object
  void set_from_options();

  /// Return pointer to PETSc Vec object
  Vec vec() const;

  /// Switch underlying PETSc object. Intended for internal library
  /// usage.
  void reset(Vec vec);

private:
  // PETSc Vec pointer
  Vec _x;
};
}

#endif
