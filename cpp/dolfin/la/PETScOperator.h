// Copyright (C) 2005-2006 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>
#include <cinttypes>
#include <cstddef>
#include <petscmat.h>
#include <string>

namespace dolfin
{
namespace la
{
class PETScVector;

/// This class is a base class for matrices that can be used in
/// PETScKrylovSolver.

class PETScOperator
{
public:
  /// Constructor
  PETScOperator() : _matA(nullptr) {}

  /// Constructor
  explicit PETScOperator(Mat A);

  /// Copy constructor (deleted)
  PETScOperator(const PETScOperator& A) = delete;

  /// Move constructor
  PETScOperator(PETScOperator&& A);

  /// Destructor
  ~PETScOperator();

  /// Assignment operator (deleted)
  PETScOperator& operator=(const PETScOperator& A) = delete;

  /// Move assignment operator
  PETScOperator& operator=(PETScOperator&& A);

  /// Return number of rows and columns (num_rows, num_cols). PETSc
  /// returns -1 if size has not been set.
  std::array<std::int64_t, 2> size() const;

  /// Return local range along dimension dim
  std::array<std::int64_t, 2> local_range(std::size_t dim) const;

  /// Initialize vector to be compatible with the matrix-vector product
  /// y = Ax. In the parallel case, both size and layout are
  /// important.
  ///
  /// @param z (PETScVector&)
  ///         Vector to initialise
  /// @param      dim (std::size_t)
  ///         The dimension (axis): dim = 0 --> z = y, dim = 1 --> z = x
  PETScVector init_vector(std::size_t dim) const;

  /// Return PETSc Mat pointer
  Mat mat() const;

  /// Return the MPI communicator
  MPI_Comm mpi_comm() const;

  /// Return informal string representation (pretty-print)
  virtual std::string str(bool verbose) const;

protected:
  // PETSc Mat pointer
  Mat _matA;
};
} // namespace la
} // namespace dolfin
