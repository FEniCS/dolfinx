// Copyright (C) 2005-2006 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "PETScVector.h"
#include <cinttypes>
#include <cstddef>
#include <dolfin/common/Variable.h>
#include <dolfin/common/types.h>
#include <memory>
#include <petscmat.h>
#include <string>
#include <utility>

namespace dolfin
{
namespace la
{
class PETScVector;

/// This class is a base class for matrices that can be used in
/// PETScKrylovSolver.

class PETScBaseMatrix : public virtual common::Variable
{
public:
  /// Constructor
  PETScBaseMatrix() : _matA(nullptr) {}

  /// Constructor
  explicit PETScBaseMatrix(Mat A);

  /// Copy constructor
  PETScBaseMatrix(const PETScBaseMatrix& A);

  /// Destructor
  ~PETScBaseMatrix();

  /// Return number of rows (dim = 0) or columns (dim = 1)
  std::int64_t size(std::size_t dim) const;

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
  void init_vector(PETScVector& z, std::size_t dim) const;

  /// Return PETSc Mat pointer
  Mat mat() const { return _matA; }

  /// Return the MPI communicator
  MPI_Comm mpi_comm() const;

  /// Return informal string representation (pretty-print)
  virtual std::string str(bool verbose) const
  {
    return "No str function for this PETSc matrix operator.";
  }

protected:
  // PETSc Mat pointer
  Mat _matA;
};
}
}
