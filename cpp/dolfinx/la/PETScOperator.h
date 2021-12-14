// Copyright (C) 2005-2018 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>
#include <petscmat.h>
#include <stdint.h>

namespace dolfinx::la::petsc
{
class Vector;

/// This class is a base class for matrices that can be used in
/// petsc::KrylovSolver.

class Operator
{
public:
  /// Constructor
  Operator(Mat A, bool inc_ref_count);

  // Copy constructor (deleted)
  Operator(const Operator& A) = delete;

  /// Move constructor
  Operator(Operator&& A);

  /// Destructor
  virtual ~Operator();

  /// Assignment operator (deleted)
  Operator& operator=(const Operator& A) = delete;

  /// Move assignment operator
  Operator& operator=(Operator&& A);

  /// Return number of rows and columns (num_rows, num_cols). PETSc
  /// returns -1 if size has not been set.
  std::array<std::int64_t, 2> size() const;

  /// Initialize vector to be compatible with the matrix-vector product
  /// y = Ax. In the parallel case, size and layout are both important.
  ///
  /// @param[in] dim The dimension (axis): dim = 0 --> z = y, dim = 1
  ///                --> z = x
  Vector create_vector(std::size_t dim) const;

  /// Return PETSc Mat pointer
  Mat mat() const;

protected:
  // PETSc Mat pointer
  Mat _matA;
};
} // namespace dolfinx::la::petsc
