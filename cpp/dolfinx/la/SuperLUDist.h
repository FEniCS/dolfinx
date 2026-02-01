// Copyright (C) 2026 Jack S. Hale, Chris N. Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#ifdef HAS_SUPERLU_DIST
#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/Vector.h>

namespace dolfinx::la
{

// Trick for forward declaration of anonymous structs
// Avoids including SuperLU_DIST headers in DOLFINx headers
class SuperLUDistStructs
{
public:
  struct SuperMatrix;
  struct gridinfo_t;
  struct vec_int_t;
};

/// Linear solver using SuperLU_DIST
template <typename T>
class SuperLUDistSolver
{
public:
  /// @brief SuperLU_DIST solver wrapper
  /// @param Amat Assembled matrix to invert
  /// @param verbose Verbosity
  /// @tparam T Scalar type
  SuperLUDistSolver(std::shared_ptr<const MatrixCSR<T>> Amat,
                    bool verbose = false);

  /// Copy constructor
  SuperLUDistSolver(const SuperLUDistSolver&) = delete;

  /// Copy assignment
  SuperLUDistSolver& operator=(const SuperLUDistSolver&) = delete;

  /// Solve linear system Au = b
  /// @param b Right-hand side Vector
  /// @param u Solution Vector
  /// @note Must be compatible with A
  int solve(const dolfinx::la::Vector<T>& b, dolfinx::la::Vector<T>& u) const;

private:
  // Call library cleanup and delete pointer. For use with std::unique_ptr
  // holding gridinfo_t.
  struct GridInfoDeleter
  {
    void operator()(SuperLUDistStructs::gridinfo_t* g) const noexcept;
  };

  // Call library cleanup and delete pointer. For use with std::unique_ptr
  // holding SuperMatrix.
  struct SuperMatrixDeleter
  {
    void operator()(SuperLUDistStructs::SuperMatrix* A) const noexcept;
  };

  struct VecIntDeleter
  {
    void operator()(SuperLUDistStructs::vec_int_t* v) const noexcept;
  };

  /// Set the matrix operator
  void set_operator(const la::MatrixCSR<T>& Amat);

  // Saved matrix operator with rows and cols in
  // required integer type
  std::shared_ptr<const la::MatrixCSR<T>> _Amat;

  // Pointer to struct gridinfo_t
  std::unique_ptr<SuperLUDistStructs::gridinfo_t, GridInfoDeleter> _gridinfo;
  // Pointer to SuperMatrix
  std::unique_ptr<SuperLUDistStructs::SuperMatrix, SuperMatrixDeleter>
      _supermatrix;

  // cols and rowptr are required in opaque type "int_t" of SuperLU_DIST.
  std::unique_ptr<SuperLUDistStructs::vec_int_t, VecIntDeleter> cols;
  std::unique_ptr<SuperLUDistStructs::vec_int_t, VecIntDeleter> rowptr;

  // Flag for diagnostic output
  bool _verbose;
};
} // namespace dolfinx::la
#endif
