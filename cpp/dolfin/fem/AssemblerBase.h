// Copyright (C) 2007-2009 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <string>
#include <utility>
#include <vector>

namespace dolfin
{
namespace la
{
class PETScMatrix;
class PETScVector;
}

namespace fem
{
class Form;

/// Provide some common functions used in assembler classes.
class AssemblerBase
{
public:
  /// Constructor
  AssemblerBase()
      : add_values(false), finalize_tensor(true), keep_diagonal(false)
  {
  }

  /// add_values (bool)
  ///     Default value is false.
  ///     This controls whether values are added to the given
  ///     tensor or if it is zeroed prior to assembly.
  bool add_values;

  /// finalize_tensor (bool)
  ///     Default value is true.
  ///     This controls whether the assembler finalizes the
  ///     given tensor after assembly is completed by calling
  ///     A.apply().
  bool finalize_tensor;

  /// keep_diagonal (bool)
  ///     Default value is false.
  ///     This controls whether the assembler enures that a diagonal
  ///     entry exists in an assembled matrix. It may be removed
  ///     if the matrix is finalised.
  bool keep_diagonal;

  /// Initialize global tensor
  /// @param[out] A (GenericTensor&)
  ///  GenericTensor to assemble into
  /// @param[in] a (Form&)
  ///  Form to assemble from
  void init_global_tensor(la::PETScVector& A, const Form& a);
  void init_global_tensor(la::PETScMatrix& A, const Form& a);

protected:
  /// Check form
  static void check(const Form& a);

  /// Pretty-printing for progress bar
  static std::string progress_message(std::size_t rank,
                                      std::string integral_type);
};
}
}
