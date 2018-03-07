// Copyright (C) 2018 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>
#include <boost/multi_array.hpp>
#include <memory>
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
// Forward declarations
class DirichletBC;
class Form;

class Assembler
{
public:
  enum class BlockType
  {
    monolithic,
    nested
  };

  /// Constructor
  Assembler(std::vector<std::vector<std::shared_ptr<const Form>>> a,
            std::vector<std::shared_ptr<const Form>> L,
            std::vector<std::shared_ptr<const DirichletBC>> bcs);

  // Assemble matrix. Dirichlet rows/columns are zeroed, with '1' placed on
  // diagonal
  void assemble(la::PETScMatrix& A, BlockType type = BlockType::nested);

  // Assemble vector
  void assemble(la::PETScVector& b);

  // Assemble matrix and vector
  void assemble(la::PETScMatrix& A, la::PETScVector& b);

private:
  // Assemble matrix. Dirichlet rows/columns are zeroed, with '1' placed on
  // diagonal
  static void assemble(la::PETScMatrix& A, const Form& a,
                       std::vector<std::shared_ptr<const DirichletBC>> bcs);

  // Assemble vector
  static void assemble(la::PETScVector& b, const Form& L);

  // Apply bcs to vector (b <- b - Ax, where x holds prescribed boundary
  // values)
  static void apply_bc(la::PETScVector& b, const Form& a,
                       std::vector<std::shared_ptr<const DirichletBC>> bcs);

  // Set bcs (set entries of b to be equal to boundary value)
  static void set_bc(la::PETScVector& b, const Form& L,
                     std::vector<std::shared_ptr<const DirichletBC>> bcs);

  // Bilinear and linear forms
  std::vector<std::vector<std::shared_ptr<const Form>>> _a;
  std::vector<std::shared_ptr<const Form>> _l;

  // Dirichlet boundary conditions
  std::vector<std::shared_ptr<const DirichletBC>> _bcs;
};
}
}
