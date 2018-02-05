// Copyright (C) 2018 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>
#include <memory>
#include <utility>
#include <vector>

namespace dolfin
{

// Forward declarations
class DirichletBC;
class Form;
class PETScMatrix;
class PETScVector;

namespace fem
{

class Assembler
{
public:
  /// Constructor
  Assembler(std::vector<std::vector<std::shared_ptr<const Form>>> a,
            std::vector<std::shared_ptr<const Form>> L,
            std::vector<std::shared_ptr<const DirichletBC>> bcs);

  // Assemble matrix
  void assemble(PETScMatrix& A);

  // Assemble vector
  void assemble(PETScVector& b);

  // Assemble matrix and vector
  void assemble(PETScMatrix& A, PETScVector& b);

private:
  // Assemble matrix. Dirichlet rows/columns are zeroed, with '1' placed on
  // diagonal
  static void assemble(PETScMatrix& A, const Form& a,
                       std::vector<std::shared_ptr<const DirichletBC>> bcs);

  // Assemble vector
  static void assemble(PETScVector& b, const Form& L);

  // Apply bcs to vector (b <- b - Ax, where x holds prescribed boundary
  // values)
  static void apply_bc(PETScVector& b, const Form& a,
                       std::vector<std::shared_ptr<const DirichletBC>> bcs);

  // Set bcs (set entries of b to be equal to boundary value)
  static void set_bc(PETScVector& b, const Form& L,
                     std::vector<std::shared_ptr<const DirichletBC>> bcs);

  // Bilinear and linear forms
  std::vector<std::vector<std::shared_ptr<const Form>>> _a;
  std::vector<std::shared_ptr<const Form>> _l;

  // Dirichlet boundary conditions
  std::vector<std::shared_ptr<const DirichletBC>> _bcs;
};
}
}
