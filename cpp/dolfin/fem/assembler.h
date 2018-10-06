// Copyright (C) 2018 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <boost/variant.hpp>
#include <dolfin/common/types.h>
#include <dolfin/la/PETScMatrix.h>
#include <memory>
#include <petscvec.h>
#include <vector>

namespace dolfin
{
namespace function
{
class FunctionSpace;
} // namespace function
namespace la
{
class PETScMatrix;
class PETScVector;
} // namespace la

namespace fem
{
// Forward declarations
class DirichletBC;
class Form;

/// Assembly type for block forms
enum class BlockType
{
  monolithic,
  nested
};

/// Assemble form
boost::variant<double, la::PETScVector, la::PETScMatrix>
assemble(const Form& a);

/// Assemble blocked linear forms. The vector is modified such that b <-
/// b - A x_bc.
la::PETScVector
assemble(std::vector<const Form*> L,
         const std::vector<std::vector<std::shared_ptr<const Form>>> a,
         std::vector<std::shared_ptr<const DirichletBC>> bcs,
         BlockType block_type, double scale = 1.0);

/// Re-assemble blocked linear forms
void assemble(la::PETScVector& b, std::vector<const Form*> L,
              const std::vector<std::vector<std::shared_ptr<const Form>>> a,
              std::vector<std::shared_ptr<const DirichletBC>> bcs,
              double scale = 1.0);

/// Assemble blocked bilinear forms into a matrix
la::PETScMatrix assemble(const std::vector<std::vector<const Form*>> a,
                         std::vector<std::shared_ptr<const DirichletBC>> bcs,
                         BlockType block_type);

/// Re-assemble blocked bilinear forms into a matrix
void assemble(la::PETScMatrix& A, const std::vector<std::vector<const Form*>> a,
              std::vector<std::shared_ptr<const DirichletBC>> bcs);

// FIXME: Consider if L is required
/// Set bc values in owned (local) part of the PETScVector
void set_bc(la::PETScVector& b, const Form& L,
            std::vector<std::shared_ptr<const DirichletBC>> bcs, double scale);

} // namespace fem
} // namespace dolfin
