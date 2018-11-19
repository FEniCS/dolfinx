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
#include <dolfin/la/PETScVector.h>
#include <memory>
#include <petscvec.h>
#include <vector>

namespace dolfin
{
namespace function
{
class FunctionSpace;
} // namespace function

namespace fem
{
class DirichletBC;
class Form;

/// Assembly type for block forms
enum class BlockType
{
  monolithic,
  nested
};

/// Assemble variational form
boost::variant<double, la::PETScVector, la::PETScMatrix>
assemble(const Form& a);

/// Assemble blocked linear forms. The vector is modified such that
///  (i) b <- b - A x_bc, and
/// (ii) boundary condition values are inserted Dirichlet bcs position
///      in vector (multiplied by 'scale').
la::PETScVector
assemble(std::vector<const Form*> L,
         const std::vector<std::vector<std::shared_ptr<const Form>>> a,
         std::vector<std::shared_ptr<const DirichletBC>> bcs,
         const la::PETScVector* x0, BlockType block_type, double scale = 1.0);

/// Re-assemble blocked linear forms. The vector is modified such that:
///  (i) b <- b - A x_bc, and
/// (ii) boundary condition values are inserted Dirichlet bcs position
///      in vector (multiplied by 'scale').
void assemble(la::PETScVector& b, std::vector<const Form*> L,
              const std::vector<std::vector<std::shared_ptr<const Form>>> a,
              std::vector<std::shared_ptr<const DirichletBC>> bcs,
              const la::PETScVector* x0, double scale = 1.0);

/// Assemble blocked bilinear forms into a matrix. Rows and columns
/// associated with Dirichlet boundary conditions are zeroed, and
/// 'diagonal' is placed on the diagonal of Dirichlet bcs.
la::PETScMatrix assemble(const std::vector<std::vector<const Form*>> a,
                         std::vector<std::shared_ptr<const DirichletBC>> bcs,
                         BlockType block_type, double diagonal = 1.0);

/// Re-assemble blocked bilinear forms into a matrix
void assemble(la::PETScMatrix& A, const std::vector<std::vector<const Form*>> a,
              std::vector<std::shared_ptr<const DirichletBC>> bcs,
              double diagonal = 1.0);

// FIXME: Consider if L is required
/// Set bc values in owned (local) part of the PETScVector, multiplied by
/// 'scale'
void set_bc(la::PETScVector& b,
            std::vector<std::shared_ptr<const DirichletBC>> bcs,
            const la::PETScVector* x0, double scale = 1.0);

} // namespace fem
} // namespace dolfin
