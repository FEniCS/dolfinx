// Copyright (C) 2018 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <dolfin/common/types.h>
#include <memory>
#include <petscvec.h>
#include <vector>

namespace dolfin
{

namespace fem
{
class DirichletBC;
class Form;

namespace impl
{

/// Assemble linear form into a ghosted PETSc Vec. The vector is
/// modified such that:
///
/// 1. If x0 is null b <- b - scale*A x_bc; or
///
/// 2. If x0 is not null b <- b - scale * A (x_bc - x0).
///
/// Essential bc dofs entries are *not* set.
///
/// This function essentially unwraps the pointer to the PETSc Vec data
/// and calls the Eigen-based functions for assembly.
void assemble_ghosted(Vec b, const Form& L,
                      const std::vector<std::shared_ptr<const Form>> a,
                      const std::vector<std::shared_ptr<const DirichletBC>> bcs,
                      const Vec x0, double scale);

/// Set bc values in owned (local) part of the PETSc Vec to scale*x_bc
/// value
void set_bc(Vec b, std::vector<std::shared_ptr<const DirichletBC>> bcs,
            double scale);

/// Set bc values in owned (local) part of the PETSc Vec to scale*(x0 -
/// x_bc)
void set_bc(Vec b, std::vector<std::shared_ptr<const DirichletBC>> bcs,
            const Vec x0, double scale);

/// Assemble linear form into an Eigen vector. Assembly is performed
/// over the portion of the mesh belonging to the process. No
/// communication is performed. The Eigen vector must be passed in with
/// the correct size.
// FIXME: Clarify docstring regarding ghosts
void assemble_eigen(Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> b,
                    const Form& L);

/// Modify RHS vector to account for boundary condition b <- b - scale*Ax_bc
void modify_bc(Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic, 1>> b,
               const Form& a,
               std::vector<std::shared_ptr<const DirichletBC>> bcs,
               double scale);

/// Modify RHS vector to account for boundary condition such that b <- b
/// - scale*A (x_bc - x0)
void
    modify_bc(Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic, 1>> b,
              const Form& a,
              std::vector<std::shared_ptr<const DirichletBC>> bcs,
              Eigen::Ref<const Eigen::Array<PetscScalar, Eigen::Dynamic, 1>> x0,
              double scale);

/// Set bc entries in b. Does not set ghosts and size of b must be same
/// as owned length.
///
/// - If length of x0 is zero, then b = scale* x_bc (bc dofs only)
///
/// - If length of x0 is equal to length of b, then b <- scale*(x0 -
///   x_bc) (bc dofs only)
void set_bc(
    Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic, 1>> b,
    std::vector<std::shared_ptr<const DirichletBC>> bcs,
    const Eigen::Ref<const Eigen::Array<PetscScalar, Eigen::Dynamic, 1>> x0,
    double scale);

// Implementation of bc application
void _modify_bc(
    Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic, 1>> b, const Form& a,
    std::vector<std::shared_ptr<const DirichletBC>> bcs,
    Eigen::Ref<const Eigen::Array<PetscScalar, Eigen::Dynamic, 1>> x0,
    double scale);

// Assemble linear form into a local PETSc Vec. The vector b is modified
// to account for essential (Dirichlet) boundary conditions.
//
// The implementation of this function unwraps the PETSc Vec as a plain
// pointer, and call the Eigen-based assembly interface.
void _assemble_local(Vec b, const Form& L,
                     const std::vector<std::shared_ptr<const Form>> a,
                     const std::vector<std::shared_ptr<const DirichletBC>> bcs,
                     const Vec x0, double scale);
} // namespace impl
} // namespace fem
} // namespace dolfin