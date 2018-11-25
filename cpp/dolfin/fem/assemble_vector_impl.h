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

/// Set bc values in owned (local) part of the PETSc Vec
void set_bc(Vec b, std::vector<std::shared_ptr<const DirichletBC>> bcs, Vec x0,
            double scale);

// Hack for setting bcs (set entries of b to be equal to boundary
// value). Does not set ghosts. Size of b must be same as owned length.
void set_bc(
    Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic, 1>> b,
    std::vector<std::shared_ptr<const DirichletBC>> bcs,
    const Eigen::Ref<const Eigen::Array<PetscScalar, Eigen::Dynamic, 1>> x0,
    double scale);

/// Modify RHS vector to account for boundary condition b <- b - Ax_bc
void modify_bc(Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic, 1>> b,
               const Form& a,
               std::vector<std::shared_ptr<const DirichletBC>> bcs);

/// Modify RHS vector to account for boundary condition such that b <- b
/// - A (x0 - x_bc)
void modify_bc(
    Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic, 1>> b, const Form& a,
    std::vector<std::shared_ptr<const DirichletBC>> bcs,
    Eigen::Ref<const Eigen::Array<PetscScalar, Eigen::Dynamic, 1>> x0);

/// Implementation of bc application
void _modify_bc(
    Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic, 1>> b, const Form& a,
    std::vector<std::shared_ptr<const DirichletBC>> bcs,
    Eigen::Ref<const Eigen::Array<PetscScalar, Eigen::Dynamic, 1>> x0);

/// Assemble linear form into an Eigen vector. Assembly is performed
/// over the portion of the mesh belonging to the process. No
/// communication is performed. The Eigen vector must be passed in with
/// the correct size.
// FIXME: Clarify docstring regarding ghosts
void assemble_eigen(Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> b,
                    const Form& L);

/// Assemble linear form into a ghosted PETSc Vec. The vector is
/// modified for b <- b - A x_bc, where x_bc contains prescribed values,
/// and BC values set in bc positions.
void assemble_ghosted(Vec b, const Form& L,
                      const std::vector<std::shared_ptr<const Form>> a,
                      const std::vector<std::shared_ptr<const DirichletBC>> bcs,
                      const Vec x0, double scale);

/// Assemble linear form into a local PETSc Vec. The vector is modified
/// for b <- b - A x_bc, where x_bc contains prescribed values. BC
/// values are not inserted into bc positions.
void assemble_local(Vec& b, const Form& L,
                    const std::vector<std::shared_ptr<const Form>> a,
                    const std::vector<std::shared_ptr<const DirichletBC>> bcs,
                    const Vec x0);
} // namespace impl
} // namespace fem
} // namespace dolfin