// Copyright (C) 2018 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <dolfin/common/types.h>
#include <memory>
#include <petscsys.h>
#include <vector>

namespace dolfin
{

namespace fem
{
class DirichletBC;
class Form;

namespace impl
{

void modify_bc(
    Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> b, const Form& L,
    const std::vector<std::shared_ptr<const Form>> a,
    const std::vector<std::shared_ptr<const DirichletBC>> bcs,
    const Eigen::Ref<const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> x0,
    double scale);

void modify_bc(Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> b,
               const Form& L, const std::vector<std::shared_ptr<const Form>> a,
               const std::vector<std::shared_ptr<const DirichletBC>> bcs,
               double scale);

/// Set bc values in owned (local) part of the PETSc Vec to scale*x_bc
/// value
void set_bc(Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> b,
            std::vector<std::shared_ptr<const DirichletBC>> bcs, double scale);

/// Set bc values in owned (local) part of the PETSc Vec to scale*(x0 -
/// x_bc)
void set_bc(
    Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> b,
    std::vector<std::shared_ptr<const DirichletBC>> bcs,
    const Eigen::Ref<const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> x0,
    double scale);

/// Assemble linear form into an Eigen vector. Assembly is performed
/// over the portion of the mesh belonging to the process. No
/// communication is performed. The Eigen vector must be passed in with
/// the correct size.
// FIXME: Clarify docstring regarding ghosts
void assemble(Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> b,
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

// Implementation of bc application
void _modify_bc(
    Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic, 1>> b, const Form& a,
    std::vector<std::shared_ptr<const DirichletBC>> bcs,
    Eigen::Ref<const Eigen::Array<PetscScalar, Eigen::Dynamic, 1>> x0,
    double scale);

} // namespace impl
} // namespace fem
} // namespace dolfin