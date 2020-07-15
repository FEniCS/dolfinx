// Copyright (C) 2018-2020 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfinx/la/PETScMatrix.h>
#include <dolfinx/la/PETScVector.h>
#include <memory>
#include <petscvec.h>
#include <vector>

namespace dolfinx
{
namespace function
{
class FunctionSpace;
} // namespace function

namespace fem
{
template <typename T>
class DirichletBC;
template <typename T>
class Form;

/// Create a matrix
/// @param[in] a  A bilinear form
/// @return A matrix. The matrix is not zeroed.
la::PETScMatrix create_matrix(const Form<PetscScalar>& a);

/// Initialise monolithic matrix for an array for bilinear forms. Matrix
/// is not zeroed.
la::PETScMatrix create_matrix_block(
    const Eigen::Ref<
        const Eigen::Array<const fem::Form<PetscScalar>*, Eigen::Dynamic,
                           Eigen::Dynamic, Eigen::RowMajor>>& a);

/// Create nested (MatNest) matrix. Matrix is not zeroed.
la::PETScMatrix create_matrix_nest(
    const Eigen::Ref<
        const Eigen::Array<const fem::Form<PetscScalar>*, Eigen::Dynamic,
                           Eigen::Dynamic, Eigen::RowMajor>>& a);

/// Initialise monolithic vector. Vector is not zeroed.
la::PETScVector create_vector_block(
    const std::vector<std::reference_wrapper<const common::IndexMap>>& maps);

/// Create nested (VecNest) vector. Vector is not zeroed.
la::PETScVector
create_vector_nest(const std::vector<const common::IndexMap*>& maps);

// -- Vectors ----------------------------------------------------------------

/// Assemble linear form into an already allocated PETSc vector. Ghost
/// contributions are not accumulated (not sent to owner). Caller is
/// responsible for calling VecGhostUpdateBegin/End.
///
/// @param[in,out] b The PETsc vector to assemble the form into. The
///   vector must already be initialised with the correct size. The
///   process-local contribution of the form is assembled into this
///   vector. It is not zeroed before assembly.
/// @param[in] L The linear form to assemble
void assemble_vector_petsc(Vec b, const Form<PetscScalar>& L);

// FIXME: clarify how x0 is used
// FIXME: if bcs entries are set

// FIXME: need to pass an array of Vec for x0?
// FIXME: clarify zeroing of vector

/// Modify b such that:
///
///   b <- b - scale * A_j (g_j - x0_j)
///
/// where j is a block (nest) index. For a non-blocked problem j = 0. The
/// boundary conditions bcs1 are on the trial spaces V_j. The forms in
/// [a] must have the same test space as L (from which b was built), but the
/// trial space may differ. If x0 is not supplied, then it is treated as
/// zero.
///
/// Ghost contributions are not accumulated (not sent to owner). Caller
/// is responsible for calling VecGhostUpdateBegin/End.
void apply_lifting_petsc(
    Vec b, const std::vector<std::shared_ptr<const Form<PetscScalar>>>& a,
    const std::vector<
        std::vector<std::shared_ptr<const DirichletBC<PetscScalar>>>>& bcs1,
    const std::vector<Vec>& x0, double scale);

// -- Setting bcs ------------------------------------------------------------

// FIXME: Move these function elsewhere?

// FIXME: clarify x0
// FIXME: clarify what happens with ghosts

/// Set bc values in owned (local) part of the PETScVector, multiplied
/// by 'scale'. The vectors b and x0 must have the same local size. The
/// bcs should be on (sub-)spaces of the form L that b represents.
void set_bc_petsc(
    Vec b,
    const std::vector<std::shared_ptr<const DirichletBC<PetscScalar>>>& bcs,
    const Vec x0, double scale = 1.0);

} // namespace fem
} // namespace dolfinx
