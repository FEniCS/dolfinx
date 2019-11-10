// Copyright (C) 2018-2019 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <dolfin/common/types.h>
#include <memory>
#include <petscmat.h>
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

// -- Scalar ----------------------------------------------------------------

/// Assemble functional into scalar. Caller is responsible for
/// accumulation across processes.
/// @param[in] M The form (functional) to assemble
/// @return The contribution to the form (functional) from the local
///         process
PetscScalar assemble_scalar(const Form& M);

// -- Vectors ----------------------------------------------------------------

/// Assemble linear form into an already allocated PETSc vector. Ghost
/// contributions are not accumulated (not sent to owner). Caller is
/// responsible for calling VecGhostUpdateBegin/End.
/// @param[in,out] b The PETsc vector to assemble the form into. The
///                  vector must already be initialised with the correct
///                  size. The process-local contribution of the form is
///                  assembled into this vector. It is not zeroed before
///                  assembly.
/// @param[in] L The linear form to assemble
void assemble_vector(Vec b, const Form& L);

/// Assemble linear form into an Eigen vector
/// @param[in,out] b The Eigen vector to be assembled. It will not be
///                  zeroed before assembly.
/// @param[in] L The linear forms to assemble into b
void
    assemble_vector(Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> b,
                    const Form& L);

// FIXME: clarify how x0 is used
// FIXME: if bcs entries are set
// FIXME: split into assemble and lift stages?

/// Re-assemble a blocked linear form L into the vector b. The vector is
/// modified for any boundary conditions such that:
///
///   b_i <- b_i - scale * A_ij g_j
///
// where L_i i assembled into b_i, and where i and j are the block
// indices. For non-blocked problem i = j / = 1. The boundary
// conditions bc1 are on the trial spaces V_j, which / can be different
// from the trial space of L (V_i). The forms in [a] / must have the
// same test space as L, but the trial space may differ.
// void assemble_vector(
//     Vec b, std::vector<const Form*> L,
//     const std::vector<std::vector<std::shared_ptr<const Form>>> a,
//     std::vector<std::shared_ptr<const DirichletBC>> bcs, const Vec x0,
//     double scale = 1.0);

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
void apply_lifting(
    Vec b, const std::vector<std::shared_ptr<const Form>>& a,
    const std::vector<std::vector<std::shared_ptr<const DirichletBC>>>& bcs1,
    const std::vector<Vec>& x0, double scale);

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
void apply_lifting(
    Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> b,
    const std::vector<std::shared_ptr<const Form>>& a,
    const std::vector<std::vector<std::shared_ptr<const DirichletBC>>>& bcs1,
    const std::vector<
        Eigen::Ref<const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>>>& x0,
    double scale);

// -- Matrices ---------------------------------------------------------------

/// Assemble bilinear form into a matrix. Matrix must already be
/// initialised. Does not zero or finalise the matrix.
/// @param[in,out] A The PETsc matrix to assemble the form into. The
///                  matrix size/layout must be initialised before
///                  calling this function. The matrix is not zeroed and
///                  it is not finalised (shared entries not
///                  communicated).
/// @param[in] a The bilinear from to assemble
/// @param[in] bcs Boundary conditions to apply. For boundary condition
///                dofs the row and column are zeroed. The diagonal
///                entry is not set.
void assemble_matrix(
    Mat A, const Form& a,
    const std::vector<std::shared_ptr<const DirichletBC>>& bcs);

/// Assemble bilinear form into a matrix. Matrix must already be
/// initialised. Does not zero or finalise the matrix.
/// @param[in,out] A The matrix to assemble in to. Matrix must be
///                  initialised.
/// @param[in] a The bilinear form to assemble
/// @param[in] bc0 Boundary condition markers for the rows. If bc[i] is
///                true then rows i in A will be zeroed. The index i is
///                a local index.
/// @param[in] bc1 Boundary condition markers for the columns. If bc[i]
///                is true then rows i in A will be zeroed. The index i
///                is a local index.
void assemble_matrix(Mat A, const Form& a, const std::vector<bool>& bc0,
                     const std::vector<bool>& bc1);

/// Re-assemble blocked bilinear forms into a matrix. Does not zero the
/// matrix.
void assemble_matrix_block(
    Mat A, const std::vector<std::vector<const Form*>>& a,
    const std::vector<std::shared_ptr<const DirichletBC>>& bcs,
    double diagonal = 1.0);

/// Re-assemble blocked bilinear forms into a matrix. Does not zero the
/// matrix.
void assemble_matrix_nest(
    Mat A, const std::vector<std::vector<const Form*>>& a,
    const std::vector<std::shared_ptr<const DirichletBC>>& bcs,
    double diagonal = 1.0);

/// Set diagonal
void set_diagonal(Mat A, const Form& a,
                  const std::vector<std::shared_ptr<const DirichletBC>>& bcs,
                  PetscScalar diagonal = 1.0);

/// Set diagonal
/// @param[in,out] A The matrix
/// @param[in] rows The rows, in local indices, for which to set the
///                 diagonal
/// @param[in] diagonal The value to set on the diagonal
void set_diagonal(
    Mat A, Eigen::Ref<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>> rows,
    PetscScalar diagonal = 1.0);

// -- Setting bcs ------------------------------------------------------------

// FIXME: Move these function elsewhere?

// FIXME: clarify x0
// FIXME: clarify what happens with ghosts

/// Set bc values in owned (local) part of the PETScVector, multiplied
/// by 'scale'. The vectors b and x0 must have the same local size. The
/// bcs should be on (sub-)spaces of the form L that b represents.
void set_bc(Vec b, const std::vector<std::shared_ptr<const DirichletBC>>& bcs,
            const Vec x0, double scale = 1.0);

/// Set bc values in owned (local) part of the PETScVector, multiplied
/// by 'scale'. The vectors b and x0 must have the same local size. The
/// bcs should be on (sub-)spaces of the form L that b represents.
void set_bc(
    Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> b,
    const std::vector<std::shared_ptr<const DirichletBC>>& bcs,
    const Eigen::Ref<const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>>& x0,
    double scale = 1.0);

/// Set bc values in owned (local) part of the PETScVector, multiplied
/// by 'scale'. The bcs should be on (sub-)spaces of the form L that b
/// represents.
void set_bc(Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> b,
            const std::vector<std::shared_ptr<const DirichletBC>>& bcs,
            double scale = 1.0);

// FIXME: Handle null block
// FIXME: Pass function spaces rather than forms
/// Arrange boundary conditions by block
/// @param[in] L Linear forms for each block
/// @param[in] bcs Boundary conditions
/// @return The boundary conditions collected by block, i.e.
///         bcs_block[i] is the list of boundary conditions applied to
///         L[i]. The order within bcs_block[i] preserves the input
///         order of the bcs array.
std::vector<std::vector<std::shared_ptr<const fem::DirichletBC>>>
bcs_rows(const std::vector<const Form*>& L,
         const std::vector<std::shared_ptr<const fem::DirichletBC>>& bcs);

// FIXME: Handle null block
// FIXME: Pass function spaces rather than forms
/// Arrange boundary conditions by block
/// @param[in] a Biinear forms for each block
/// @param[in] bcs Boundary conditions
/// @return The boundary conditions collected by block, i.e.
///         bcs_block[i] is the list of boundary conditions applied to
///         the trial space of a[i]. The order within bcs_block[i]
///         preserves the input order of the bcs array.
std::vector<std::vector<std::vector<std::shared_ptr<const fem::DirichletBC>>>>
bcs_cols(const std::vector<std::vector<std::shared_ptr<const Form>>>& a,
         const std::vector<std::shared_ptr<const DirichletBC>>& bcs);

} // namespace fem
} // namespace dolfin
