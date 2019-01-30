// Copyright (C) 2018-2019 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <boost/variant.hpp>
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

// Assemble functional into scalar. Scalar is summed across all
// processes.
PetscScalar assemble_scalar(const Form& M);

// -- Vectors ----------------------------------------------------------------

/// Assemble linear form into an already allocated vector. Ghost
/// contributions are no accumulated (not sent to owner). Caller is
/// responsible for calling VecGhostUpdateBegin/End.
void assemble_vector(Vec b, const Form& L);

// FIXME: clarify how x0 is used
// FIXME: if bcs entries are set
// FIXME: split into assemble and lift stages?

/// Re-assemble a blocked linear form L into the vector b. The vector is
/// modified for any boundary conditions such that:
///
///   b_i <- b_i - scale * A_ij g_j
///
// where L_i i assembled into b_i, and where i and j are the block
// indices. For non-blocked probelem i = j / = 1. The boundary
// conditions bc1 are on the trial spaces V_j, which / can be different
// from the trial space of L (V_i). The forms in [a] / must have the
// same test space as L, but the trial space may differ.
void assemble_vector(
    Vec b, std::vector<const Form*> L,
    const std::vector<std::vector<std::shared_ptr<const Form>>> a,
    std::vector<std::shared_ptr<const DirichletBC>> bcs, const Vec x0,
    double scale = 1.0);

// FIXME: need to pass an array of Vec for x0?
// FIXME: clarify zeroing of vector

/// Modify b such that:
///
///   b <- b - scale * A_j (g_j - x0_j)
///
/// where j is a block (nest) index. For non-blocked probelem j = 1. The
/// boundary conditions bc1 are on the trial spaces V_j. The forms in
/// [a] must have the same test space as L (from b was built), but the
/// trial space may differ. If x0 is not supplied, then it is treated as
/// zero.
///
/// Ghost contributions are no accumulated (not sent to owner). Caller
/// is responsible for calling VecGhostUpdateBegin/End.
void apply_lifting(
    Vec b, const std::vector<std::shared_ptr<const Form>> a,
    std::vector<std::vector<std::shared_ptr<const DirichletBC>>> bcs1,
    const std::vector<Vec> x0, double scale);

// -- Matrices ---------------------------------------------------------------

/// Re-assemble blocked bilinear forms into a matrix. Does not zero the
/// matrix.
void assemble_matrix(Mat A, const std::vector<std::vector<const Form*>> a,
                     std::vector<std::shared_ptr<const DirichletBC>> bcs,
                     double diagonal = 1.0, bool use_nest_extract = true);

/// Assemble bilinear form into a matrix. Matrix must be initialised.
/// Does not zero or finalise the matrix.
void assemble_matrix(Mat A, const Form& a,
                     std::vector<std::shared_ptr<const DirichletBC>> bcs,
                     double diagonal = 1.0);

// -- Setting bcs ------------------------------------------------------------

// FIXME: Move these function elsewhere?

// FIXME: clarify x0
// FIXME: clarify what happens with ghosts
// FIXME: sort out case when x0 is nested, i.e. len(b) \ne len(x0)

/// Set bc values in owned (local) part of the PETScVector, multiplied
/// by 'scale'. The vectors b and x0 must have the same local size. The
/// bcs should be on (sub-)spaces of the form L that b represents.
void set_bc(Vec b, std::vector<std::shared_ptr<const DirichletBC>> bcs,
            const Vec x0, double scale = 1.0);
} // namespace fem
} // namespace dolfin
