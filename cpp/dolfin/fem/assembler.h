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
boost::variant<PetscScalar, la::PETScVector, la::PETScMatrix>
assemble(const Form& a);

// -- Vectors ----------------------------------------------------------------

// FIXME: clarify how x0 is used
// FIXME: if bcs entries are set
// FIXME: add doc for BlockType block_type

/// Assemble a blocked linear form L into the vector and return the
/// vector b. The vector is modified for any boundary conditions such
/// that:
///
///   b_i <- b_i - scale * A_ij g_j
///
// where L_i i assembled into b_i, and where i and j are the block
// indices. For non-blocked probelem i = j / = 1. The boundary
// conditions bc1 are on the trial spaces V_j, which / can be different
// from the trial space of L (V_i). The forms in [a] / must have the
// same test space as L, but the trial space may differ.
la::PETScVector
assemble(std::vector<const Form*> L,
         const std::vector<std::vector<std::shared_ptr<const Form>>> a,
         std::vector<std::shared_ptr<const DirichletBC>> bcs,
         const la::PETScVector* x0, BlockType block_type, double scale = 1.0);

// FIXME: clarify how x0 is used
// FIXME: if bcs entries are set

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
void assemble(la::PETScVector& b, std::vector<const Form*> L,
              const std::vector<std::vector<std::shared_ptr<const Form>>> a,
              std::vector<std::shared_ptr<const DirichletBC>> bcs,
              const la::PETScVector* x0, double scale = 1.0);

// FIXME: clarify how x0 is used
// FIXME: need to pass an array of Vec for x0?

/// Re-assemble a single linear form L into the vector b. The vector is
/// modified for any boundary conditions such that:
///
///   b_i <- b_i - scale * A_ij g_j
///
/// where i and j are the block indices. For non-blocked probelem i = j
/// = 1. The boundary conditions bc1 are on the trial spaces V_j, which
/// can be different from the trial space of L (V_i). The forms in [a]
/// must have the same test space as L, but the trial space may differ.
void assemble_petsc(
    Vec b, const Form& L, const std::vector<std::shared_ptr<const Form>> a,
    std::vector<std::vector<std::shared_ptr<const DirichletBC>>> bcs1,
    const Vec x0, double scale = 1.0);

/// Assemble a single linear form L into the vector b. The vector b must
/// already be initialized. It will not be zeroed.
///
/// If boundary conditions (DirichletBC) are supplied, the vector is
/// modified such that:
///
///   b_i <- b_i - scale * A_ij (g_j - x0_j)
///
/// where i and j are the block (nest) indices. For non-blocked probelem
/// i = j = 1. The boundary conditions bc1 are on the trial spaces V_j,
/// which can be different from the trial space of L (V_i). The forms in
/// [a] must have the same test space as L, but the trial space may
/// differ. If x0 is not supplied, then it is treated as zero.
void assemble_eigen(
    Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> b, const Form& L,
    const std::vector<std::shared_ptr<const Form>> a,
    std::vector<std::vector<std::shared_ptr<const DirichletBC>>> bcs1,
    std::vector<Eigen::Ref<const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>>>
        x0,
    double scale = 1.0);

// -- Matrices ---------------------------------------------------------------

/// Assemble blocked bilinear forms into a matrix. Rows and columns
/// associated with Dirichlet boundary conditions are zeroed, and
/// 'diagonal' is placed on the diagonal of Dirichlet bcs.
la::PETScMatrix assemble(const std::vector<std::vector<const Form*>> a,
                         std::vector<std::shared_ptr<const DirichletBC>> bcs,
                         BlockType block_type, double diagonal = 1.0);

/// Re-assemble blocked bilinear forms into a matrix
void assemble(la::PETScMatrix& A, const std::vector<std::vector<const Form*>> a,
              std::vector<std::shared_ptr<const DirichletBC>> bcs,
              double diagonal = 1.0, bool use_nest_extract = true);

/// Assemble bilinear form into a matrix. Matrix must be initialised.
/// Does not finalise matrix.
void assemble_petsc(Mat A, const Form& a,
                    std::vector<std::shared_ptr<const DirichletBC>> bcs,
                    double diagonal);

// -- Setting bcs ------------------------------------------------------------

// FIXME: Move these function elsewhere?

// FIXME: clarify x0
// FIXME: clarify what happens with ghosts
// FIXME: sort out case when x0 is nested, i.e. len(b) \ne len(x0)

/// Set bc values in owned (local) part of the PETScVector, multiplied
/// by 'scale'. The vectors b and x0 must have the same local size. The
/// bcs should be on (sub-)spaces of the form L that b represents.
void set_bc(la::PETScVector& b,
            std::vector<std::shared_ptr<const DirichletBC>> bcs,
            const la::PETScVector* x0, double scale = 1.0);

/// Set bc values in owned (local) part of the PETScVector, multiplied
/// by 'scale'. The vectors b and x0 must have the same local size. The
/// bcs should be on (sub-)spaces of the form L that b represents.
void set_bc_petsc(Vec b, std::vector<std::shared_ptr<const DirichletBC>> bcs,
                  const Vec x0, double scale = 1.0);
} // namespace fem
} // namespace dolfin
