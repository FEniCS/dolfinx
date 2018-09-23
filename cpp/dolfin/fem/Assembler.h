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
namespace common
{
class IndexMap;
} // namespace common
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

/// Assemble blocked linear forms
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

/// Re-assemble single linear form. The vector must already be
/// appropriately initialised. set_bc must be called after this call to
/// insert bc values. The 'test space' for L should be the same as the
/// test space for the bilinear forms in [a]. The vector b is modified
/// for boundary conditions in [bcs] that share a a trial space with
/// [a], i.e. b <- b - Ax.
void assemble(la::PETScVector& b, const Form& L,
              const std::vector<std::shared_ptr<const Form>> a,
              std::vector<std::shared_ptr<const DirichletBC>> bcs,
              double scale = 1.0);

/// Assemble blocked bilinear forms into a matrix
la::PETScMatrix assemble(const std::vector<std::vector<const Form*>> a,
                         std::vector<std::shared_ptr<const DirichletBC>> bcs,
                         BlockType block_type, double scale = 1.0);

/// Re-assemble blocked bilinear forms into a matrix
void assemble(la::PETScMatrix& A, const std::vector<std::vector<const Form*>> a,
              std::vector<std::shared_ptr<const DirichletBC>> bcs,
              double scale = 1.0);

/// Re-assemble bilinear form. The matrix must already be appropriately
/// initialised.
void assemble(la::PETScMatrix& A, const Form& a,
              std::vector<std::shared_ptr<const DirichletBC>> bcs,
              double scale = 1.0);

//----------------------------------------------------------------------------

// FIXME: Consider if L is required
/// Set bc values in owned (local) part of the PETScVector
void set_bc(la::PETScVector& b, const Form& L,
            std::vector<std::shared_ptr<const DirichletBC>> bcs);

// FIXME: Consider if L is required
/// Set bc values in owned (local) part of the PETSc Vec
void set_bc(Vec b, const Form& L,
            std::vector<std::shared_ptr<const DirichletBC>> bcs);

// FIXME: Consider if L is required
// Hack for setting bcs (set entries of b to be equal to boundary
// value). Does not set ghosts. Size of b must be same as owned
// length.
void set_bc(Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic, 1>> b,
            const Form& L, std::vector<std::shared_ptr<const DirichletBC>> bcs);

/// Assembly of LHS and RHS Forms with DirichletBC boundary conditions
/// applied
class Assembler
{
public:
  /// Assembly type for block forms
  enum class BlockType
  {
    monolithic,
    nested
  };

  /// Constructor
  Assembler(std::vector<std::vector<std::shared_ptr<const Form>>> a,
            std::vector<std::shared_ptr<const Form>> L,
            std::vector<std::shared_ptr<const DirichletBC>> bcs);

  /// Destructor
  ~Assembler();

  /// Assemble linear form into an Eigen vector. The Eigen vector must
  /// the correct size. This local to a process. The vector is modified
  /// for b <- b - A x_bc, where x_bc contains prescribed values. BC
  /// values are not inserted into bc positions.
  static void
      assemble(Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> b,
               const Form& L, const std::vector<std::shared_ptr<const Form>> a,
               const std::vector<std::shared_ptr<const DirichletBC>> bcs);

  /// Assemble linear form into a ghosted PETSc Vec. The vector is modified
  // for b <- b - A x_bc, where x_bc contains prescribed values, and BC
  // values set in bc positions.
  static void
  assemble(Vec b, const Form& L,
           const std::vector<std::shared_ptr<const Form>> a,
           const std::vector<std::shared_ptr<const DirichletBC>> bcs);

  // Get IndexSets (IS) for stacked index maps
  static std::vector<IS>
  compute_index_sets(std::vector<const common::IndexMap*> maps);

  // private:
  // Assemble linear form into a local PETSc Vec. The vector is modified
  // for b <- b - A x_bc, where x_bc contains prescribed values. BC
  // values are not inserted into bc positions.
  static void
  assemble_local(Vec& b, const Form& L,
                 const std::vector<std::shared_ptr<const Form>> a,
                 const std::vector<std::shared_ptr<const DirichletBC>> bcs);

  // Add '1' to diagonal for Dirichlet rows. Rows must be local to the
  // process.
  static void
  ident(la::PETScMatrix& A,
        const Eigen::Ref<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>> rows,
        PetscScalar diag = 1.0);

  // Get dof indices that have a boundary condition applied. Indices
  // are local and ghost indices are not included.
  static Eigen::Array<PetscInt, Eigen::Dynamic, 1>
  get_local_bc_rows(const function::FunctionSpace& V,
                    std::vector<std::shared_ptr<const DirichletBC>> bcs);

  // Get sub-matrix
  static la::PETScMatrix get_sub_matrix(const la::PETScMatrix& A, int i, int j);

  // Assemble matrix, with Dirichlet rows/columns zeroed. The matrix A
  // must already be initialised. The matrix may be a proxy, i.e. a view
  // into a larger matrix, and assembly is performed using local
  // indices. Matrix is not finalised.
  static void _assemble_matrix(la::PETScMatrix& A, const Form& a,
                               const std::vector<std::int32_t>& bc_dofs0,
                               const std::vector<std::int32_t>& bc_dofs1);

  // Modify RHS vector to account for boundary condition (b <- b - Ax,
  // where x holds prescribed boundary values)
  static void
      modify_bc(Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic, 1>> b,
                const Form& a,
                std::vector<std::shared_ptr<const DirichletBC>> bcs);

  // Bilinear and linear forms
  std::vector<std::vector<std::shared_ptr<const Form>>> _a;
  std::vector<std::shared_ptr<const Form>> _l;

  // Dirichlet boundary conditions
  std::vector<std::shared_ptr<const DirichletBC>> _bcs;

  // std::array<std::vector<IS>, 2> _block_is;
};
} // namespace fem
} // namespace dolfin
