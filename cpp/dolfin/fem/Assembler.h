// Copyright (C) 2018 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
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
// class PETScMatrix;
class PETScVector;
} // namespace la

namespace fem
{
// Forward declarations
class DirichletBC;
class Form;

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

  /// Return assembled matrix. Dirichlet rows/columns are zeroed, and
  /// '1' placed on diagonal.
  la::PETScMatrix assemble_matrix(BlockType type = BlockType::nested);

  /// Assemble matrix. Dirichlet rows/columns are zeroed, and '1'
  /// placed on diagonal
  void assemble(la::PETScMatrix& A);

  /// Return assembled vector. Boundary conditions have no effect on the
  /// assembled vector.
  la::PETScVector assemble_vector(BlockType type = BlockType::nested);

  /// Assemble vector and modify for boundary conditions.
  void assemble(la::PETScVector& b);

  // Assemble linear form into an Eigen vector. The Eigen vector must
  // the correct size. This local to a process.
  static void
      assemble(Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> b,
               const Form& L, const std::vector<std::shared_ptr<const Form>> a,
               const std::vector<std::shared_ptr<const DirichletBC>> bcs);

private:
  // Assemble linear form into a ghosted PETSc Vec
  static void
  assemble_single(Vec b, const Form& L,
                  const std::vector<std::shared_ptr<const Form>> a,
                  const std::vector<std::shared_ptr<const DirichletBC>> bcs);

  // Assemble linear form into a local PETSc Vec.
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

  // Get IndexSets (IS) for stacked index maps
  std::vector<IS> compute_index_sets(std::vector<const common::IndexMap*> maps);

  // Get sub-matrix
  la::PETScMatrix get_sub_matrix(const la::PETScMatrix& A, int i, int j);

  // Flag indicating if cell has a Dirichlet bc applied to it (true ->
  // has bc)
  // std::vector<bool> has_dirichlet_bc();

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

  // Set bc values in owned part of the PETSc Vec
  static void set_bc(Vec b, const Form& L,
                     std::vector<std::shared_ptr<const DirichletBC>> bcs);

  // Hack for setting bcs (set entries of b to be equal to boundary
  // value). Does not set ghosts. Size of b must be same as owned
  // length.
  static void set_bc(Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic, 1>> b,
                     const Form& L,
                     std::vector<std::shared_ptr<const DirichletBC>> bcs);

  // static std::vector<std::int32_t> compute_bc_indices(const DirichletBC& bc);

  // Bilinear and linear forms
  std::vector<std::vector<std::shared_ptr<const Form>>> _a;
  std::vector<std::shared_ptr<const Form>> _l;

  // Dirichlet boundary conditions
  std::vector<std::shared_ptr<const DirichletBC>> _bcs;

  // std::array<std::vector<IS>, 2> _block_is;
};
} // namespace fem
} // namespace dolfin
