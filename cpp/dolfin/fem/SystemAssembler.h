// Copyright (C) 2008-2015 Kent-Andre Mardal and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "DirichletBC.h"
#include <Eigen/Dense>
#include <array>
#include <map>
#include <memory>
#include <utility>
#include <vector>

struct ufc_cell_integral;
struct ufc_exterior_facet_integral;
struct ufc_interior_facet_integral;

namespace dolfin
{
namespace la
{
class PETScMatrix;
class PETScVector;
} // namespace la

namespace common
{
template <typename T>
class ArrayView;
}

namespace function
{
class Function;
}

namespace mesh
{
class Cell;
class Facet;
class Mesh;
template <typename T>
class MeshFunction;
} // namespace mesh

namespace fem
{
class Form;
class GenericDofMap;
class UFC;

/// This class provides an assembler for systems of the form Ax =
/// b. It differs from the default DOLFIN assembler in that it
/// applies boundary conditions at the time of assembly, which
/// preserves any symmetries in A.

class SystemAssembler
{
public:
  /// Constructor
  SystemAssembler(std::shared_ptr<const Form> a, std::shared_ptr<const Form> L,
                  std::vector<std::shared_ptr<const DirichletBC>> bcs);

  /// Assemble system (A, b)
  void assemble(la::PETScMatrix& A, la::PETScVector& b);

  /// Assemble matrix A
  void assemble(la::PETScMatrix& A);

  /// Assemble vector b
  void assemble(la::PETScVector& b);

  /// Assemble system (A, b) for (negative) increment dx, where x =
  /// x0 - dx is solution to system a == -L subject to bcs.
  /// Suitable for use inside a (quasi-)Newton solver.
  void assemble(la::PETScMatrix& A, la::PETScVector& b,
                const la::PETScVector& x0);

  /// Assemble rhs vector b for (negative) increment dx, where x =
  /// x0 - dx is solution to system a == -L subject to bcs.
  /// Suitable for use inside a (quasi-)Newton solver.
  void assemble(la::PETScVector& b, const la::PETScVector& x0);

private:
  // Class to hold temporary data
  class Scratch
  {
  public:
    Scratch(const Form& a, const Form& L);
    ~Scratch();
    std::array<std::vector<PetscScalar>, 2> Ae;
  };

  // Check form arity
  static void check_arity(std::shared_ptr<const Form> a,
                          std::shared_ptr<const Form> L);

  // Check if _bcs[bc_index] is part of function::FunctionSpace fs
  bool
  check_functionspace_for_bc(std::shared_ptr<const function::FunctionSpace> fs,
                             std::size_t bc_index);

  // Assemble system
  void assemble(la::PETScMatrix* A, la::PETScVector* b,
                const la::PETScVector* x0);

  // Bilinear and linear forms
  std::shared_ptr<const Form> _a, _l;

  // Boundary conditions
  std::vector<std::shared_ptr<const DirichletBC>> _bcs;

  static void cell_wise_assembly(
      std::pair<la::PETScMatrix*, la::PETScVector*>& tensors,
      std::array<UFC*, 2>& ufc, Scratch& data,
      const std::vector<DirichletBC::Map>& boundary_values,
      std::shared_ptr<const mesh::MeshFunction<std::size_t>> cell_domains,
      std::shared_ptr<const mesh::MeshFunction<std::size_t>>
          exterior_facet_domains);

  static void facet_wise_assembly(
      std::pair<la::PETScMatrix*, la::PETScVector*>& tensors,
      std::array<UFC*, 2>& ufc, Scratch& data,
      const std::vector<DirichletBC::Map>& boundary_values,
      std::shared_ptr<const mesh::MeshFunction<std::size_t>> cell_domains,
      std::shared_ptr<const mesh::MeshFunction<std::size_t>>
          exterior_facet_domains,
      std::shared_ptr<const mesh::MeshFunction<std::size_t>>
          interior_facet_domains);

  // Compute exterior facet (and possibly connected cell)
  // contribution
  static void compute_exterior_facet_tensor(
      std::array<std::vector<PetscScalar>, 2>& Ae, std::array<UFC*, 2>& ufc,
      Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
          coordinate_dofs,
      const std::array<bool, 2>& tensor_required_cell,
      const std::array<bool, 2>& tensor_required_facet, const mesh::Cell& cell,
      const mesh::Facet& facet,
      const std::array<const ufc_cell_integral*, 2>& cell_integrals,
      const std::array<const ufc_exterior_facet_integral*, 2>&
          exterior_facet_integrals,
      const bool compute_cell_tensor);

  // Compute interior facet (and possibly connected cell)
  // contribution
  static void compute_interior_facet_tensor(
      std::array<UFC*, 2>& ufc,
      std::array<
          Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
          2>& coordinate_dofs,
      const std::array<bool, 2>& tensor_required_cell,
      const std::array<bool, 2>& tensor_required_facet,
      const std::array<mesh::Cell, 2>& cell,
      const std::array<std::size_t, 2>& local_facet, const bool facet_owner,
      const std::array<const ufc_cell_integral*, 2>& cell_integrals,
      const std::array<const ufc_interior_facet_integral*, 2>&
          interior_facet_integrals,
      const std::array<std::size_t, 2>& matrix_size,
      const std::size_t vector_size,
      const std::array<bool, 2> compute_cell_tensor);

  // Modified matrix insertion for case when rhs has facet integrals
  // and lhs has no facet integrals
  static void matrix_block_add(
      la::PETScMatrix& tensor, std::vector<PetscScalar>& Ae,
      std::vector<PetscScalar>& macro_A,
      const std::array<bool, 2>& add_local_tensor,
      const std::array<std::vector<common::ArrayView<const PetscInt>>, 2>&
          cell_dofs);

  static void apply_bc(PetscScalar* A, PetscScalar* b,
                       const std::vector<DirichletBC::Map>& boundary_values,
                       const common::ArrayView<const PetscInt>& global_dofs0,
                       const common::ArrayView<const PetscInt>& global_dofs1);

  // Return true if cell has an Dirichlet/essential boundary
  // condition applied
  static bool has_bc(const DirichletBC::Map& boundary_values,
                     const common::ArrayView<const PetscInt>& dofs);

  // Return true if element matrix is required
  static bool
  cell_matrix_required(const la::PETScMatrix* A, const void* integral,
                       const std::vector<DirichletBC::Map>& boundary_values,
                       const common::ArrayView<const PetscInt>& dofs);
};
} // namespace fem
} // namespace dolfin
