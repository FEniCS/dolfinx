// Copyright (C) 2008-2015 Kent-Andre Mardal and Garth N. Wells
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Anders Logg 2008-2011

#ifndef __SYSTEM_ASSEMBLER_H
#define __SYSTEM_ASSEMBLER_H

#include <array>
#include <map>
#include <memory>
#include <vector>

#include "DirichletBC.h"
#include "AssemblerBase.h"

namespace ufc
{
  class cell;
  class cell_integral;
  class exterior_facet_integral;
  class interior_facet_integral;
}

namespace dolfin
{

  // Forward declarations
  template<typename T> class ArrayView;
  class Cell;
  class Facet;
  class Form;
  class GenericDofMap;
  class GenericMatrix;
  class GenericVector;
  template<typename T> class MeshFunction;
  class UFC;

  /// This class provides an assembler for systems of the form Ax =
  /// b. It differs from the default DOLFIN assembler in that it
  /// applies boundary conditions at the time of assembly, which
  /// preserves any symmetries in A.

  class SystemAssembler : public AssemblerBase
  {
  public:

    /// Constructor
    SystemAssembler(std::shared_ptr<const Form> a,
                    std::shared_ptr<const Form> L,
                    std::vector<std::shared_ptr<const DirichletBC>> bcs);

    /// Assemble system (A, b)
    void assemble(GenericMatrix& A, GenericVector& b);

    /// Assemble matrix A
    void assemble(GenericMatrix& A);

    /// Assemble vector b
    void assemble(GenericVector& b);

    /// Assemble system (A, b) for (negative) increment dx, where x =
    /// x0 - dx is solution to system a == -L subject to bcs.
    /// Suitable for use inside a (quasi-)Newton solver.
    void assemble(GenericMatrix& A, GenericVector& b, const GenericVector& x0);

    /// Assemble rhs vector b for (negative) increment dx, where x =
    /// x0 - dx is solution to system a == -L subject to bcs.
    /// Suitable for use inside a (quasi-)Newton solver.
    void assemble(GenericVector& b, const GenericVector& x0);

  private:

    // Class to hold temporary data
    class Scratch
    {
    public:
      Scratch(const Form& a, const Form& L);
      ~Scratch();
      std::array<std::vector<double>, 2> Ae;
    };

    // Check form arity
    static void check_arity(std::shared_ptr<const Form> a,
                            std::shared_ptr<const Form> L);

    // Check if _bcs[i] is part of FunctionSpace fs
    bool check_functionspace_for_bc
      (std::shared_ptr<const FunctionSpace> fs, std::size_t i);

    // Assemble system
    void assemble(GenericMatrix* A, GenericVector* b,
                  const GenericVector* x0);

    // Bilinear and linear forms
    std::shared_ptr<const Form> _a, _l;

    // Boundary conditions
    std::vector<std::shared_ptr<const DirichletBC>> _bcs;

    static void cell_wise_assembly(
      std::array<GenericTensor*, 2>& tensors,
      std::array<UFC*, 2>& ufc,
      Scratch& data,
      const std::vector<DirichletBC::Map>& boundary_values,
      std::shared_ptr<const MeshFunction<std::size_t>> cell_domains,
      std::shared_ptr<const MeshFunction<std::size_t>> exterior_facet_domains);

    static void facet_wise_assembly(
      std::array<GenericTensor*, 2>& tensors,
      std::array<UFC*, 2>& ufc,
      Scratch& data,
      const std::vector<DirichletBC::Map>& boundary_values,
      std::shared_ptr<const MeshFunction<std::size_t>> cell_domains,
      std::shared_ptr<const MeshFunction<std::size_t>> exterior_facet_domains,
      std::shared_ptr<const MeshFunction<std::size_t>> interior_facet_domains);

    // Compute exterior facet (and possibly connected cell)
    // contribution
    static void compute_exterior_facet_tensor(
      std::array<std::vector<double>, 2>& Ae,
      std::array<UFC*, 2>& ufc,
      ufc::cell& ufc_cell,
      std::vector<double>& coordinate_dofs,
      const std::array<bool, 2>& tensor_required_cell,
      const std::array<bool, 2>& tensor_required_facet,
      const Cell& cell,
      const Facet& facet,
      const std::array<const ufc::cell_integral*, 2>& cell_integrals,
      const std::array<const ufc::exterior_facet_integral*, 2>& exterior_facet_integrals,
      const bool compute_cell_tensor);

    // Compute interior facet (and possibly connected cell)
    // contribution
    static void compute_interior_facet_tensor(
      std::array<UFC*, 2>& ufc,
      std::array<ufc::cell, 2>& ufc_cell,
      std::array<std::vector<double>, 2>& coordinate_dofs,
      const std::array<bool, 2>& tensor_required_cell,
      const std::array<bool, 2>& tensor_required_facet,
      const std::array<Cell, 2>& cell,
      const std::array<std::size_t, 2>& local_facet,
      const bool facet_owner,
      const std::array<const ufc::cell_integral*, 2>& cell_integrals,
      const std::array<const ufc::interior_facet_integral*, 2>& interior_facet_integrals,
      const std::array<std::size_t, 2>& matrix_size,
      const std::size_t vector_size,
      const std::array<bool, 2> compute_cell_tensor);

    // Modified matrix insertion for case when rhs has facet integrals
    // and lhs has no facet integrals
    static void matrix_block_add(
      GenericTensor& tensor,
      std::vector<double>& Ae,
      std::vector<double>& macro_A,
      const std::array<bool, 2>& add_local_tensor,
      const std::array<std::vector<ArrayView<const la_index>>, 2>& cell_dofs);

    static void apply_bc(double* A, double* b,
                         const std::vector<DirichletBC::Map>& boundary_values,
                         const ArrayView<const dolfin::la_index>& global_dofs0,
                         const ArrayView<const dolfin::la_index>& global_dofs1);

    // Return true if cell has an Dirichlet/essential boundary
    // condition applied
    static bool has_bc(const DirichletBC::Map& boundary_values,
                       const ArrayView<const dolfin::la_index>& dofs);

    // Return true if element matrix is required
    static bool
      cell_matrix_required(const GenericTensor* A,
                           const void* integral,
                           const std::vector<DirichletBC::Map>& boundary_values,
                           const ArrayView<const dolfin::la_index>& dofs);

  };

}

#endif
