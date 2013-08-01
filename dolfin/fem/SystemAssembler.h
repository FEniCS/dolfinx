// Copyright (C) 2008-2013 Kent-Andre Mardal and Garth N. Wells
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
//
// First added:  2009-06-22
// Last changed: 2013-04-23

#ifndef __SYSTEM_ASSEMBLER_H
#define __SYSTEM_ASSEMBLER_H

#include <map>
#include <vector>
#include <boost/array.hpp>
#include <boost/shared_ptr.hpp>
#include "DirichletBC.h"
#include "AssemblerBase.h"

namespace ufc
{
  class cell_integral;
}

namespace dolfin
{

  // Forward declarations
  class Cell;
  class Facet;
  class Form;
  class GenericMatrix;
  class GenericVector;
  template<typename T> class MeshFunction;
  class UFC;

  /// This class provides an assembler for systems of the form
  /// Ax = b. It differs from the default DOLFIN assembler in that it
  /// applies boundary conditions at the time of assembly, which
  /// preserves any symmetries in A.

  class SystemAssembler : public AssemblerBase
  {
  public:

    /// Constructor
    SystemAssembler(const Form& a, const Form& L);

    /// Constructor
    SystemAssembler(const Form& a, const Form& L, const DirichletBC& bc);

    /// Constructor
    SystemAssembler(const Form& a, const Form& L,
                    const std::vector<const DirichletBC*> bcs);

    /// Constructor
    SystemAssembler(boost::shared_ptr<const Form> a,
                    boost::shared_ptr<const Form> L);

    /// Constructor
    SystemAssembler(boost::shared_ptr<const Form> a,
                    boost::shared_ptr<const Form> L,
                    const DirichletBC& bc);

    /// Constructor
    SystemAssembler(boost::shared_ptr<const Form> a,
                    boost::shared_ptr<const Form> L,
                    const std::vector<const DirichletBC*> bcs);

    /// Assemble system (A, b)
    void assemble(GenericMatrix& A, GenericVector& b);

    /// Assemble matrix A
    void assemble(GenericMatrix& A);

    /// Assemble vector b
    void assemble(GenericVector& b);

    /// Assemble system (A, b) for (negative) increment dx, where
    /// x = x0 - dx is solution to system a == -L subject to bcs.
    /// Suitable for use inside a (quasi-)Newton solver.
    void assemble(GenericMatrix& A, GenericVector& b, const GenericVector& x0);

    /// Assemble rhs vector b for (negative) increment dx, where
    /// x = x0 - dx is solution to system a == -L subject to bcs.
    /// Suitable for use inside a (quasi-)Newton solver.
    void assemble(GenericVector& b, const GenericVector& x0);

    /// Rescale Dirichlet (essential) boundary condition entries in
    /// assembled system. Should be false if the RHS is assembled
    /// independently of the LHS.
    bool rescale;

  private:

    // Check form arity
    static void check_arity(boost::shared_ptr<const Form> a,
                            boost::shared_ptr<const Form> L);

    // Assemble system
    void assemble(GenericMatrix* A, GenericVector* b,
                  const GenericVector* x0);

    // Bilinear and linear forms
    boost::shared_ptr<const Form> _a, _L;

    // Boundary conditions
    std::vector<const DirichletBC*> _bcs;

    class Scratch;

    static void compute_tensor_on_one_interior_facet(const Form& a,
                                                     UFC& ufc,
                                                     const Cell& cell0,
                                                     const Cell& cell1,
                                                     const Facet& facet,
                      const MeshFunction<std::size_t>* exterior_facet_domains);

    static void cell_wise_assembly(boost::array<GenericTensor*, 2>& tensors,
                                   boost::array<UFC*, 2>& ufc,
                                   Scratch& data,
                                   const DirichletBC::Map& boundary_values,
                       const MeshFunction<std::size_t>* cell_domains,
                       const MeshFunction<std::size_t>* exterior_facet_domains,
                       const bool rescale );

    static void facet_wise_assembly(boost::array<GenericTensor*, 2>& tensors,
                                    boost::array<UFC*, 2>& ufc,
                                    Scratch& data,
                                    const DirichletBC::Map& boundary_values,
                       const MeshFunction<std::size_t>* cell_domains,
                       const MeshFunction<std::size_t>* exterior_facet_domains,
                       const MeshFunction<std::size_t>* interior_facet_domains,
                                    const bool rescale);

    static void apply_bc(double* A, double* b,
                         const DirichletBC::Map& boundary_values,
                         const std::vector<dolfin::la_index>& global_dofs0,
                         const std::vector<dolfin::la_index>& global_dofs1,
                         const bool rescale);

    // Return true if cell has an Dirichlet/essential boundary
    // condition applied
    static bool has_bc(const DirichletBC::Map& boundary_values,
                       const std::vector<dolfin::la_index>& dofs);

    // Return true if element matrix is required
    static bool cell_matrix_required(const GenericTensor* A,
                                     const ufc::cell_integral* integral,
                                     const DirichletBC::Map& boundary_values,
                                     const std::vector<dolfin::la_index>& dofs);

    // Class to hold temporary data
    class Scratch
    {
    public:

      Scratch(const Form& a, const Form& L);

      ~Scratch();

      void zero_cell();

      boost::array<std::vector<double>, 2> Ae;

    };

  };

}

#endif
