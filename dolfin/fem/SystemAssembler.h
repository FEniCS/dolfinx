// Copyright (C) 2008-2009 Kent-Andre Mardal and Garth N. Wells
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
// Last changed: 2011-01-25

#ifndef __SYSTEM_ASSEMBLER_H
#define __SYSTEM_ASSEMBLER_H

#include <map>
#include <vector>
#include <dolfin/common/types.h>
#include "DirichletBC.h"

namespace dolfin
{

  // Forward declarations
  class GenericMatrix;
  class GenericTensor;
  class GenericVector;
  class Form;
  class Mesh;
  class SubDomain;
  class UFC;
  class Cell;
  class Facet;
  class Function;
  template<typename T> class MeshFunction;

  /// This class provides implements an assembler for systems
  /// of the form Ax = b. It differs from the default DOLFIN
  /// assembler in that it assembles both A and b and the same
  /// time (leading to better performance) and in that it applies
  /// boundary conditions at the time of assembly.

  class SystemAssembler
  {
  public:

    /// Assemble system (A, b)
    static void assemble(GenericMatrix& A,
                         GenericVector& b,
                         const Form& a,
                         const Form& L,
                         bool reset_sparsity=true,
                         bool add_values=false,
                         bool finalize_tensor=true);

    /// Assemble system (A, b) and apply Dirichlet boundary condition
    static void assemble(GenericMatrix& A,
                         GenericVector& b,
                         const Form& a,
                         const Form& L,
                         const DirichletBC& bc,
                         bool reset_sparsity=true,
                         bool add_values=true,
                         bool finalize_tensor=true);

    /// Assemble system (A, b) and apply Dirichlet boundary conditions
    static void assemble(GenericMatrix& A,
                         GenericVector& b,
                         const Form& a,
                         const Form& L,
                         const std::vector<const DirichletBC*>& bcs,
                         bool reset_sparsity=true,
                         bool add_values=false,
                         bool finalize_tensor=true);

    /// Assemble system (A, b) and apply Dirichlet boundary conditions
    static void assemble(GenericMatrix& A,
                         GenericVector& b,
                         const Form& a,
                         const Form& L,
                         const std::vector<const DirichletBC*>& bcs,
                         const MeshFunction<uint>* cell_domains,
                         const MeshFunction<uint>* exterior_facet_domains,
                         const MeshFunction<uint>* interior_facet_domains,
                         const GenericVector* x0,
                         bool reset_sparsity=true,
                         bool add_values=false,
                         bool finalize_tensor=true);

  private:

    class Scratch;

    static void compute_tensor_on_one_interior_facet(const Form& a,
                                                     UFC& ufc,
                                                     const Cell& cell1,
                                                     const Cell& cell2,
                                                     const Facet& facet,
                                                     const MeshFunction<uint>* exterior_facet_domains);

    static void cell_wise_assembly(GenericMatrix& A, GenericVector& b,
                                   const Form& a, const Form& L,
                                   UFC& A_ufc, UFC& b_ufc, Scratch& data,
                                   const DirichletBC::Map& boundary_values,
                                   const MeshFunction<uint>* cell_domains,
                                   const MeshFunction<uint>* exterior_facet_domains);

    static void facet_wise_assembly(GenericMatrix& A, GenericVector& b,
                                    const Form& a, const Form& L,
                                    UFC& A_ufc, UFC& b_ufc, Scratch& data,
                                    const DirichletBC::Map& boundary_values,
                                    const MeshFunction<uint>* cell_domains,
                                    const MeshFunction<uint>* exterior_facet_domains,
                                    const MeshFunction<uint>* interior_facet_domains);

    static void assemble_interior_facet(GenericMatrix& A, GenericVector& b,
                                        UFC& A_ufc, UFC& b_ufc,
                                        const Form& a, const Form& L,
                                        const Cell& cell0, const Cell& cell1,
                                        const Facet& facet,
                                        Scratch& data,
                                        const DirichletBC::Map& boundary_values);

    static void assemble_exterior_facet(GenericMatrix& A, GenericVector& b,
                                        UFC& A_ufc, UFC& b_ufc,
                                        const Form& a,
                                        const Form& L,
                                        const Cell& cell, const Facet& facet,
                                        Scratch& data,
                                        const DirichletBC::Map& boundary_values);

    static void apply_bc(double* A, double* b,
                         const DirichletBC::Map& boundary_values,
                         const std::vector<const std::vector<uint>* >& global_dofs);

    // Class to hold temporary data
    class Scratch
    {
    public:

      Scratch(const Form& a, const Form& L);

      ~Scratch();

      void zero_cell();

      std::vector<double> Ae;
      std::vector<double> be;

    };

  };

}

#endif
