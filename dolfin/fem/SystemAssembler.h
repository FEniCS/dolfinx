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
// Last changed: 2013-03-12

#ifndef __SYSTEM_ASSEMBLER_H
#define __SYSTEM_ASSEMBLER_H

#include <map>
#include <vector>
#include <boost/shared_ptr.hpp>
#include "DirichletBC.h"
#include "AssemblerBase.h"

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

  /// This class provides implements an assembler for systems of the
  /// form Ax = b. It differs from the default DOLFIN assembler in that
  /// it applies boundary conditions at the time of assembly, which
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

    /// Assemble system (A, b) (suitable for use inside a (quasi-)
    /// Newton solver)
    void assemble(GenericMatrix& A, GenericVector& b, const GenericVector& x0);

    /// Assemble vectpr b (suitable for use inside a (quasi-) Newton
    /// solver)
    void assemble(GenericVector& b, const GenericVector& x0);

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
                                                     const Cell& cell1,
                                                     const Cell& cell2,
                                                     const Facet& facet,
                                                     const MeshFunction<std::size_t>* exterior_facet_domains);

    static void cell_wise_assembly(GenericMatrix* A, GenericVector* b,
                                   const Form& a, const Form& L,
                                   UFC& A_ufc, UFC& b_ufc, Scratch& data,
                                   const DirichletBC::Map& boundary_values,
                                   const MeshFunction<std::size_t>* cell_domains,
                                   const MeshFunction<std::size_t>* exterior_facet_domains);

    static void facet_wise_assembly(GenericMatrix* A, GenericVector* b,
                                    const Form& a, const Form& L,
                                    UFC& A_ufc, UFC& b_ufc, Scratch& data,
                                    const DirichletBC::Map& boundary_values,
                                    const MeshFunction<std::size_t>* cell_domains,
                                    const MeshFunction<std::size_t>* exterior_facet_domains,
                                    const MeshFunction<std::size_t>* interior_facet_domains);

    static void assemble_interior_facet(GenericMatrix* A, GenericVector* b,
                                        UFC& A_ufc, UFC& b_ufc,
                                        const Form& a, const Form& L,
                                        const Cell& cell0, const Cell& cell1,
                                        const Facet& facet,
                                        Scratch& data,
                                        const DirichletBC::Map& boundary_values);

    static void assemble_exterior_facet(GenericMatrix* A, GenericVector* b,
                                        UFC& A_ufc, UFC& b_ufc,
                                        const Form& a,
                                        const Form& L,
                                        const Cell& cell, const Facet& facet,
                                        Scratch& data,
                                        const DirichletBC::Map& boundary_values);

    static void apply_bc(double* A, double* b,
                         const DirichletBC::Map& boundary_values,
                         const std::vector<const std::vector<dolfin::la_index>* >& global_dofs);

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
