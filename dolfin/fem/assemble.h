// Copyright (C) 2007-2011 Anders Logg
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
// Modified by Garth N. Wells, 2008, 2009.
// Modified by Johan Hake, 2009.
//
// First added:  2007-01-17
// Last changed: 2011-09-29
//
// This file duplicates the Assembler::assemble* and SystemAssembler::assemble*
// functions in namespace dolfin, and adds special versions returning the value
// directly for scalars. For documentation, refer to Assemble.h and
// SystemAssemble.h

#ifndef __ASSEMBLE_H
#define __ASSEMBLE_H

#include <vector>
#include "DirichletBC.h"

namespace dolfin
{

  class DirichletBC;
  class Form;
  class GenericTensor;
  class GenericMatrix;
  class GenericVector;
  template<typename T> class MeshFunction;
  class SubDomain;

  //--- Copies of assembly functions in Assembler.h ---

  /// Assemble tensor
  void assemble(GenericTensor& A,
                const Form& a,
                bool reset_sparsity=true,
                bool add_values=false,
                bool finalize_tensor=true);

  /// Assemble tensor on sub domain
  void assemble(GenericTensor& A,
                const Form& a,
                const SubDomain& sub_domain,
                bool reset_sparsity=true,
                bool add_values=false,
                bool finalize_tensor=true);

  /// Assemble tensor on sub domains
  void assemble(GenericTensor& A,
                const Form& a,
                const MeshFunction<unsigned int>* cell_domains,
                const MeshFunction<unsigned int>* exterior_facet_domains,
                const MeshFunction<unsigned int>* interior_facet_domains,
                bool reset_sparsity=true,
                bool add_values=false,
                bool finalize_tensor=true);

  /// Assemble system (A, b)
  void assemble_system(GenericMatrix& A,
                       GenericVector& b,
                       const Form& a,
                       const Form& L,
                       bool reset_sparsities=true,
                       bool add_values=false);

  /// Assemble system (A, b) and apply Dirichlet boundary condition
  void assemble_system(GenericMatrix& A,
                       GenericVector& b,
                       const Form& a,
                       const Form& L,
                       const DirichletBC& bc,
                       bool reset_sparsities=true,
                       bool add_values=false);

  /// Assemble system (A, b) and apply Dirichlet boundary conditions
  void assemble_system(GenericMatrix& A,
                       GenericVector& b,
                       const Form& a,
                       const Form& L,
                       const std::vector<const DirichletBC*>& bcs,
                       bool reset_sparsities=true,
                       bool add_values=false);

  /// Assemble system (A, b) on sub domains and apply Dirichlet boundary conditions
  void assemble_system(GenericMatrix& A,
                       GenericVector& b,
                       const Form& a,
                       const Form& L,
                       const std::vector<const DirichletBC*>& bcs,
                       const MeshFunction<unsigned int>* cell_domains,
                       const MeshFunction<unsigned int>* exterior_facet_domains,
                       const MeshFunction<unsigned int>* interior_facet_domains,
                       const GenericVector* x0,
                       bool reset_sparsities=true,
                       bool add_values=false);

  //--- Specialized versions for scalars ---

  /// Assemble scalar
  double assemble(const Form& a,
                  bool reset_sparsity=true,
                  bool add_values=false,
                  bool finalize_tensor=true);

  /// Assemble scalar on sub domain
  double assemble(const Form& a,
                  const SubDomain& sub_domain,
                  bool reset_sparsity=true,
                  bool add_values=false,
                  bool finalize_tensor=true);

  /// Assemble scalar on sub domains
  double assemble(const Form& a,
                  const MeshFunction<unsigned int>* cell_domains,
                  const MeshFunction<unsigned int>* exterior_facet_domains,
                  const MeshFunction<unsigned int>* interior_facet_domains,
                  bool reset_sparsity=true,
                  bool add_values=false,
                  bool finalize_tensor=true);

}

#endif
