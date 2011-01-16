// Copyright (C) 2007-2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2008, 2009.
// Modified by Johan Hake, 2009.
//
// First added:  2007-01-17
// Last changed: 2009-09-02
//
// This file duplicates the Assembler::assemble* and SystemAssembler::assemble*
// functions in namespace dolfin, and adds special versions returning the value
// directly for scalars. For documentation, refer to Assemble.h and
// SystemAssemble.h

#ifndef __ASSEMBLE_H
#define __ASSEMBLE_H

#include <vector>
//#include <dolfin/mesh/MeshFunction.h>
#include "DirichletBC.h"

namespace dolfin
{

  class DirichletBC;
  class Form;
  class GenericTensor;
  class GenericMatrix;
  class GenericVector;
  template<class T> class MeshFunction;
  class SubDomain;

  //--- Copies of assembly functions in Assembler.h ---

  /// Assemble tensor
  void assemble(GenericTensor& A,
                const Form& a,
                bool reset_sparsity=true,
                bool add_values=false);

  /// Assemble tensor on sub domain
  void assemble(GenericTensor& A,
                const Form& a,
                const SubDomain& sub_domain,
                bool reset_sparsity=true,
                bool add_values=false);

  /// Assemble tensor on sub domains
  void assemble(GenericTensor& A,
                const Form& a,
                const MeshFunction<uint>* cell_domains,
                const MeshFunction<uint>* exterior_facet_domains,
                const MeshFunction<uint>* interior_facet_domains,
                bool reset_sparsity=true,
                bool add_values=false);

  /// Assemble system (A, b)
  void assemble_system(GenericMatrix& A,
                       GenericVector& b,
                       const Form& a,
                       const Form& L,
                       bool reset_sparsitys=true,
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
                       const MeshFunction<uint>* cell_domains,
                       const MeshFunction<uint>* exterior_facet_domains,
                       const MeshFunction<uint>* interior_facet_domains,
                       const GenericVector* x0,
                       bool reset_sparsitys=true,
                       bool add_values=false);

  //--- Specialized versions for scalars ---

  /// Assemble scalar
  double assemble(const Form& a,
                  bool reset_sparsity=true,
                  bool add_values=false);

  /// Assemble scalar on sub domain
  double assemble(const Form& a,
                  const SubDomain& sub_domain,
                  bool reset_sparsity=true,
                  bool add_values=false);

  /// Assemble scalar on sub domains
  double assemble(const Form& a,
                  const MeshFunction<uint>* cell_domains,
                  const MeshFunction<uint>* exterior_facet_domains,
                  const MeshFunction<uint>* interior_facet_domains,
                  bool reset_sparsity=true,
                  bool add_values=false);

}

#endif
