// Copyright (C) 2007-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2008.
//
// First added:  2007-01-17
// Last changed: 2008-12-18
//
// This file duplicates the Assembler::assemble* functions in
// namespace dolfin, and adds special versions returning the value
// directly for scalars. For documentation, refer to Assemble.h.

#ifndef __ASSEMBLE_H
#define __ASSEMBLE_H

#include <dolfin/log/log.h>
#include "Assembler.h"

namespace dolfin
{
  
  //--- Copies of assembly functions in Assembler.h ---
  
  /// Assemble tensor
  void assemble(GenericTensor& A,
                const Form& a,
                bool reset_tensor=true)
  {
    Assembler::assemble(A, a, reset_tensor);
  }

  /// Assemble tensor on sub domain
  void assemble(GenericTensor& A,
                const Form& a,
                const SubDomain& sub_domain,
                bool reset_tensor=true)
  {
    Assembler::assemble(A, a, sub_domain, reset_tensor);
  }

  /// Assemble tensor on sub domains
  void assemble(GenericTensor& A,
                const Form& a,
                const MeshFunction<uint>* cell_domains,
                const MeshFunction<uint>* exterior_facet_domains,
                const MeshFunction<uint>* interior_facet_domains,
                bool reset_tensor=true)
  {
    Assembler::assemble(A, a, cell_domains, exterior_facet_domains, interior_facet_domains);
  }

  /// Assemble system (A, b) and apply Dirichlet boundary condition
  void assemble_system(GenericMatrix& A,
                       GenericVector& b,
                       const Form& a,
                       const Form& L,
                       const DirichletBC& bc,
                       bool reset_tensors=true)
  {
    Assembler::assemble_system(A, b, a, L, bc, reset_tensors);
  }
 
  /// Assemble system (A, b) and apply Dirichlet boundary conditions
  void assemble_system(GenericMatrix& A,
                       GenericVector& b,
                       const Form& a,
                       const Form& L, 
                       std::vector<const DirichletBC*>& bcs,
                       bool reset_tensors=true)
  {
    Assembler::assemble_system(A, b, a, L, bcs, reset_tensors);
  }

  /// Assemble system (A, b) on sub domains and apply Dirichlet boundary conditions
  void assemble_system(GenericMatrix& A,
                       GenericVector& b,
                       const Form& a,
                       const Form& L,
                       std::vector<const DirichletBC*>& bcs,
                       const MeshFunction<uint>* cell_domains,
                       const MeshFunction<uint>* exterior_facet_domains,
                       const MeshFunction<uint>* interior_facet_domains,
                       const GenericVector* x0,
                       bool reset_tensors=true)
  {
    Assembler::assemble_system(A, b, a, L, bcs, 
                               cell_domains, exterior_facet_domains, interior_facet_domains,
                               x0, reset_tensors);
  }

  //--- Specialized versions for scalars ---

  /// Assemble scalar
  double assemble(const Form& a,
                  bool reset_tensor=true)
  {
    if (a.rank() != 0) error("Unable to assemble, form is not scalar.");
    Scalar s;
    Assembler::assemble(s, a, reset_tensor);
    return s;
  }

  /// Assemble scalar on sub domain
  double assemble(const Form& a,
                  const SubDomain& sub_domain,
                  bool reset_tensor=true)
  {
    if (a.rank() != 0) error("Unable to assemble, form is not scalar.");
    Scalar s;
    Assembler::assemble(s, a, sub_domain, reset_tensor);
    return s;
  }

  /// Assemble scalar on sub domains
  double assemble(const Form& a,
                  const MeshFunction<uint>* cell_domains,
                  const MeshFunction<uint>* exterior_facet_domains,
                  const MeshFunction<uint>* interior_facet_domains,
                  bool reset_tensor=true)
  {
    if (a.rank() != 0) error("Unable to assemble, form is not scalar.");
    Scalar s;
    Assembler::assemble(s, a, cell_domains, exterior_facet_domains, interior_facet_domains, reset_tensor);
    return s;
  }

}

#endif
