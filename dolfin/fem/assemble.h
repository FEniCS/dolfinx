// Copyright (C) 2007-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2008.
//
// First added:  2007-01-17
// Last changed: 2008-10-24

#ifndef __ASSEMBLE_H
#define __ASSEMBLE_H

#include "Assembler.h"

namespace dolfin
{
  /// Assemble tensor from given variational form
  void assemble(GenericTensor& A, Form& form, bool reset_tensor=true)
  { Assembler::assemble(A, form, reset_tensor); }
  
  /// Assemble system (A, b) and apply Dirichlet boundary condition from 
  /// given variational forms
  void assemble(GenericMatrix& A, Form& a, GenericVector& b, Form& L, 
                DirichletBC& bc, bool reset_tensor=true)
  { Assembler::assemble(A, a, b, L, bc, reset_tensor); }
  
  /// Assemble system (A, b) and apply Dirichlet boundary conditions from 
  /// given variational forms
  void assemble(GenericMatrix& A, Form& a, GenericVector& b, Form& L, 
                std::vector<DirichletBC*>& bcs, bool reset_tensor=true)
  { Assembler::assemble(A, a, b, L, bcs, reset_tensor); }
  
  /// Assemble tensor from given variational form over a sub domain
  void assemble(GenericTensor& A, Form& form, const SubDomain& sub_domain,
                bool reset_tensor=true)
  { Assembler::assemble(A, form, sub_domain, reset_tensor); }
  
  /// Assemble tensor from given variational form over a sub domain
  //void assemble(GenericTensor& A, Form& form,
  //              const MeshFunction<uint>& domains, uint domain, bool reset_tensor = true);
  
  /// Assemble tensor from given variational form over sub domains
  void assemble(GenericTensor& A, Form& form,
                       const MeshFunction<uint>& cell_domains,
                       const MeshFunction<uint>& exterior_facet_domains,
                       const MeshFunction<uint>& interior_facet_domains,
                       bool reset_tensor=true)
  { Assembler::assemble(A, form, cell_domains, exterior_facet_domains, interior_facet_domains, reset_tensor); }
  
  /// Assemble scalar from given variational form
  double assemble(Form& form, bool reset_tensor=true)
  { return Assembler::assemble(form, reset_tensor); }
  
  /// Assemble scalar from given variational form over a sub domain
  double assemble(Form& form, const SubDomain& sub_domain, bool reset_tensor)
  { return Assembler::assemble(form, sub_domain, reset_tensor); }
  
  /// Assemble scalar from given variational form over sub domains
  double assemble(Form& form,
                  const MeshFunction<uint>& cell_domains,
                  const MeshFunction<uint>& exterior_facet_domains,
                  const MeshFunction<uint>& interior_facet_domains,
                  bool reset_tensor)
  { return Assembler::assemble(form, cell_domains, exterior_facet_domains, interior_facet_domains, reset_tensor); }
  
  /// Assemble tensor from given (UFC) form, coefficients and sub domains.
  /// This is the main assembly function in DOLFIN. All other assembly functions
  /// end up calling this function.
  ///
  /// The MeshFunction arguments can be used to specify assembly over subdomains
  /// of the mesh cells, exterior facets and interior facets. Either a null pointer
  /// or an empty MeshFunction may be used to specify that the tensor should be
  /// assembled over the entire set of cells or facets.
  void assemble(GenericTensor& A, const Form& form,
                       const std::vector<Function*>& coefficients,
                       const MeshFunction<uint>* cell_domains,
                       const MeshFunction<uint>* exterior_facet_domains,
                       const MeshFunction<uint>* interior_facet_domains,
                       bool reset_tensor = true)
  { Assembler::assemble(A, form, coefficients, cell_domains, exterior_facet_domains, interior_facet_domains, reset_tensor); }
  
  /// Assemble linear system Ax = b and enforce Dirichlet conditions.  
  //  Notice that the Dirichlet conditions are enforced in a symmetric way.  
  void assemble_system(GenericMatrix& A, const Form& A_form, 
                       const std::vector<Function*>& A_coefficients,
                       GenericVector& b, const Form& b_form, 
                       const std::vector<Function*>& b_coefficients,
                       const GenericVector* x0,
                       std::vector<DirichletBC*> bcs, const MeshFunction<uint>* cell_domains, 
                       const MeshFunction<uint>* exterior_facet_domains,
                       const MeshFunction<uint>* interior_facet_domains,
                       bool reset_tensors=true)
  { Assembler::assemble_system(A,
                               A_form,
                               A_coefficients,
                               b,
                               b_form,
                               b_coefficients,
                               x0,
                               bcs,
                               cell_domains,
                               exterior_facet_domains,
                               interior_facet_domains,
                               reset_tensors); }

}

#endif
