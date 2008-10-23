// Copyright (C) 2007-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2008.
//
// First added:  2007-01-17
// Last changed: 2008-08-21

#ifndef __ASSEMBLE_H
#define __ASSEMBLE_H

#include <vector>
#include <dolfin/common/types.h>

// Forward declaration
namespace ufc 
{
  class form; 
}

namespace dolfin
{

  // Forward declarations
  class DirichletBC;
  class GenericMatrix;
  class GenericTensor;
  class GenericVector;
  class Function;
  class Form;
  class SubDomain;
  class Mesh;
  template<class T> class MeshFunction;

  /// These functions provide automated assembly of linear systems,
  /// or more generally, assembly of a sparse tensor from a given
  /// variational form. If you need to assemble a system more than
  /// once, consider using the Assembler class, which may improve
  /// performance by reuse of data structures.

  /// Assemble tensor from given variational form and mesh
  void assemble(GenericTensor& A, Form& form, Mesh& mesh,
                bool reset_tensor=true);
  
  /// Assemble system (A, b) and apply Dirichlet boundary condition from 
  /// given variational forms
  void assemble(GenericMatrix& A, Form& a, GenericVector& b, Form& L, 
                DirichletBC& bc, Mesh& mesh, bool reset_tensor=true);

  /// Assemble system (A, b) and apply Dirichlet boundary conditions from 
  /// given variational forms
  void assemble(GenericMatrix& A, Form& a, GenericVector& b, Form& L, 
                std::vector<DirichletBC*>& bcs, Mesh& mesh, bool reset_tensor=true);

  /// Assemble tensor from given variational form and mesh over a sub domain
  void assemble(GenericTensor& A, Form& form, Mesh& mesh, const SubDomain& sub_domain,
                bool reset_tensor=true);

  /// Assemble tensor from given variational form and mesh over sub domains
  void assemble(GenericTensor& A, Form& form, Mesh& mesh, 
                const MeshFunction<uint>& cell_domains,
                const MeshFunction<uint>& exterior_facet_domains,
                const MeshFunction<uint>& interior_facet_domains, 
                bool reset_tensor=true);

  /// Assemble scalar from given variational form and mesh
  double assemble(Form& form, Mesh& mesh,
                bool reset_tensor=true);

  /// Assemble scalar from given variational form and mesh over a sub domain
  double assemble(Form& form, Mesh& mesh, const SubDomain& sub_domain,
                bool reset_tensor=true);

  /// Assemble scalar from given variational form and mesh over sub domains
  double assemble(Form& form, Mesh& mesh,
                const MeshFunction<uint>& cell_domains,
                const MeshFunction<uint>& exterior_facet_domains,
                const MeshFunction<uint>& interior_facet_domains,
                bool reset_tensor=true);

  /// Assemble tensor from given (UFC) form, mesh, coefficients and sub domains
  void assemble(GenericTensor& A, const Form& form, Mesh& mesh,
                std::vector<Function*>& coefficients,
                const MeshFunction<uint>* cell_domains,
                const MeshFunction<uint>* exterior_facet_domains,
                const MeshFunction<uint>* interior_facet_domains,
                bool reset_tensor = true);

  /// Assemble tensor from given (UFC) form, mesh, coefficients and sub domains
  void assemble_system(GenericMatrix& A, const Form& A_form, 
                       const std::vector<Function*>& A_coefficients,
                       GenericVector& b, const Form& b_form, 
                       const std::vector<Function*>& b_coefficients,
                       const GenericVector* x0,
                       Mesh& mesh, 
                       std::vector<DirichletBC*>& bcs, const MeshFunction<uint>* cell_domains, 
                       const MeshFunction<uint>* exterior_facet_domains,
                       const MeshFunction<uint>* interior_facet_domains,
                       bool reset_tensors=true);

}

#endif
