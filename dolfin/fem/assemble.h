// Copyright (C) 2007-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-01-17
// Last changed: 2008-07-22

#ifndef __ASSEMBLE_H
#define __ASSEMBLE_H

#include <ufc.h>
#include <dolfin/mesh/MeshFunction.h>
#include "DirichletBC.h"

namespace dolfin
{

  class DofMapSet;
  class GenericTensor;
  class Function;
  class Form;
  class SubDomain;
  class Mesh;

  /// These functions provide automated assembly of linear systems,
  /// or more generally, assembly of a sparse tensor from a given
  /// variational form. If you need to assemble a system more than
  /// once, consider using the Assembler class, which may improve
  /// performance by reuse of data structures.

  /// Assemble tensor from given variational form and mesh
  void assemble(GenericTensor& A, Form& form, Mesh& mesh,
                bool reset_tensor=true);
  
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
  real assemble(Form& form, Mesh& mesh,
                bool reset_tensor=true);

  /// Assemble scalar from given variational form and mesh over a sub domain
  real assemble(Form& form, Mesh& mesh, const SubDomain& sub_domain,
                bool reset_tensor=true);

  /// Assemble scalar from given variational form and mesh over sub domains
  real assemble(Form& form, Mesh& mesh,
                const MeshFunction<uint>& cell_domains,
                const MeshFunction<uint>& exterior_facet_domains,
                const MeshFunction<uint>& interior_facet_domains,
                bool reset_tensor=true);

  /// Assemble tensor from given (UFC) form, mesh, coefficients and sub domains
  void assemble(GenericTensor& A, const ufc::form& form, Mesh& mesh,
                Array<Function*>& coefficients,
                DofMapSet& dof_map_set,
                const MeshFunction<uint>* cell_domains,
                const MeshFunction<uint>* exterior_facet_domains,
                const MeshFunction<uint>* interior_facet_domains,
                bool reset_tensor = true);

  /// Assemble tensor from given (UFC) form, mesh, coefficients and sub domains
  void assemble_system(GenericTensor& A, const ufc::form& A_form, 
                       const Array<Function*>& A_coefficients, const DofMapSet& A_dof_map_set,
                       GenericTensor& b, const ufc::form& b_form, 
                       const Array<Function*>& b_coefficients, const DofMapSet& b_dof_map_set,
                       Mesh& mesh, 
                       DirichletBC& bc, const MeshFunction<uint>* cell_domains, 
                       const MeshFunction<uint>* exterior_facet_domains,
                       const MeshFunction<uint>* interior_facet_domains,
                       bool reset_tensors=true);

}

#endif
