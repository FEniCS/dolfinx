// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-01-17
// Last changed: 2007-07-22

#ifndef __ASSEMBLE_H
#define __ASSEMBLE_H

#include <ufc.h>
#include <dolfin/MeshFunction.h>

namespace dolfin
{

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
  void assemble(GenericTensor& A, const Form& form, Mesh& mesh);
  
  /// Assemble tensor from given variational form and mesh over a sub domain
  void assemble(GenericTensor& A, const Form& form, Mesh& mesh,
                const SubDomain& sub_domain);

  /// Assemble tensor from given variational form and mesh over sub domains
  void assemble(GenericTensor& A, const Form& form, Mesh& mesh, 
                const MeshFunction<uint>& cell_domains,
                const MeshFunction<uint>& exterior_facet_domains,
                const MeshFunction<uint>& interior_facet_domains);

  /// Assemble scalar from given variational form and mesh
  real assemble(const Form& form, Mesh& mesh);

  /// Assemble scalar from given variational form and mesh over a sub domain
  real assemble(const Form& form, Mesh& mesh,
                const SubDomain& sub_domain);

  /// Assemble scalar from given variational form and mesh over sub domains
  real assemble(const Form& form, Mesh& mesh,
                const MeshFunction<uint>& cell_domains,
                const MeshFunction<uint>& exterior_facet_domains,
                const MeshFunction<uint>& interior_facet_domains);

  /// Assemble tensor from given (UFC) form, mesh, coefficients and sub domains
  void assemble(GenericTensor& A, const ufc::form& form, Mesh& mesh,
                Array<Function*>& coefficients,
                const MeshFunction<uint>* cell_domains,
                const MeshFunction<uint>* exterior_facet_domains,
                const MeshFunction<uint>* interior_facet_domains, bool reset_tensor = true);  

  // FIXME: For testing JIT compiler
  void assemble_test(const ufc::form& form);

}

#endif
