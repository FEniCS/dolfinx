// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-01-17
// Last changed: 2007-05-14

#ifndef __ASSEMBLE_H
#define __ASSEMBLE_H

#include <ufc.h>

#include <dolfin/Array.h>

namespace dolfin
{

  class GenericTensor;
  class Function;
  class Form;
  class Mesh;

  /// These functions provide automated assembly of linear systems,
  /// or more generally, assembly of a sparse tensor from a given
  /// variational form. If you need to assemble a system more than
  /// once, consider using the Assembler class, which may improve
  /// performance by reuse of data structures.

  /// Assemble tensor from given variational form and mesh
  void assemble(GenericTensor& A, const Form& form, Mesh& mesh);
  
  /// Assemble tensor from given variational form and mesh over a sub domain
  //void assemble(GenericTensor& A, const Form& form, Mesh& mesh, const SubDomain& sub_domain);

  /// Assemble scalar from given variational form and mesh
  real assemble(const Form& form, Mesh& mesh);
  
}

#endif
