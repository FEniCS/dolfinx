// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2007-01-17
// Last changed: 2007-01-17

#ifndef __ASSEMBLE_H
#define __ASSEMBLE_H

#include <ufc.h>

namespace dolfin
{

  class GenericTensor;
  class Mesh;

  /// These functions provide automated assembly of linear systems,
  /// or more generally, assembly of a sparse tensor from a given
  /// variational form. If you need to assemble a system more than
  /// once, consider using the Assembler class, which may improve
  /// performance by reuse of data structures.

  /// Assemble tensor from given variational form and mesh
  void assemble(GenericTensor& A, const ufc::form& form, Mesh& mesh);  

}

#endif
