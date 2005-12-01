// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-05-02
// Last changed: 2005-11-30

#ifndef __P1_TET_H
#define __P1_TET_H

#include <dolfin/P1TetTemplate.h>
#include <dolfin/FiniteElement.h>

namespace dolfin
{

  /// This class represents the standard scalar- or vector-valued
  /// linear finite element on a tetrahedron. Note that finite
  /// elements are normally generated automatically by FFC, but this
  /// class might be useful for simple computations with standard
  /// linear elements.

  typedef P1TetTemplate::LinearForm::TestElement P1Tet;

}

#endif
