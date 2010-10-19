// Copyright (C) 2010 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2010-10-19
// Last changed: 2010-10-19

#ifndef __UNIT_TRIANGLE_H
#define __UNIT_TRIANGLE_H

#include "Mesh.h"

namespace dolfin
{

  /// A mesh consisting of a single triangle with vertices at
  ///
  ///   (0, 0)
  ///   (1, 0)
  ///   (0, 1)
  ///
  /// This class is useful for testing.

  class UnitTriangle : public Mesh
  {
  public:

    /// Create mesh of unit triangle
    UnitTriangle();

  };

}

#endif
