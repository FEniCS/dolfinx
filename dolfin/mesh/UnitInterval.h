// Copyright (C) 2007 Kristian B. Oelgaard.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-11-23
// Last changed: 2010-10-19
//
// Modified by Anders Logg, 2010.

#ifndef __UNIT_INTERVAL_H
#define __UNIT_INTERVAL_H

#include "Mesh.h"

namespace dolfin
{

  /// A mesh of the unit interval (0, 1) with a given number of cells
  /// (nx) in the axial direction. The total number of intervals will
  /// be nx and the total number of vertices will be (nx + 1).

  class UnitInterval : public Mesh
  {
  public:

    /// Create mesh of unit interval
    UnitInterval(uint nx=1);

  };

}

#endif
