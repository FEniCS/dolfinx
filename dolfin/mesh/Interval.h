// Copyright (C) 2007 Kristian B. Oelgaard.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by N. Lopes, 2008.
//
// First added:  2007-11-23
// Last changed: 2008-10-13

#ifndef __INTERVAL_H
#define __INTERVAL_H

#include "Mesh.h"

namespace dolfin
{

  /// Interval mesh of the 1D line (a,b).
  /// Given the number of cells (nx) in the axial direction,
  /// the total number of intervals will be nx and the
  /// total number of vertices will be (nx + 1).

  class Interval : public Mesh
  {
  public:
    
    Interval(uint nx,double a,double b);

  };
  
}

#endif
