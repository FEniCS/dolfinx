// Copyright (C) 2005-2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Nuno Lopes 2008
//
// First added:  2005-12-02
// Last changed: 2006-08-19

#ifndef __UNIT_SPHERE_H
#define __UNIT_SPHERE_H

#include "Mesh.h"

namespace dolfin
{

  /// Triangular mesh of the 3D unit SPHERE or variation. 
  /// Given the number of cells (nx, ny, nz) in each direction,
  /// the total number of tetrahedra will be 6*nx*ny*nz and the
  /// total number of vertices will be (nx + 1)*(ny + 1)*(nz + 1).
  
 
  class UnitSphere : public Mesh
  {
  public:
  
    UnitSphere(uint nx);

  private:
    real transformx(real x,real y,real z);
    real transformy(real x,real y,real z);
    real transformz(real x,real y,real z);
    real max(real x,real y,real z); 
  };
  
}

#endif
