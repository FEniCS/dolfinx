// Copyright (C) 2008 Nuno Lopes.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-07-15
// Last changed: 2008-07-15

#ifndef __UNIT_SPHERE_H
#define __UNIT_SPHERE_H

#include "Mesh.h"

namespace dolfin
{

  /// Triangular mesh of the 3D unit sphere.
  ///
  /// Given the number of cells (nx, ny, nz) in each direction,
  /// the total number of tetrahedra will be 6*nx*ny*nz and the
  /// total number of vertices will be (nx + 1)*(ny + 1)*(nz + 1).
  
 
  class UnitSphere : public Mesh
  {
  public:
  
    UnitSphere(uint nx);

  private:

    double transformx(double x,double y,double z);
    double transformy(double x,double y,double z);
    double transformz(double x,double y,double z);
    double max(double x,double y,double z);

  };
  
}

#endif
