// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __UNIT_CUBE_H
#define __UNIT_CUBE_H

#include <dolfin/Mesh.h>

namespace dolfin
{

  /// This class represents a tetrahedral mesh of the 3D unit
  /// cube, i.e., (0,1) x (0,1) x (0,1). Given the number of
  /// cells (nx, ny, nz) in each direction, the total number
  /// of tetrahedra will be 6*nx*ny*nz and the total number
  /// of nodes will be (nx + 1)*(ny + 1)*(nz + 1).

  class UnitCube : public Mesh
  {
  public:

    UnitCube(uint nx, uint ny, uint nz);

  };
  
}

#endif
