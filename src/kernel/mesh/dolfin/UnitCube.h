// Copyright (C) 2005-2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-12-02
// Last changed: 2006-08-07

#ifndef __NEW_UNIT_CUBE_H
#define __NEW_UNIT_CUBE_H

#include <dolfin/Mesh.h>

namespace dolfin
{

  /// Tetrahedral mesh of the 3D unit cube (0,1) x (0,1) x (0,1).
  /// Given the number of cells (nx, ny, nz) in each direction,
  /// the total number of tetrahedra will be 6*nx*ny*nz and the
  /// total number of vertices will be (nx + 1)*(ny + 1)*(nz + 1).

  class UnitCube : public Mesh
  {
  public:

    UnitCube(uint nx, uint ny, uint nz);

  };
  
}

#endif
