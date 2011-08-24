// Copyright (C) 2005-2011 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Garth N. Wells, 2007.
// Modified by Nuno Lopes, 2008
//
// First added:  2005-12-02
// Last changed: 2011-08-23

#include <dolfin/common/MPI.h>
#include "MeshPartitioning.h"
#include "MeshEditor.h"
#include "UnitSphere.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
UnitSphere::UnitSphere(uint nx) : Mesh()
{
  // Receive mesh according to parallel policy
  if (MPI::is_receiver()) { MeshPartitioning::partition(*this); return; }

  const uint ny = nx;
  const uint nz = nx;

  if (nx < 1 || ny < 1 || nz < 1)
    error("Size of unit cube must be at least 1 in each dimension.");

  rename("mesh", "Mesh of the unit cube (0,1) x (0,1) x (0,1)");

  // Open mesh for editing
  MeshEditor editor;
  editor.open(*this, CellType::tetrahedron, 3, 3);

  // Create vertices
  editor.init_vertices((nx+1)*(ny+1)*(nz+1));
  uint vertex = 0;
  for (uint iz = 0; iz <= nz; iz++)
  {
    const double z = -1.0+ static_cast<double>(iz)*2.0 / static_cast<double>(nz);
    for (uint iy = 0; iy <= ny; iy++)
    {
      const double y =-1.0+ static_cast<double>(iy)*2.0 / static_cast<double>(ny);
      for (uint ix = 0; ix <= nx; ix++)
      {
        const double x = -1.0+static_cast<double>(ix)*2.0 / static_cast<double>(nx);
        double trns_x = transformx(x,y,z);
        double trns_y = transformy(x,y,z);
        double trns_z = transformz(x,y,z);
        editor.add_vertex(vertex++, trns_x, trns_y, trns_z);
      }
    }
  }

  // Create tetrahedra
  editor.init_cells(6*nx*ny*nz);
  uint cell = 0;
  for (uint iz = 0; iz < nz; iz++)
  {
    for (uint iy = 0; iy < ny; iy++)
    {
      for (uint ix = 0; ix < nx; ix++)
      {
        const uint v0 = iz*(nx + 1)*(ny + 1) + iy*(nx + 1) + ix;
        const uint v1 = v0 + 1;
        const uint v2 = v0 + (nx + 1);
        const uint v3 = v1 + (nx + 1);
        const uint v4 = v0 + (nx + 1)*(ny + 1);
        const uint v5 = v1 + (nx + 1)*(ny + 1);
        const uint v6 = v2 + (nx + 1)*(ny + 1);
        const uint v7 = v3 + (nx + 1)*(ny + 1);

        editor.add_cell(cell++, v0, v1, v3, v7);
        editor.add_cell(cell++, v0, v1, v7, v5);
        editor.add_cell(cell++, v0, v5, v7, v4);
        editor.add_cell(cell++, v0, v3, v2, v7);
        editor.add_cell(cell++, v0, v6, v4, v7);
        editor.add_cell(cell++, v0, v2, v6, v7);
      }
    }
  }

  // Close mesh editor
  editor.close();

  // Broadcast mesh according to parallel policy
  if (MPI::is_broadcaster()) { MeshPartitioning::partition(*this); }
}
//-----------------------------------------------------------------------------
double UnitSphere::transformx(double x,double y,double z)
{
  if (x || y || z)
    return x*max(fabs(x),fabs(y),fabs(z))/sqrt(x*x+y*y+z*z);
  else
    return x;
}
//-----------------------------------------------------------------------------
double UnitSphere::transformy(double x,double y,double z)
{
  if (x || y || z)
    return y*max(fabs(x),fabs(y),fabs(z))/sqrt(x*x+y*y+z*z);
  else
    return y;
}
//-----------------------------------------------------------------------------
double UnitSphere::transformz(double x,double y,double z)
{
  if (x || y || z)
    return z*max(fabs(x),fabs(y),fabs(z))/sqrt(x*x+y*y+z*z);
  else
    return z;
}
//-----------------------------------------------------------------------------
double UnitSphere::max(double x,double y, double z)
{
  if ((x >= y)*(x >= z))
    return x;
  else if ((y >= x)*(y >= z))
    return y;
  else
    return z;
}
//-----------------------------------------------------------------------------
