// Copyright (C) 2005-2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2007.
// Modified by Nuno Lopes, 2008
//
// First added:  2005-12-02
// Last changed: 2008-06-20

#include "MeshEditor.h"
#include "UnitSphere.h"
#include <dolfin/main/MPI.h>
#include "MPIMeshCommunicator.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
UnitSphere::UnitSphere(uint nx) : Mesh()
{

  message("UnitSphere is Experimental: It could have a bad quality mesh");

  uint ny=nx;
  uint nz=nx;
  
  // Receive mesh according to parallel policy
  if (MPI::receive()) { MPIMeshCommunicator::receive(*this); return; }

  if ( nx < 1 || ny < 1 || nz < 1 )
    error("Size of unit cube must be at least 1 in each dimension.");

  rename("mesh", "Mesh of the unit cube (0,1) x (0,1) x (0,1)");

  // Open mesh for editing
  MeshEditor editor;
  editor.open(*this, CellType::tetrahedron, 3, 3);

  // Create vertices
  editor.initVertices((nx+1)*(ny+1)*(nz+1));
  uint vertex = 0;
  for (uint iz = 0; iz <= nz; iz++)
  {
    const real z = -1.0+ static_cast<real>(iz)*2.0 / static_cast<real>(nz);
    for (uint iy = 0; iy <= ny; iy++)
    {
      const real y =-1.0+ static_cast<real>(iy)*2.0 / static_cast<real>(ny);
      for (uint ix = 0; ix <= nx; ix++)
      {
        const real x = -1.0+static_cast<real>(ix)*2.0 / static_cast<real>(nx);
        real trns_x=transformx(x,y,z);
        real trns_y=transformy(x,y,z);
        real trns_z=transformz(x,y,z);
        editor.addVertex(vertex++, trns_x, trns_y, trns_z);
      }
    }
  }

  // Create tetrahedra
  editor.initCells(6*nx*ny*nz);
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

        editor.addCell(cell++, v0, v1, v3, v7);
        editor.addCell(cell++, v0, v1, v7, v5);
        editor.addCell(cell++, v0, v5, v7, v4);
        editor.addCell(cell++, v0, v3, v2, v7);
        editor.addCell(cell++, v0, v6, v4, v7);
        editor.addCell(cell++, v0, v2, v6, v7);
      }
    }
  }

  // Close mesh editor
  editor.close();

  // Broadcast mesh according to parallel policy
  if (MPI::broadcast()) { MPIMeshCommunicator::broadcast(*this); }
}
//-----------------------------------------------------------------------------
real UnitSphere::transformx(real x,real y,real z)
{
  real retrn=0.0;
  if (x||y||z) 
    retrn=x*max(fabs(x),fabs(y),fabs(z))/sqrt(x*x+y*y+z*z);
  else
    retrn=x;
  return retrn;
}
//-----------------------------------------------------------------------------
real UnitSphere::transformy(real x,real y,real z)
{
  real retrn=0.0;
  if (x||y||z) 
    retrn=y*max(fabs(x),fabs(y),fabs(z))/sqrt(x*x+y*y+z*z);
  else
    retrn=y;
  return retrn;
}
//-----------------------------------------------------------------------------
real UnitSphere::transformz(real x,real y,real z)
{
  real retrn=0.0;
  //maxn transformation
  if (x||y||z)
    retrn=z*max(fabs(x),fabs(y),fabs(z))/sqrt(x*x+y*y+z*z);
  else
    retrn=z;
  return retrn;
}
//-----------------------------------------------------------------------------
real UnitSphere::max(real x,real y, real z)
{
  real rtrn=0.0;

  if ((x>=y)*(x>=z))
    rtrn=x;
  else if ((y>=x)*(y>=z))
    rtrn=y;
  else
    rtrn=z; 
  return rtrn;
}
//-----------------------------------------------------------------------------
