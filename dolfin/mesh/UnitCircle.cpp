// Copyright (C) 2005-2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells 2007.
// Modified by Nuno Lopes 2008
//
// First added:  2005-12-02
// Last changed: 2008-06-19

#include "MeshEditor.h"
#include "UnitCircle.h"
#include <dolfin/main/MPI.h>
#include "MPIMeshCommunicator.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
UnitCircle::UnitCircle(uint nx, Type type,Trans trans) : Mesh()
{
  message("UnitCircle is Experimental: It could have a bad quality mesh");

  uint ny=nx;
  // Receive mesh according to parallel policy
  if (MPI::receive()) { MPIMeshCommunicator::receive(*this); return; }
  
  if ( nx < 1 || ny < 1 )
    error("Size of unit square must be at least 1 in each dimension.");

  rename("mesh", "Mesh of the unit square (0,1) x (0,1)");

  // Open mesh for editing
  MeshEditor editor;
  editor.open(*this, CellType::triangle, 2, 2);

  // Create vertices and cells:
  if (type == crisscross) 
  {
    editor.initVertices((nx+1)*(ny+1) + nx*ny);
    editor.initCells(4*nx*ny);
  } 
  else 
  {
    editor.initVertices((nx+1)*(ny+1));
    editor.initCells(2*nx*ny);
  }
  
  // Create main vertices:
  // variables for transformation
  real trns_x=0.0;
  real trns_y=0.0;
  uint vertex = 0;
  for (uint iy = 0; iy <= ny; iy++) 
  {
    const real y = -1.+static_cast<real>(iy)*2. / static_cast<real>(ny);
    for (uint ix = 0; ix <= nx; ix++) 
    {
      const real x =-1.+ static_cast<real>(ix)*2. / static_cast<real>(nx);
      trns_x=transformx(x,y,trans);
      trns_y=transformy(x,y,trans);
      editor.addVertex(vertex++, trns_x, trns_y);
    }
  }
  
  // Create midpoint vertices if the mesh type is crisscross
  if (type == crisscross) 
  {
    for (uint iy = 0; iy < ny; iy++) 
    {
      const real y =-1.+ (static_cast<real>(iy) + 0.5)*2. / static_cast<real>(ny);
      for (uint ix = 0; ix < nx; ix++) 
      {
        const real x =-1.+ (static_cast<real>(ix) + 0.5)*2. / static_cast<real>(nx);
        trns_x=transformx(x,y,trans);
        trns_y=transformy(x,y,trans);
        editor.addVertex(vertex++, trns_x, trns_y);
      }
    }
  }

  // Create triangles
  uint cell = 0;
  if (type == crisscross) 
  {
    for (uint iy = 0; iy < ny; iy++) 
    {
      for (uint ix = 0; ix < nx; ix++) 
      {
        const uint v0 = iy*(nx + 1) + ix;
        const uint v1 = v0 + 1;
        const uint v2 = v0 + (nx + 1);
        const uint v3 = v1 + (nx + 1);
        const uint vmid = (nx + 1)*(ny + 1) + iy*nx + ix;
	
        // Note that v0 < v1 < v2 < v3 < vmid.
        editor.addCell(cell++, v0, v1, vmid);
        editor.addCell(cell++, v0, v2, vmid);
        editor.addCell(cell++, v1, v3, vmid);
        editor.addCell(cell++, v2, v3, vmid);
      }
    }
  } 
  else if (type == left ) 
  {
    for (uint iy = 0; iy < ny; iy++) 
    {
      for (uint ix = 0; ix < nx; ix++) 
      {
        const uint v0 = iy*(nx + 1) + ix;
        const uint v1 = v0 + 1;
        const uint v2 = v0 + (nx + 1);
        const uint v3 = v1 + (nx + 1);

        editor.addCell(cell++, v0, v1, v2);
        editor.addCell(cell++, v1, v2, v3);
      }
    }
  } 
  else 
  { 
    for (uint iy = 0; iy < ny; iy++) 
    {
      for (uint ix = 0; ix < nx; ix++) 
      {
        const uint v0 = iy*(nx + 1) + ix;
        const uint v1 = v0 + 1;
        const uint v2 = v0 + (nx + 1);
        const uint v3 = v1 + (nx + 1);

        editor.addCell(cell++, v0, v1, v3);
        editor.addCell(cell++, v0, v2, v3);
      }
    }
  }

  // Close mesh editor
  editor.close();

  // Broadcast mesh according to parallel policy
  if (MPI::broadcast()) { MPIMeshCommunicator::broadcast(*this); }
}
//-----------------------------------------------------------------------------
real UnitCircle::transformx(real x,real y,Trans trans)
{
  real retrn=0.0;
  //maxn transformation
  if(trans==maxn)
    {
      if (x||y) //in (0,0) (trns_x,trans_y)=(nan,nan)
        retrn=x*max(fabs(x),fabs(y))/sqrt(x*x+y*y);
      else
        retrn=x;
    }
  //sumn transformation
  else if (trans==sumn)
    {
      if (x||y) //in (0,0) (trns_x,trans_y)=(nan,nan)
        retrn=x*(fabs(x)+fabs(y))/sqrt(x*x+y*y);
      else
        retrn=x;
    }
  else 
    {
      if ((trans!=maxn)*(trans!=sumn)*(trans!=rotsumn))
        { 
          message("Implemented  transformations are: maxn,sumn and rotsumn");
          message("Using rotsumn transformation");
        }
      if (x||y) //in (0,0) (trns_x,trans_y)=(nan,nan)
        {
          real xx=0.5*(x+y);
          real yy=0.5*(-x+y);
          retrn=xx*(fabs(xx)+fabs(yy))/sqrt(xx*xx+yy*yy);
        }
      else
        retrn=y;
    }
  
  return retrn;
}

real UnitCircle::transformy(real x,real y,Trans trans)
{
  real retrn=0.0;
  //maxn transformation
  if(trans==maxn)
    {
      if (x||y) //in (0,0) (trns_x,trans_y)=(nan,nan)
        retrn=y*max(fabs(x),fabs(y))/sqrt(x*x+y*y);
      else
        retrn=y;
    }
  //sumn transformation
  else if (trans==sumn)
    {
      if (x||y) //in (0,0) (trns_x,trans_y)=(nan,nan)
        retrn=y*(fabs(x)+fabs(y))/sqrt(x*x+y*y);
      else
        retrn=y;
    }
  else 
    {
      if ((trans!=maxn)*(trans!=sumn)*(trans!=rotsumn))
        {
          message("Implemented  transformations are: maxn,sumn and rotsumn");
          message("Using rotsumn transformation");
        }
      if (x||y) //in (0,0) (trns_x,trans_y)=(nan,nan)
        {
          real xx=0.5*(x+y);
          real yy=0.5*(-x+y);
          retrn=yy*(fabs(xx)+fabs(yy))/sqrt(xx*xx+yy*yy);
        }
      else
        retrn=y;
    }
  
  return retrn;
}
