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
// Modified by Garth N. Wells 2007.
// Modified by Nuno Lopes 2008
//
// First added:  2005-12-02
// Last changed: 2011-08-23

#include <dolfin/common/constants.h>
#include <dolfin/common/MPI.h>
#include <dolfin/mesh/MeshPartitioning.h>
#include <dolfin/mesh/MeshEditor.h>
#include "UnitCircle.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
UnitCircle::UnitCircle(uint nx, std::string diagonal,
                       std::string transformation) : Mesh()
{
  // Receive mesh according to parallel policy
  if (MPI::is_receiver()) { MeshPartitioning::build_distributed_mesh(*this); return; }

  if (diagonal != "left" && diagonal != "right" && diagonal != "crossed")
    dolfin_error("UnitCircle.cpp",
                 "create unit circle",
                 "Unknown mesh diagonal definition: Allowed options are \"left\", \"right\" and \"crossed\"");

  if (transformation != "maxn" && transformation != "sumn" && transformation != "rotsumn")
    dolfin_error("UnitCircle.cpp",
                 "create unit circle",
                 "Unknown transformation '%s' in UnitCircle. Allowed options are \"maxn\", \"sumn\" and \"rotsumn\"",
                 transformation.c_str());

  if ( nx < 1 )
    dolfin_error("UnitCircle.cpp",
                 "create unit circle",
                 "Size of unit square must be at least 1");

  const uint ny = nx;

  rename("mesh", "Mesh of the unit circle");

  // Open mesh for editing
  MeshEditor editor;
  editor.open(*this, CellType::triangle, 2, 2);

  // Create vertices and cells:
  if (diagonal == "crossed")
  {
    editor.init_vertices((nx+1)*(ny+1) + nx*ny);
    editor.init_cells(4*nx*ny);
  }
  else
  {
    editor.init_vertices((nx+1)*(ny+1));
    editor.init_cells(2*nx*ny);
  }

  // Create main vertices
  uint vertex = 0;
  double x_trans[2];
  for (uint iy = 0; iy <= ny; iy++)
  {
    const double y = -1.0 + static_cast<double>(iy)*2.0/static_cast<double>(ny);
    for (uint ix = 0; ix <= nx; ix++)
    {
      const double x = -1.0 + static_cast<double>(ix)*2.0/static_cast<double>(nx);
      transform(x_trans, x, y, transformation);
      editor.add_vertex(vertex++, x_trans[0], x_trans[1]);
    }
  }

  // Create midpoint vertices if the mesh type is crisscross
  if (diagonal == "crossed")
  {
    for (uint iy = 0; iy < ny; iy++)
    {
      const double y = -1.0 + (static_cast<double>(iy) + 0.5)*2.0 / static_cast<double>(ny);
      for (uint ix = 0; ix < nx; ix++)
      {
        const double x = -1.0 + (static_cast<double>(ix) + 0.5)*2.0 / static_cast<double>(nx);
        transform(x_trans, x, y, transformation);
        editor.add_vertex(vertex++, x_trans[0], x_trans[1]);
      }
    }
  }

  // Create triangles
  uint cell = 0;
  if (diagonal == "crossed")
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
        editor.add_cell(cell++, v0, v1, vmid);
        editor.add_cell(cell++, v0, v2, vmid);
        editor.add_cell(cell++, v1, v3, vmid);
        editor.add_cell(cell++, v2, v3, vmid);
      }
    }
  }
  else if (diagonal == "left" ||  diagonal == "right")
  {
    for (uint iy = 0; iy < ny; iy++)
    {
      for (uint ix = 0; ix < nx; ix++)
      {
        const uint v0 = iy*(nx + 1) + ix;
        const uint v1 = v0 + 1;
        const uint v2 = v0 + (nx + 1);
        const uint v3 = v1 + (nx + 1);

        if(diagonal == "left")
        {
          editor.add_cell(cell++, v0, v1, v2);
          editor.add_cell(cell++, v1, v2, v3);
        }
        else
        {
          editor.add_cell(cell++, v0, v1, v3);
          editor.add_cell(cell++, v0, v2, v3);
        }
      }
    }
  }
  else
    dolfin_error("UnitCircle.cpp",
                 "create unit circle",
                 "Unknown mesh diagonal definition: Allowed options are \"left\", \"right\" and \"crossed\"");

  // Close mesh editor
  editor.close();

  // Broadcast mesh according to parallel policy
  if (MPI::is_broadcaster()) { MeshPartitioning::build_distributed_mesh(*this); return; }
}
//-----------------------------------------------------------------------------
void UnitCircle::transform(double* trans, double x, double y, std::string transformation)
{
  if (std::abs(x) < DOLFIN_EPS && std::abs(y) < DOLFIN_EPS)
  {
    trans[0] = x;
    trans[1] = y;
    return;
  }

  if(transformation == "maxn")
  {
    trans[0] = x*max(fabs(x),fabs(y))/sqrt(x*x+y*y);
    trans[1] = y*max(fabs(x),fabs(y))/sqrt(x*x+y*y);
  }
  else if (transformation == "sumn")
  {
    // FIXME: This option should either be removed or fixed
    dolfin_error("UnitCircle.cpp",
                 "transform to unit circle",
                 "'sumn' mapping for a UnitCircle is broken");
    trans[0] = x*(fabs(x)+fabs(y))/sqrt(x*x+y*y);
    trans[1] = y*(fabs(x)+fabs(y))/sqrt(x*x+y*y);
  }
  else if (transformation == "rotsumn")
  {
    double xx = 0.5*(x+y);
    double yy = 0.5*(-x+y);
    trans[0] = xx*(fabs(xx)+fabs(yy))/sqrt(xx*xx+yy*yy);
    trans[1] = yy*(fabs(xx)+fabs(yy))/sqrt(xx*xx+yy*yy);
  }
  else
    dolfin_error("UnitCircle.cpp",
                 "transform to unit circle",
                 "Unknown transformation '%s' in UnitCircle. Allowed options are \"maxn\", \"sumn\" and \"rotsumn\"",
                 transformation.c_str());
}
//-----------------------------------------------------------------------------
