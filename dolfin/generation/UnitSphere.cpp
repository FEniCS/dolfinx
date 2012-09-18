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
#include <dolfin/mesh/MeshPartitioning.h>
#include <dolfin/mesh/MeshEditor.h>
#include "UnitSphere.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
UnitSphere::UnitSphere(uint n) : Mesh()
{
  warning("The UnitSphere class is broken and should not be used for computations. "
          "It generates meshes of very bad quality (very thin tetrahedra).");

  // Receive mesh according to parallel policy
  if (MPI::is_receiver()) { MeshPartitioning::build_distributed_mesh(*this); return; }

  if (n < 1)
    dolfin_error("UnitSphere.cpp",
                 "create unit sphere",
                 "Size of unit sphere must be at least 1");

  const uint nx = n;
  const uint ny = n;
  const uint nz = n;

  rename("mesh", "Mesh of the unit sphere");

  // Open mesh for editing
  MeshEditor editor;
  editor.open(*this, CellType::tetrahedron, 3, 3);

  // Storage for vertices
  std::vector<double> x(3);

  // Create vertices
  editor.init_vertices((nx + 1)*(ny+1)*(nz+1));
  uint vertex = 0;
  for (uint iz = 0; iz <= nz; iz++)
  {
    x[2] = -1.0 + static_cast<double>(iz)*2.0/static_cast<double>(nz);
    for (uint iy = 0; iy <= ny; iy++)
    {
      x[1]  = -1.0 + static_cast<double>(iy)*2.0/static_cast<double>(ny);
      for (uint ix = 0; ix <= nx; ix++)
      {
        x[0] = -1.0 + static_cast<double>(ix)*2.0/static_cast<double>(nx);
        const std::vector<double> trans_x = transform(x);
        editor.add_vertex(vertex++, trans_x);
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
  if (MPI::is_broadcaster()) { MeshPartitioning::build_distributed_mesh(*this); }
}
//-----------------------------------------------------------------------------
std::vector<double> UnitSphere::transform(const std::vector<double>& x) const
{
  std::vector<double> x_trans(3);
  if (x[0] || x[1] || x[2])
  {
    const double dist = sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);
    x_trans[0] = x[0]*std::max(x)/dist;
    x_trans[1] = x[1]*std::max(x)/dist;
    x_trans[2] = x[2]*std::max(x))/dist;
  }
  else
    x_trans = x;
}
//-----------------------------------------------------------------------------
double UnitSphere::max(const std::vector<double>& x) const
{
  if ((std::abs(x[0]) >= std::abs(x[1]))*(std::abs(x[0]) >= std::abs(x[2])))
    return std::abs(x[0]);
  else if ((std::abs(x[1]) >= std::abs(x[0]))*(std::abs(x[1]) >= std::abs(x[2])))
    return std::abs(x[1]);
  else
    return std::abs(x[2]);
}
//-----------------------------------------------------------------------------
