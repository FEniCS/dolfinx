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
// Last changed: 2012-11-09

#include <boost/assign.hpp>

#include <dolfin/common/MPI.h>
#include <dolfin/mesh/MeshPartitioning.h>
#include <dolfin/mesh/MeshEditor.h>
#include "UnitSphereMesh.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
UnitSphereMesh::UnitSphereMesh(uint n) : Mesh()
{
  warning("The UnitSphereMesh class is broken and should not be used for computations. "
          "It generates meshes of very bad quality (very thin tetrahedra).");

  // Receive mesh according to parallel policy
  if (MPI::is_receiver())
  {
    MeshPartitioning::build_distributed_mesh(*this);
    return;
  }

  if (n < 1)
  {
    dolfin_error("UnitSphereMesh.cpp",
                 "create unit sphere",
                 "Size of unit sphere must be at least 1");
  }

  const std::size_t nx = n;
  const std::size_t ny = n;
  const std::size_t nz = n;

  rename("mesh", "Mesh of the unit sphere");

  // Open mesh for editing
  MeshEditor editor;
  editor.open(*this, CellType::tetrahedron, 3, 3);

  // Storage for vertices
  std::vector<double> x(3);

  // Create vertices
  editor.init_vertices((nx + 1)*(ny+1)*(nz+1));
  std::size_t vertex = 0;
  for (std::size_t iz = 0; iz <= nz; iz++)
  {
    x[2] = -1.0 + static_cast<double>(iz)*2.0/static_cast<double>(nz);
    for (std::size_t iy = 0; iy <= ny; iy++)
    {
      x[1] = -1.0 + static_cast<double>(iy)*2.0/static_cast<double>(ny);
      for (std::size_t ix = 0; ix <= nx; ix++)
      {
        x[0] = -1.0 + static_cast<double>(ix)*2.0/static_cast<double>(nx);
        const std::vector<double> trans_x = transform(x);
        editor.add_vertex(vertex, trans_x);
        ++vertex;
      }
    }
  }

  // Create tetrahedra
  editor.init_cells(6*nx*ny*nz);
  std::size_t cell = 0;
  std::vector<std::vector<std::size_t> > cells(6, std::vector<std::size_t>(4));
  for (std::size_t iz = 0; iz < nz; iz++)
  {
    for (std::size_t iy = 0; iy < ny; iy++)
    {
      for (std::size_t ix = 0; ix < nx; ix++)
      {
        const std::size_t v0 = iz*(nx + 1)*(ny + 1) + iy*(nx + 1) + ix;
        const std::size_t v1 = v0 + 1;
        const std::size_t v2 = v0 + (nx + 1);
        const std::size_t v3 = v1 + (nx + 1);
        const std::size_t v4 = v0 + (nx + 1)*(ny + 1);
        const std::size_t v5 = v1 + (nx + 1)*(ny + 1);
        const std::size_t v6 = v2 + (nx + 1)*(ny + 1);
        const std::size_t v7 = v3 + (nx + 1)*(ny + 1);

        // Note that v0 < v1 < v2 < v3 < vmid.
        cells[0][0] = v0; cells[0][1] = v1; cells[0][2] = v3; cells[0][3] = v7;
        cells[1][0] = v0; cells[1][1] = v1; cells[1][2] = v7; cells[1][3] = v5;
        cells[2][0] = v0; cells[2][1] = v5; cells[2][2] = v7; cells[2][3] = v4;
        cells[3][0] = v0; cells[3][1] = v3; cells[3][2] = v2; cells[3][3] = v7;
        cells[4][0] = v0; cells[4][1] = v6; cells[4][2] = v4; cells[4][3] = v7;
        cells[5][0] = v0; cells[5][1] = v2; cells[5][2] = v6; cells[5][3] = v7;

        // Add cells
        std::vector<std::vector<std::size_t> >::const_iterator _cell;
        for (_cell = cells.begin(); _cell != cells.end(); ++_cell)
          editor.add_cell(cell++, *_cell);
      }
    }
  }

  // Close mesh editor
  editor.close();

  // Broadcast mesh according to parallel policy
  if (MPI::is_broadcaster()) { MeshPartitioning::build_distributed_mesh(*this); }
}
//-----------------------------------------------------------------------------
std::vector<double> UnitSphereMesh::transform(const std::vector<double>& x) const
{
  std::vector<double> x_trans(3);
  if (x[0] || x[1] || x[2])
  {
    const double dist = sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);
    x_trans[0] = x[0]*max(x)/dist;
    x_trans[1] = x[1]*max(x)/dist;
    x_trans[2] = x[2]*max(x)/dist;
  }
  else
    x_trans = x;

  return x_trans;
}
//-----------------------------------------------------------------------------
double UnitSphereMesh::max(const std::vector<double>& x) const
{
  if ((std::abs(x[0]) >= std::abs(x[1]))*(std::abs(x[0]) >= std::abs(x[2])))
    return std::abs(x[0]);
  else if ((std::abs(x[1]) >= std::abs(x[0]))*(std::abs(x[1]) >= std::abs(x[2])))
    return std::abs(x[1]);
  else
    return std::abs(x[2]);
}
//-----------------------------------------------------------------------------
