// Copyright (C) 2008 Kristian B. Oelgaard
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
// First added:  2007-11-23
// Last changed: 2012-09-27

#include <boost/assign.hpp>
#include <dolfin/common/MPI.h>
#include <dolfin/mesh/MeshPartitioning.h>
#include <dolfin/mesh/MeshEditor.h>
#include "UnitIntervalMesh.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
UnitIntervalMesh::UnitIntervalMesh(std::size_t nx) : Mesh()
{
  // Receive mesh according to parallel policy
  if (MPI::is_receiver())
  {
    MeshPartitioning::build_distributed_mesh(*this);
    return;
  }

  if (nx < 1)
  {
    dolfin_error("UnitIntervalMesh.cpp",
                 "create unit interval",
                 "Size of unit interval must be at least 1");
  }

  rename("mesh", "Mesh of the unit interval (0,1)");

  // Open mesh for editing
  MeshEditor editor;
  editor.open(*this, CellType::interval, 1, 1);

  // Create vertices and cells:
  editor.init_vertices((nx+1));
  editor.init_cells(nx);

  // Create main vertices:
  for (std::size_t ix = 0; ix <= nx; ix++)
  {
    const double x = static_cast<double>(ix)/static_cast<double>(nx);
    editor.add_vertex(ix, x);
  }

  // Create intervals
  for (std::size_t ix = 0; ix < nx; ix++)
  {
    std::vector<std::size_t> cell_data = boost::assign::list_of(ix)(ix + 1);
    editor.add_cell(ix, cell_data);
  }

  // Close mesh editor
  editor.close();

  // Broadcast mesh according to parallel policy
  if (MPI::is_broadcaster())
  {
    MeshPartitioning::build_distributed_mesh(*this);
    return;
  }
}
//-----------------------------------------------------------------------------
