// Copyright (C) 2007 Kristian B. Oelgaard
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
// Modified by N. Lopes, 2008.
// Modified by Mikael Mortensen, 2014.
//
// First added:  2007-11-23
// Last changed: 2014-02-17

#include <cmath>
#include "dolfin/common/constants.h"
#include "dolfin/common/MPI.h"
#include "dolfin/mesh/CellType.h"
#include "dolfin/mesh/MeshEditor.h"
#include "dolfin/mesh/MeshPartitioning.h"
#include "IntervalMesh.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
IntervalMesh::IntervalMesh(std::size_t nx, double a, double b)
  : Mesh(MPI_COMM_WORLD)
{
  build(nx, a, b);
}
//-----------------------------------------------------------------------------
IntervalMesh::IntervalMesh(MPI_Comm comm, std::size_t nx, double a, double b)
  : Mesh(comm)
{
  build(nx, a, b);
}
//-----------------------------------------------------------------------------
void IntervalMesh::build(std::size_t nx, double a, double b)
{
  // Receive mesh according to parallel policy
  if (MPI::is_receiver(this->mpi_comm()))
  {
    MeshPartitioning::build_distributed_mesh(*this);
    return;
  }

  if (std::abs(a - b) < DOLFIN_EPS)
  {
    dolfin_error("Interval.cpp",
                 "create interval",
                 "Length of interval is zero. Consider checking your dimensions");
  }

  if (b < a)
  {
    dolfin_error("Interval.cpp",
                 "create interval",
                 "Length of interval is negative. Consider checking the order of your arguments");
  }

  if (nx < 1)
  {
    dolfin_error("Interval.cpp",
                 "create interval",
                 "Number of points on interval is (%d), it must be at least 1", nx);
  }

  rename("mesh", "Mesh of the interval (a, b)");

  // Open mesh for editing
  MeshEditor editor;
  editor.open(*this, CellType::interval, 1, 1);

  // Create vertices and cells:
  editor.init_vertices_global((nx+1), (nx+1));
  editor.init_cells_global(nx, nx);

  // Create main vertices:
  for (std::size_t ix = 0; ix <= nx; ix++)
  {
    const std::vector<double>
      x(1, a + (static_cast<double>(ix)*(b - a)/static_cast<double>(nx)));
    editor.add_vertex(ix, x);
  }

  // Create intervals
  for (std::size_t ix = 0; ix < nx; ix++)
  {
    std::vector<std::size_t> cell(2);
    cell[0] = ix; cell[1] = ix + 1;
    editor.add_cell(ix, cell);
  }

  // Close mesh editor
  editor.close();

  // Broadcast mesh according to parallel policy
  if (MPI::is_broadcaster(this->mpi_comm()))
  {
    std::cout << "Building mesh (dist 0a)" << std::endl;
    MeshPartitioning::build_distributed_mesh(*this);
    std::cout << "Building mesh (dist 1a)" << std::endl;
    return;
  }
}
//-----------------------------------------------------------------------------
