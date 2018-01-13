// Copyright (C) 2007 Kristian B. Oelgaard
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "IntervalMesh.h"
#include "dolfin/common/MPI.h"
#include "dolfin/common/constants.h"
#include "dolfin/mesh/CellType.h"
#include "dolfin/mesh/MeshEditor.h"
#include "dolfin/mesh/MeshPartitioning.h"
#include <cmath>

using namespace dolfin;

//-----------------------------------------------------------------------------
void IntervalMesh::build(Mesh& mesh, std::size_t nx, std::array<double, 2> x)
{
  // Receive mesh according to parallel policy
  if (MPI::is_receiver(mesh.mpi_comm()))
  {
    MeshPartitioning::build_distributed_mesh(mesh);
    return;
  }

  const double a = x[0];
  const double b = x[1];

  if (std::abs(a - b) < DOLFIN_EPS)
  {
    dolfin_error(
        "Interval.cpp", "create interval",
        "Length of interval is zero. Consider checking your dimensions");
  }

  if (b < a)
  {
    dolfin_error("Interval.cpp", "create interval",
                 "Length of interval is negative. Consider checking the order "
                 "of your arguments");
  }

  if (nx < 1)
  {
    dolfin_error("Interval.cpp", "create interval",
                 "Number of points on interval is (%d), it must be at least 1",
                 nx);
  }

  mesh.rename("mesh", "Mesh of the interval (a, b)");

  // Open mesh for editing
  MeshEditor editor;
  editor.open(mesh, CellType::Type::interval, 1, 1);

  // Create vertices and cells:
  editor.init_vertices_global((nx + 1), (nx + 1));
  editor.init_cells_global(nx, nx);

  // Create main vertices:
  for (std::size_t ix = 0; ix <= nx; ix++)
  {
    Point x(a + (static_cast<double>(ix) * (b - a) / static_cast<double>(nx)));
    editor.add_vertex(ix, x);
  }

  // Create intervals
  for (std::size_t ix = 0; ix < nx; ix++)
  {
    std::vector<std::size_t> cell(2);
    cell[0] = ix;
    cell[1] = ix + 1;
    editor.add_cell(ix, cell);
  }

  // Close mesh editor
  editor.close();

  // Broadcast mesh according to parallel policy
  if (MPI::is_broadcaster(mesh.mpi_comm()))
  {
    MeshPartitioning::build_distributed_mesh(mesh);
    return;
  }
}
//-----------------------------------------------------------------------------
