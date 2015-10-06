// Copyright (C) 2015 Chris Richardson
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

#include <dolfin/common/MPI.h>
#include <dolfin/mesh/MeshEditor.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/Edge.h>
#include "SphericalShellMesh.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
SphericalShellMesh::SphericalShellMesh(MPI_Comm comm, std::size_t degree)
  : Mesh(comm)
{

  MeshEditor editor;
  const std::size_t tdim = 2;
  const std::size_t gdim = 3;

  dolfin_assert(degree > 0 and degree < 3);
  editor.open(*this, tdim, gdim, degree);

  editor.init_vertices_global(12, 12);

  const double l0 = 2.0/(sqrt(10.0 + 2.0*sqrt(5.0)));
  const double l1 = l0*(1.0 + sqrt(5.0))/2.0;

  // Generate an icosahedron

  editor.add_vertex(0,  Point(  0,  l0, l1));
  editor.add_vertex(1,  Point(  0,  l0, -l1));
  editor.add_vertex(2,  Point(  0, -l0, -l1));
  editor.add_vertex(3,  Point(  0, -l0, l1));
  editor.add_vertex(4,  Point( l1,   0, l0));
  editor.add_vertex(5,  Point(-l1,   0, l0));
  editor.add_vertex(6,  Point(-l1,   0, -l0));
  editor.add_vertex(7,  Point( l1,   0, -l0));
  editor.add_vertex(8,  Point( l0,  l1, 0));
  editor.add_vertex(9,  Point( l0, -l1, 0));
  editor.add_vertex(10, Point(-l0, -l1, 0));
  editor.add_vertex(11, Point(-l0,  l1, 0));

  editor.init_cells_global(20, 20);

  editor.add_cell(0, 0, 4, 8);
  editor.add_cell(1, 0, 5, 11);
  editor.add_cell(2, 1, 6, 11);
  editor.add_cell(3, 1, 7, 8);
  editor.add_cell(4, 2, 6, 10);
  editor.add_cell(5, 2, 7, 9);
  editor.add_cell(6, 3, 4, 9);
  editor.add_cell(7, 3, 5, 10);

  editor.add_cell( 8, 0, 3, 4);
  editor.add_cell( 9, 0, 3, 5);
  editor.add_cell(10, 1, 2, 6);
  editor.add_cell(11, 1, 2, 7);
  editor.add_cell(12, 4, 7, 8);
  editor.add_cell(13, 4, 7, 9);
  editor.add_cell(14, 5, 6, 10);
  editor.add_cell(15, 5, 6, 11);
  editor.add_cell(16, 8, 11, 0);
  editor.add_cell(17, 8, 11, 1);
  editor.add_cell(18, 9, 10, 2);
  editor.add_cell(19, 9, 10, 3);

  if (degree == 2)
  {
    // Initialise entities required for this degree polynomial mesh
    // and allocate space for the point coordinate data
    editor.init_entities();

    for (EdgeIterator e(*this); !e.end(); ++e)
    {
      Point v0 = Vertex(*this, e->entities(0)[0]).point();
      Point pt = e->midpoint();
      pt *= v0.norm()/pt.norm();

      // Add Edge-based point
      editor.add_entity_point(1, 0, e->index(), pt);
    }
  }

  editor.close();
}
//-----------------------------------------------------------------------------
