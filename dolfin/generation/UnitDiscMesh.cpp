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

#include <cmath>
#include <dolfin/common/MPI.h>
#include <dolfin/mesh/MeshEditor.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/Edge.h>
#include "UnitDiscMesh.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
UnitDiscMesh::UnitDiscMesh(MPI_Comm comm, std::size_t n, std::size_t degree,
                           std::size_t gdim) : Mesh(comm)
{
  dolfin_assert(n > 0);

  if (!(gdim == 2 or gdim == 3))
  {
    dolfin_error("UnitDiscMesh.cpp",
                 "create mesh",
                 "geometric dimension must be two or three");
  }

  if (!(degree == 1 or degree == 2))
  {
    dolfin_error("UnitDiscMesh.cpp",
                 "create mesh",
                 "isoparametric degree must be one or two");
  }

  MeshEditor editor;
  editor.open(*this, 2, gdim, degree);
  editor.init_vertices_global(1 + 3*n*(n + 1), 1 + 3*n*(n + 1));

  std::size_t c = 0;
  editor.add_vertex(c, Point(0.0, 0.0, 0.0));
  ++c;

  for (std::size_t i = 1; i <= n; ++i)
  {
    for (std::size_t j = 0; j < 6*i; ++j)
    {
      const double r = (double)i/(double)n;
      const double th = 2*M_PI*(double)j/(double)(6*i);
      const double x = r*cos(th);
      const double y = r*sin(th);
      editor.add_vertex(c, Point(x, y, 0));
      ++c;
    }
  }

  editor.init_cells(6*n*n);
  c = 0;
  std::size_t base_i = 0;
  std::size_t row_i = 1;
  for (std::size_t i = 1; i <= n; ++i)
  {
    std::size_t base_m = base_i;
    base_i = 1 + 3*i*(i - 1);
    std::size_t row_m = row_i;
    row_i = 6*i;
    for (std::size_t k = 0; k != 6; ++k)
    {
      for (std::size_t j = 0; j < (i*2 - 1); ++j)
      {
        std::size_t i0, i1, i2;
        if (j%2 == 0)
        {
          i0 = base_i + (k*i + j/2)%row_i;
          i1 = base_i + (k*i + j/2 + 1)%row_i;
          i2 = base_m + (k*(i-1) + j/2)%row_m;
        }
        else
        {
          i0 = base_m + (k*(i-1) + j/2)%row_m;
          i1 = base_m + (k*(i-1) + j/2 + 1)%row_m;
          i2 = base_i + (k*i + j/2 + 1)%row_i;
        }

        editor.add_cell(c, i0, i1, i2);
        ++c;
      }
    }
  }

  // Initialise entities required for this degree polynomial mesh and
  // allocate space for the point coordinate data
  if (degree == 2)
  {
    editor.init_entities();

    for (EdgeIterator e(*this); !e.end(); ++e)
    {
      const Point v0 = Vertex(*this, e->entities(0)[0]).point();
      const Point v1 = Vertex(*this, e->entities(0)[1]).point();
      Point pt = e->midpoint();

      if (std::abs(v0.norm() - 1.0) < 1e-6 and
          std::abs(v1.norm() - 1.0) < 1e-6)
      {
        pt *= v0.norm()/pt.norm();
      }

      // Add Edge-based point
      editor.add_entity_point(1, 0, e->index(), pt);
    }
  }

  editor.close();
}
//-----------------------------------------------------------------------------
