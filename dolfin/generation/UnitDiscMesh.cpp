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
#include "UnitDiscMesh.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
UnitDiscMesh::UnitDiscMesh(MPI_Comm comm, std::size_t n, std::size_t gdim)
  : Mesh(comm)
{
  dolfin_assert(n > 0);
  dolfin_assert(gdim == 2 or gdim == 3);

  MeshEditor editor;
  editor.open(*this, 2, gdim);
  editor.init_vertices(1 + 3*n*(n + 1));

  std::size_t c = 0;
  editor.add_vertex(c, Point(0,0,0));
  ++c;

  for (std::size_t i = 1; i <= n; ++i)
    for (std::size_t j = 0; j < 6*i; ++j)
    {
      double r = (double)i/(double)n;
      double th = 2*M_PI*(double)j/(double)(6*i);
      double x = r*cos(th);
      double y = r*sin(th);
      editor.add_vertex(c, Point(x, y, 0));
      ++c;
    }

  editor.init_cells(6*n*n);

  c = 0;
  for (std::size_t i = 1; i <= n; ++i)
  {
    const std::size_t base_i = 1 + 3*i*(i - 1);
    const std::size_t base_m = (i>1 ? 1 : 0) + 3*(i-1)*(i-2);
    const std::size_t row_i = 6*i;
    const std::size_t row_m = (i>1) ? 6*(i-1) : 1;
    for (std::size_t k = 0; k != 6; ++k)
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
        std::cout << "(" << i <<", " << j  << ", " << k << ") : ";
        std::cout << i0 <<", " << i1  << ", " << i2 << "\n" ;


        editor.add_cell(c, i0, i1, i2);
        ++c;
      }
  }


  editor.close();
}
//-----------------------------------------------------------------------------
