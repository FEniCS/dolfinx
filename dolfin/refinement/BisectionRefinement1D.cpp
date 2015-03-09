// Copyright (C) 2014 Chris Richardson
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

#include <vector>

#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/MeshEditor.h>
#include "BisectionRefinement1D.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void BisectionRefinement1D::refine(Mesh& refined_mesh,
                                  const Mesh& mesh,
                                  const MeshFunction<bool>& cell_markers)
{
  not_working_in_parallel("BisectionRefinement1D::refine");

  if (mesh.topology().dim() != 1)
  {
    dolfin_error("BisectionRefinement1D.cpp",
                 "refine mesh",
                 "Mesh is not one-dimensional");
  }

  // Count the number of cells in refined mesh
  std::size_t num_new_vertices = 0;

  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    if (cell_markers[*cell])
      ++num_new_vertices;
  }

  // Initialize mesh editor
  const std::size_t vertex_offset = mesh.num_vertices();
  const std::size_t num_vertices = vertex_offset + num_new_vertices;
  const std::size_t num_cells = mesh.num_cells() + num_new_vertices;

  MeshEditor editor;
  editor.open(refined_mesh, mesh.topology().dim(), mesh.geometry().dim());
  editor.init_vertices_global(num_vertices, num_vertices);
  editor.init_cells_global(num_cells, num_cells);

  // Set vertex coordinates
  std::size_t current_vertex = 0;
  for (VertexIterator vertex(mesh); !vertex.end(); ++vertex)
  {
    editor.add_vertex(current_vertex, vertex->point());
    ++current_vertex;
  }

  std::size_t current_cell = 0;
  std::vector<std::size_t> cell_data(2);
  for (CellIterator cell(mesh); !cell.end(); ++cell) {
    if (cell_markers[*cell])
    {
      editor.add_vertex(current_vertex, cell->midpoint());

      cell_data[0] = cell->entities(0)[0];
      cell_data[1] = current_vertex;
      editor.add_cell(current_cell++, cell_data);
      cell_data[0] = current_vertex;
      cell_data[1] = cell->entities(0)[1];
      editor.add_cell(current_cell, cell_data);

      ++current_vertex;
    }
    else
    {
      cell_data[0] = cell->entities(0)[0];
      cell_data[1] = cell->entities(0)[1];
      editor.add_cell(current_cell, cell_data);
    }

    ++current_cell;
  }

  // Close mesh editor
  dolfin_assert(num_cells == current_cell);
  dolfin_assert(num_vertices == current_vertex);
  editor.close();
}
//-----------------------------------------------------------------------------
