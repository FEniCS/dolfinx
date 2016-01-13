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
#include "ParallelRefinement.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void BisectionRefinement1D::refine(Mesh& refined_mesh,
                                   const Mesh& mesh,
                                   const MeshFunction<bool>& cell_markers,
                                   bool redistribute)
{
  if (mesh.topology().dim() != 1)
  {
    dolfin_error("BisectionRefinement1D.cpp",
                 "refine mesh",
                 "Mesh is not one-dimensional");
  }

  ParallelRefinement p_ref(mesh);
  // Edges are the same as cells in 1D
  for (CellIterator cell(mesh); !cell.end(); ++cell)
    if (cell_markers[*cell])
      p_ref.mark(cell->index());

  p_ref.create_new_vertices();
  const std::map<std::size_t, std::size_t>& new_vertex_map
    = *(p_ref.edge_to_new_vertex());

  std::vector<std::size_t> parent_cell;
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    std::size_t cell_index = cell->index();

    std::vector<std::size_t> indices;
    for (VertexIterator v(*cell); !v.end(); ++v)
      indices.push_back(v->global_index());

    if (p_ref.is_marked(cell_index))
    {
      auto it = new_vertex_map.find(cell_index);
      dolfin_assert (it != new_vertex_map.end());

      std::vector<std::size_t> new_cells
        = {indices[0], it->second,
           it->second, indices[1]};
      p_ref.new_cells(new_cells);
      parent_cell.push_back(cell_index);
      parent_cell.push_back(cell_index);
    }
    else
    {
      p_ref.new_cell(*cell);
      parent_cell.push_back(cell_index);
    }
  }

  const bool serial = (MPI::size(mesh.mpi_comm()) == 1);

  if (serial)
    p_ref.build_local(refined_mesh);
  else
    p_ref.partition(refined_mesh, redistribute);

  if (serial || !redistribute)
  {
    // Create parent data on new mesh
    std::vector<std::size_t>& new_parent_cell
      = refined_mesh.data().create_array("parent_cell", refined_mesh.topology().dim());
    new_parent_cell = parent_cell;
  }

}
//-----------------------------------------------------------------------------
void BisectionRefinement1D::refine(Mesh& refined_mesh,
                                   const Mesh& mesh, bool redistribute)
{
  auto _mesh = reference_to_no_delete_pointer(mesh);
  const CellFunction<bool> cell_markers(_mesh, true);
  BisectionRefinement1D::refine(refined_mesh, mesh, cell_markers, redistribute);
}
//-----------------------------------------------------------------------------
