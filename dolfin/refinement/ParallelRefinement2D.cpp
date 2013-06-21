// Copyright (C) 2012 Chris Richardson
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
//
// First Added: 2012-12-19
// Last Changed: 2013-05-12

#include <vector>
#include <map>
#include <boost/unordered_map.hpp>
#include <boost/multi_array.hpp>

#include <dolfin/common/types.h>
#include <dolfin/common/MPI.h>
#include <dolfin/common/Timer.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Edge.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/refinement/ParallelRefinement.h>
#include "ParallelRefinement2D.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
bool ParallelRefinement2D::length_compare(std::pair<double, std::size_t> a,
                                          std::pair<double, std::size_t> b)
{
  return (a.first > b.first);
}
//-----------------------------------------------------------------------------
void ParallelRefinement2D::generate_reference_edges(const Mesh& mesh,
                                          std::vector<std::size_t>& ref_edge)
{
  std::size_t D = mesh.topology().dim();
  ref_edge.resize(mesh.size(D));

  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    std::size_t cell_index = cell->index();

    std::vector<std::pair<double, std::size_t> > lengths;
    EdgeIterator celledge(*cell);
    for (std::size_t i = 0; i < 3; i++)
      lengths.push_back(std::make_pair(celledge[i].length(), i));
    std::sort(lengths.begin(), lengths.end(), length_compare);

    // For now - just pick longest edge - this is not the Carstensen
    // algorithm, which tries to pair edges off. Because that is more
    // difficult in parallel, it is not implemented yet.
    const std::size_t edge_index = lengths[0].second;
    ref_edge[cell_index] = edge_index;
  }
}
//-----------------------------------------------------------------------------
void ParallelRefinement2D::refine(Mesh& new_mesh, const Mesh& mesh,
                                  bool redistribute)
{
  if (MPI::num_processes()==1)
  {
    dolfin_error("ParallelRefinement2D.cpp",
                 "refine mesh",
                 "Only works in parallel");
  }

  const std::size_t tdim = mesh.topology().dim();
  if (tdim != 2)
  {
    dolfin_error("ParallelRefinement2D.cpp",
                 "refine mesh",
                 "Only works in 2D");
  }

  // Ensure connectivity is there
  mesh.init(tdim - 1, tdim);

  // Create an object to hold most of the refinement information
  ParallelRefinement p(mesh);

  // Mark all edges, and create new vertices
  p.mark_all();
  p.create_new_vertices();
  const std::map<std::size_t, std::size_t>& edge_to_new_vertex
    = p.edge_to_new_vertex();

  // Convenienence iterator
  std::map<std::size_t, std::size_t>::const_iterator it;

  // Generate new topology
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    EdgeIterator e(*cell);
    VertexIterator v(*cell);

    const std::size_t v0 = v[0].global_index();
    const std::size_t v1 = v[1].global_index();
    const std::size_t v2 = v[2].global_index();

    it = edge_to_new_vertex.find(e[0].index());
    dolfin_assert(it != edge_to_new_vertex.end());
    const std::size_t e0 = it->second;

    it = edge_to_new_vertex.find(e[1].index());
    dolfin_assert(it != edge_to_new_vertex.end());
    const std::size_t e1 = it->second;

    it = edge_to_new_vertex.find(e[2].index());
    dolfin_assert(it != edge_to_new_vertex.end());
    const std::size_t e2 = it->second;

    //const std::size_t e0 = edge_to_new_vertex[e[0].index()];
    //const std::size_t e1 = edge_to_new_vertex[e[1].index()];
    //const std::size_t e2 = edge_to_new_vertex[e[2].index()];

    p.new_cell(v0, e2, e1);
    p.new_cell(e2, v1, e0);
    p.new_cell(e1, e0, v2);
    p.new_cell(e0, e1, e2);
  }

  p.partition(new_mesh, redistribute);
}
//-----------------------------------------------------------------------------
void ParallelRefinement2D::refine(Mesh& new_mesh, const Mesh& mesh,
                                  const MeshFunction<bool>& refinement_marker,
                                  bool redistribute)
{
  if (MPI::num_processes()==1)
  {
    dolfin_error("ParallelRefinement2D.cpp",
                 "refine mesh",
                 "Only works in parallel");
  }

  const std::size_t tdim = mesh.topology().dim();
  if (tdim != 2)
  {
    // Note: gdim may be 3
    dolfin_error("ParallelRefinement2D.cpp",
                 "refine mesh",
                 "Only works in 2D");
  }

  // Ensure connectivity is there
  mesh.init(tdim - 1, tdim);

  // Create a class to hold most of the refinement information
  ParallelRefinement p(mesh);

  // This refinement algorithm creates a 'reference' edge for each
  // cell.  In this case, the reference edge is the longest edge.  Any
  // cell with edges marked for bisection, must also bisect the
  // reference edge.

  // Vector over all cells - the reference edge is the cell's edge (0,
  // 1 or 2) which always must split, if any edge splits in the cell
  std::vector<std::size_t> ref_edge;
  generate_reference_edges(mesh, ref_edge);

  // Set marked edges from marked cells
  p.mark(refinement_marker);

  // Mark reference edges of cells with any marked edge
  // and repeat until no more marking takes place

  std::size_t update_count = 1;

  while (update_count != 0)
  {
    update_count = 0;

    // Transmit values between processes - could be streamlined
    p.update_logical_edgefunction();
    for (CellIterator cell(mesh); !cell.end(); ++cell)
    {
      std::size_t n_marked = p.marked_edge_count(*cell);
      EdgeIterator edge(*cell);
      std::size_t ref_edge_index = edge[ref_edge[cell->index()]].index();
      if (n_marked != 0 && p.is_marked(ref_edge_index) == false)
      {
        p.mark(ref_edge_index);
        update_count++;
      }
    }
    update_count = MPI::sum(update_count);
  }

  // Generate new vertices from marked edges, and assign global vertex
  // index map.
  p.create_new_vertices();
  const std::map<std::size_t, std::size_t>& edge_to_new_vertex
    = p.edge_to_new_vertex();

  // Convenienence iterator
  std::map<std::size_t, std::size_t>::const_iterator it;

  // Stage 4 - do refinement
  // FIXME - keep reference edges somehow?...

  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    std::size_t rgb_count = p.marked_edge_count(*cell);
    EdgeIterator e(*cell);
    VertexIterator v(*cell);
    const std::size_t ref = ref_edge[cell->index()];
    const std::size_t i0 = ref;
    const std::size_t i1 = (ref + 1)%3;
    const std::size_t i2 = (ref + 2)%3;
    const std::size_t v0 = v[i0].global_index();
    const std::size_t v1 = v[i1].global_index();
    const std::size_t v2 = v[i2].global_index();

    //const std::size_t e0 = edge_to_new_vertex[e[i0].index()];
    //const std::size_t e1 = edge_to_new_vertex[e[i1].index()];
    //const std::size_t e2 = edge_to_new_vertex[e[i2].index()];

    if (rgb_count == 0) //straight copy of cell (1->1)
      p.new_cell(*cell);
    else if (rgb_count == 1) // "green" refinement (1->2)
    {
      // Always splitting the reference edge (only)
      it = edge_to_new_vertex.find(e[i0].index());
      dolfin_assert(it != edge_to_new_vertex.end());
      const std::size_t e0 = it->second;

      p.new_cell(e0, v0, v1);
      p.new_cell(e0, v2, v0);
    }
    else if (rgb_count == 2) // "blue" refinement (1->3) left or right
    {
      // FIXME: more possibilities here - need to do more tests
      if (p.is_marked(e[i2].index()))
      {
        it = edge_to_new_vertex.find(e[i0].index());
        dolfin_assert(it != edge_to_new_vertex.end());
        const std::size_t e0 = it->second;

        it = edge_to_new_vertex.find(e[i2].index());
        dolfin_assert(it != edge_to_new_vertex.end());
        const std::size_t e2 = it->second;

        p.new_cell(e2, v1, e0);
        p.new_cell(e2, e0, v0);
        p.new_cell(e0, v2, v0);
      }
      else if (p.is_marked(e[i1].index()))
      {
        it = edge_to_new_vertex.find(e[i0].index());
        dolfin_assert(it != edge_to_new_vertex.end());
        const std::size_t e0 = it->second;

        it = edge_to_new_vertex.find(e[i1].index());
        dolfin_assert(it != edge_to_new_vertex.end());
        const std::size_t e1 = it->second;

        p.new_cell(e0, v0, v1);
        p.new_cell(e1, e0, v2);
        p.new_cell(e1, v0, e0);
      }
    }
    else if (rgb_count == 3) // "red" refinement - all split (1->4) cells
    {
      it = edge_to_new_vertex.find(e[i0].index());
      dolfin_assert(it != edge_to_new_vertex.end());
      const std::size_t e0 = it->second;

      it = edge_to_new_vertex.find(e[i1].index());
      dolfin_assert(it != edge_to_new_vertex.end());
      const std::size_t e1 = it->second;

      it = edge_to_new_vertex.find(e[i2].index());
      dolfin_assert(it != edge_to_new_vertex.end());
      const std::size_t e2 = it->second;

      p.new_cell(v0, e2, e1);
      p.new_cell(e2, v1, e0);
      p.new_cell(e1, e0, v2);
      p.new_cell(e0, e1, e2);
    }
  }

  // Call partitioning from within ParallelRefinement class
  p.partition(new_mesh, redistribute);
}
//-----------------------------------------------------------------------------
