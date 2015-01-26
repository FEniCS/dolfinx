// Copyright (C) 2014-2015 Chris Richardson
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
// First Added: 2014-05-21

#include <boost/multi_array.hpp>

#include <vector>
#include <set>
#include <map>

#include <dolfin/common/Timer.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshEntityIterator.h>
#include <dolfin/mesh/MeshRelation.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Edge.h>
#include <dolfin/mesh/Face.h>
#include <dolfin/mesh/Facet.h>
#include <dolfin/mesh/Vertex.h>

#include "PlazaRefinementND.h"
#include "ParallelRefinement.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void PlazaRefinementND::get_simplices
(std::vector<std::vector<std::size_t> >& simplex_set,
 const std::vector<bool>& marked_edges,
 const std::vector<std::size_t>& longest_edge,
 std::size_t tdim)
{
  if (tdim == 2)
  {
    dolfin_assert(longest_edge.size() == 1);
    get_triangles(simplex_set, marked_edges, longest_edge[0]);
  }
  else if (tdim == 3)
  {
    dolfin_assert(longest_edge.size() == 4);
    get_tetrahedra(simplex_set, marked_edges, longest_edge);
  }
}
//-----------------------------------------------------------------------------
void PlazaRefinementND::get_triangles
(std::vector<std::vector<std::size_t> >& tri_set,
 const std::vector<bool>& marked_edges,
 const std::size_t longest_edge)
{
  Timer t0("PLAZA: Get triangles");

  // Longest edge must be marked
  dolfin_assert(marked_edges[longest_edge]);

  tri_set.clear();

  // v0 and v1 are at ends of longest_edge (e2)
  // opposite vertex has same index as longest_edge
  const unsigned int v0 = (longest_edge + 1)%3;
  const unsigned int v1 = (longest_edge + 2)%3;
  const unsigned int v2 = longest_edge;
  const unsigned int e0 = v0 + 3;
  const unsigned int e1 = v1 + 3;
  const unsigned int e2 = v2 + 3;

  // Break each half of triangle into one or two sub-triangles

  if (marked_edges[v0])
  {
    tri_set.push_back(std::vector<std::size_t>{e2, v2, e0});
    tri_set.push_back(std::vector<std::size_t>{e2, e0, v1});
  }
  else
    tri_set.push_back(std::vector<std::size_t>{e2, v2, v1});

  if (marked_edges[v1])
  {
    tri_set.push_back(std::vector<std::size_t>{e2, v2, e1});
    tri_set.push_back(std::vector<std::size_t>{e2, e1, v0});
  }
  else
    tri_set.push_back(std::vector<std::size_t>{e2, v2, v0});

}
//-----------------------------------------------------------------------------
void PlazaRefinementND::get_tetrahedra(
            std::vector<std::vector<std::size_t> >& tet_set,
            const std::vector<bool>& marked_edges,
            const std::vector<std::size_t> longest_edge)
{
  Timer t0("PLAZA: Get tetrahedra");

  tet_set.clear();

  // Connectivity matrix
  // Only need upper triangle, but sometimes it is easier just to insert
  // both entries (j,i) and (i,j).
  boost::multi_array<bool, 2> conn(boost::extents[10][10]);
  std::fill(conn.data(), conn.data() + 100, false);

  // Edge connectivity to vertices (and by extension facets)
  static std::size_t edges[6][2] = {{2, 3},
                                    {1, 3},
                                    {1, 2},
                                    {0, 3},
                                    {0, 2},
                                    {0, 1}};

  // Iterate through cell edges
  for (unsigned int ei = 0; ei != 6; ++ei)
  {
    const unsigned int v0 = edges[ei][0];
    const unsigned int v1 = edges[ei][1];

    if (marked_edges[ei])
    {
      // Connect to edge end vertices

      // Only add upper-triangular connections
      conn[v1][ei + 4] = true;
      conn[v0][ei + 4] = true;

      // Edge has two attached facets in cell
      // which have the same numbering as the
      // vertices which are not in the edge
      for (unsigned int j = 0; j != 2; ++j)
      {
        const std::size_t e_opp = 5 - ei;
        const std::size_t fj = edges[e_opp][j];
        const std::size_t le_j = longest_edge[fj];

        if (le_j == ei)
        {
          const std::size_t fk = edges[e_opp][1 - j];
          const std::size_t le_k = longest_edge[fk];

          // This is longest edge - connect to opposite vertex

          // Only add upper-triangular connection
          conn[fk][ei + 4] = true;

          if (le_k == ei && marked_edges[e_opp])
          {
            // Longest edge of two adjacent facets
            // Join to opposite edge (through centre of tetrahedron)
            // if marked.
            conn[ei + 4][e_opp + 4] = true;
            conn[e_opp + 4][ei + 4] = true;
          }
        }
        else
        {
          // Not longest edge, but marked, so
          // connect back to longest edge of facet
          conn[le_j + 4][ei + 4] = true;
          conn[ei + 4][le_j + 4] = true;
       }
      }
    }
    else
    {
      // No marking on this edge, just connect ends
      conn[v1][v0] = true;
      conn[v0][v1] = true;
    }
  }

  // Iterate through all possible new vertices
  std::vector<std::size_t> facet_set;
  for (std::size_t i = 0; i < 10; ++i)
  {
    for (std::size_t j = i + 1; j < 10; ++j)
      if (conn[i][j])
      {
        facet_set.clear();
        for (std::size_t k = j + 1; k < 10; ++k)
        {
          if (conn[i][k] && conn[j][k])
            facet_set.push_back(k);
        }
        // Note that j>i and k>j. facet_set is in increasing order, so *q > *p.
        // Should never repeat same tetrahedron twice.
        for(auto p = facet_set.begin(); p != facet_set.end(); ++p)
        {
          for(auto q = p + 1; q != facet_set.end(); ++q)
          {
            if(conn[*p][*q])
            {
              std::vector<std::size_t> tet{i, j, *p, *q};
              tet_set.push_back(tet);
            }
          }
        }
      }
  }
}
//-----------------------------------------------------------------------------
std::vector<std::size_t> PlazaRefinementND::face_long_edge(const Mesh& mesh)
{
  Timer t0("PLAZA: Face long edge");

  mesh.init(2);
  std::vector<std::size_t> result(mesh.num_faces());

  // Get longest edge of each face
  for (FaceIterator f(mesh); !f.end(); ++f)
  {
    std::size_t imax = 0;
    std::size_t gimax = 0;
    double max_len = 0.0;

    // Use vertex global index to decide ordering
    // if lengths are equal
    for (EdgeIterator e(*f); !e.end(); ++e)
    {
      const double e_len = e->length();
      const Vertex v(mesh, f->entities(0)[e.pos()]);
      const std::size_t global_i = v.global_index();
      if (e_len > max_len or (e_len == max_len and global_i > gimax))
      {
        max_len = e_len;
        imax = e->index();
        gimax = global_i;
      }
    }
    result[f->index()] = imax;
  }

  return result;
}
//-----------------------------------------------------------------------------
void PlazaRefinementND::enforce_rules(ParallelRefinement& p_ref,
                                      const Mesh& mesh,
                                      const std::vector<std::size_t>& long_edge)
{
  Timer t0("PLAZA: Enforce rules");

  // Enforce rule, that if any edge of a face is marked,
  // longest edge must also be marked

  std::size_t update_count = 1;
  while (update_count != 0)
  {
    update_count = 0;
    p_ref.update_logical_edgefunction();

    for (FaceIterator f(mesh); !f.end(); ++f)
    {
      const std::size_t long_e = long_edge[f->index()];
      if (p_ref.is_marked(long_e))
        continue;
      bool any_marked = false;
      for (EdgeIterator e(*f); !e.end(); ++e)
        any_marked |= p_ref.is_marked(e->index());
      if (any_marked)
      {
        p_ref.mark(long_e);
        ++update_count;
      }
    }
    update_count = MPI::sum(mesh.mpi_comm(), update_count);
  }
}
//-----------------------------------------------------------------------------
void PlazaRefinementND::refine(Mesh& new_mesh, const Mesh& mesh,
                               bool redistribute,
                               bool calculate_parent_facets)
{
  const std::size_t tdim = mesh.topology().dim();
  if (tdim != 2 and tdim != 3)
  {
    dolfin_error("PlazaRefinementND.cpp",
                 "refine mesh",
                 "Topological dimension %d not supported", tdim);
  }

  Timer t0("PLAZA: refine");
  std::vector<std::size_t> long_edge = face_long_edge(mesh);

  ParallelRefinement p_ref(mesh);
  p_ref.mark_all();

  MeshRelation mesh_relation;
  do_refine(new_mesh, mesh, p_ref, long_edge, redistribute,
            calculate_parent_facets, mesh_relation);
}
//-----------------------------------------------------------------------------
void PlazaRefinementND::refine(Mesh& new_mesh, const Mesh& mesh,
                               const MeshFunction<bool>& refinement_marker,
                               bool redistribute,
                               bool calculate_parent_facets)
{
  const std::size_t tdim = mesh.topology().dim();
  if (tdim != 2 and tdim != 3)
  {
    dolfin_error("PlazaRefinementND.cpp",
                 "refine mesh",
                 "Topological dimension %d not supported", tdim);
  }

  Timer t0("PLAZA: refine");
  std::vector<std::size_t> long_edge = face_long_edge(mesh);

  ParallelRefinement p_ref(mesh);
  p_ref.mark(refinement_marker);

  enforce_rules(p_ref, mesh, long_edge);

  MeshRelation mesh_relation;
  do_refine(new_mesh, mesh, p_ref, long_edge, redistribute,
            calculate_parent_facets, mesh_relation);
}
//-----------------------------------------------------------------------------
void PlazaRefinementND::refine(Mesh& new_mesh, const Mesh& mesh,
                               const MeshFunction<bool>& refinement_marker,
                               bool calculate_parent_facets,
                               MeshRelation& mesh_relation)
{
  const std::size_t tdim = mesh.topology().dim();
  if (tdim != 2 and tdim != 3)
  {
    dolfin_error("PlazaRefinementND.cpp",
                 "refine mesh",
                 "Topological dimension %d not supported", tdim);
  }

  Timer t0("PLAZA: refine");
  std::vector<std::size_t> long_edge = face_long_edge(mesh);

  ParallelRefinement p_ref(mesh);
  p_ref.mark(refinement_marker);

  enforce_rules(p_ref, mesh, long_edge);

  do_refine(new_mesh, mesh, p_ref, long_edge, false,
            calculate_parent_facets, mesh_relation);
}
//-----------------------------------------------------------------------------
void PlazaRefinementND::do_refine(Mesh& new_mesh, const Mesh& mesh,
                                  ParallelRefinement& p_ref,
                                  const std::vector<std::size_t>& long_edge,
                                  bool redistribute,
                                  bool calculate_parent_facets,
                                  MeshRelation& mesh_relation)
{
  const std::size_t tdim = mesh.topology().dim();
  const std::size_t num_cell_edges = tdim*3 - 3;
  const std::size_t num_cell_vertices = tdim + 1;

  // Make new vertices in parallel
  p_ref.create_new_vertices();
  const std::map<std::size_t, std::size_t> new_vertex_map
    = *(p_ref.edge_to_new_vertex());

  std::vector<std::size_t> parent_cell;
  std::vector<std::size_t> indices(num_cell_vertices + num_cell_edges);
  std::vector<std::size_t> marked_edge_list;

  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Create vector of indices in the order
    // [vertices][edges], 3+3 in 2D, 4+6 in 3D
    std::size_t j = 0;
    for (VertexIterator v(*cell); !v.end(); ++v)
      indices[j++] = v->global_index();

    marked_edge_list = p_ref.marked_edge_list(*cell);

    if (marked_edge_list.size() == 0)
    {
      indices.resize(num_cell_vertices);
      p_ref.new_cell(indices);
      parent_cell.push_back(cell->index());
    }
    else
    {
      indices.resize(num_cell_vertices + num_cell_edges);
      // Get the marked edge indices for new vertices
      // and make bool vector of marked edges
      std::vector<bool> markers(num_cell_edges, false);
      EdgeIterator e(*cell);
      for (auto &p : marked_edge_list)
      {
        markers[p] = true;
        const std::size_t edge_index = e[p].index();
        auto it = new_vertex_map.find(edge_index);
        dolfin_assert (it != new_vertex_map.end());
        indices[num_cell_vertices + p] = it->second;
      }

      // Need longest edges of each facet in cell local indexing
      std::vector<std::size_t> longest_edge;
      if (tdim == 3)
      {
        for (FaceIterator f(*cell); !f.end(); ++f)
          longest_edge.push_back(long_edge[f->index()]);
      }
      else if (tdim == 2)
        longest_edge.push_back(long_edge[cell->index()]);

      // Convert to cell local index
      for (auto &p : longest_edge)
      {
        for (EdgeIterator ej(*cell); !ej.end(); ++ej)
        {
          if (p == ej->index())
          {
            p = ej.pos();
            break;
          }
        }
      }

      std::vector<std::vector<std::size_t> > simplex_set;
      get_simplices(simplex_set, markers, longest_edge, tdim);

      // Convert from cell local index to mesh index
      for (auto &it : simplex_set)
      {
        for (auto &vit : it)
          vit = indices[vit];
        p_ref.new_cell(it);
        parent_cell.push_back(cell->index());
      }
    }
  }

  const bool serial = (MPI::size(mesh.mpi_comm()) == 1);

  if (serial)
    p_ref.build_local(new_mesh);
  else
    p_ref.partition(new_mesh, redistribute);

  if (serial || !redistribute)
  {
    // Create parent data on new mesh
    std::vector<std::size_t>& new_parent_cell
      = new_mesh.data().create_array("parent_cell", new_mesh.topology().dim());

    new_parent_cell = parent_cell;

    if (calculate_parent_facets)
      set_parent_facet_markers(mesh, new_mesh, new_vertex_map);

    mesh_relation.edge_to_global_vertex = p_ref.edge_to_new_vertex();
  }
}
//-----------------------------------------------------------------------------
void PlazaRefinementND::set_parent_facet_markers(const Mesh& mesh,
                                                 Mesh& new_mesh,
           const std::map<std::size_t, std::size_t>& new_vertex_map)
{
  if (mesh.topology().dim() != 2)
    dolfin_error("PlazaRefinementND.cpp",
                 "set parent facet markers",
                 "Only supported in 2D");

  std::vector<std::size_t>& new_parent_facet
    = new_mesh.data().create_array("parent_facet",
                                   new_mesh.topology().dim() - 1);

  // Make reverse map from new vertex to parent edge
  std::map<std::size_t, std::size_t> reverse_map;
  for (auto &p : new_vertex_map)
    reverse_map.insert(std::make_pair(p.second, p.first));

  // Make a map for every facet in old mesh
  // FIXME: this is fairly ruinous for performance
  std::map<std::pair<std::size_t, std::size_t>, std::size_t> old_fmap;
  for (FacetIterator fold(mesh); !fold.end(); ++fold)
  {
    const unsigned int *v = fold->entities(0);
    const std::size_t v0 = Vertex(mesh, v[0]).global_index();
    const std::size_t v1 = Vertex(mesh, v[1]).global_index();
    std::pair<std::size_t, std::size_t> idx
      = (v0 < v1) ? std::make_pair(v0, v1) : std::make_pair(v1, v0);
    old_fmap.insert(std::make_pair(idx, fold->index()));
  }

  new_mesh.init(new_mesh.topology().dim() - 1);
  new_parent_facet.resize(new_mesh.num_facets());
  for (FacetIterator f(new_mesh); !f.end(); ++f)
  {
    const unsigned int *v = f->entities(0);
    const std::size_t gv0 = Vertex(new_mesh, v[0]).global_index();
    const std::size_t gv1 = Vertex(new_mesh, v[1]).global_index();

    // First look for direct (unrefined) facet
    std::pair<std::size_t, std::size_t> idx
      = (gv0 < gv1) ? std::make_pair(gv0, gv1) : std::make_pair(gv1, gv0);
    auto old_facet_it = old_fmap.find(idx);

    if (old_facet_it != old_fmap.end())
      new_parent_facet[f->index()] = old_facet_it->second;
    else
    {
      // Look for refined facet
      auto map_it0 = reverse_map.find(gv0);
      auto map_it1 = reverse_map.find(gv1);
      if (map_it0 != reverse_map.end())
      {
        const Facet old_facet(mesh, map_it0->second);
        const unsigned int *old_verts = old_facet.entities(0);
        const std::size_t fv0 = Vertex(mesh, old_verts[0]).global_index();
        const std::size_t fv1 = Vertex(mesh, old_verts[1]).global_index();
        if (fv0 == gv1 or fv1 == gv1)
          new_parent_facet[f->index()] = map_it0->second;
      }
      else if (map_it1 != reverse_map.end())
      {
        const Facet old_facet(mesh, map_it1->second);
        const unsigned int *old_verts = old_facet.entities(0);
        const std::size_t fv0 = Vertex(mesh, old_verts[0]).global_index();
        const std::size_t fv1 = Vertex(mesh, old_verts[1]).global_index();
        if (fv0 == gv0 or fv1 == gv0)
          new_parent_facet[f->index()] = map_it1->second;
      }
      else
        new_parent_facet[f->index()] = std::numeric_limits<std::size_t>::max();
    }
  }
}
