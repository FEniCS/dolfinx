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

#include<map>

#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/MeshRelation.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Edge.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/refinement/PlazaRefinementND.h>

#include "MeshHierarchy.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
std::shared_ptr<const MeshHierarchy> MeshHierarchy::refine(
                           const MeshFunction<bool>& markers) const
{
  std::shared_ptr<Mesh> refined_mesh(new Mesh);
  std::shared_ptr<MeshHierarchy> refined_hierarchy(new MeshHierarchy);
  std::shared_ptr<MeshRelation> refined_relation(new MeshRelation);

  // Make sure markers are on correct mesh, i.e. finest of hierarchy
  dolfin_assert(markers.mesh()->id() == _meshes.back()->id());

  // Refine with no redistribution
  PlazaRefinementND::refine(*refined_mesh, *_meshes.back(),
                            markers, true, *refined_relation);

  refined_hierarchy->_meshes = _meshes;
  refined_hierarchy->_meshes.push_back(refined_mesh);

  refined_hierarchy->_parent = std::make_shared<const MeshHierarchy>(*this);

  refined_hierarchy->_relation = refined_relation;

  return refined_hierarchy;
}
//-----------------------------------------------------------------------------
std::shared_ptr<const MeshHierarchy>
MeshHierarchy::coarsen(const MeshFunction<bool>& coarsen_markers) const
{
  const Mesh& mesh = *(_meshes.back());

  // Make sure there is a parent MeshHierarchy
  dolfin_assert(_parent != NULL);
  const Mesh& parent_mesh = *(_parent->_meshes.back());

  // Make sure markers are on finest mesh
  dolfin_assert(coarsen_markers.mesh()->id() == mesh.id());
  // Markers must be a VertexFunction (for now)
  // FIXME: generalise
  dolfin_assert(coarsen_markers.dim() == 0);

  // FIXME: copy across boundaries in parallel
  std::set<std::size_t> coarsening_vertices;
  for (VertexIterator v(mesh); !v.end(); ++v)
    if (coarsen_markers[*v])
      coarsening_vertices.insert(v->global_index());

  // Set up refinement markers to re-refine the parent mesh
  EdgeFunction<bool> edge_markers(parent_mesh, false);
  const std::map<std::size_t, std::size_t>& edge_to_vertex
    = *(_relation->edge_to_global_vertex);

  // Find edges which were previously refined, but now only mark them
  // if not a parent of a "coarsening" vertex
  for (EdgeIterator e(parent_mesh); !e.end(); ++e)
  {
    auto edge_it = edge_to_vertex.find(e->index());
    if (edge_it != edge_to_vertex.end())
    {
      // Previously refined edge: find child vertex
      const std::size_t child_vertex_global_index = edge_it->second;
      if (coarsening_vertices.find(child_vertex_global_index)
          == coarsening_vertices.end())
      {
        // Not a "coarsening" vertex, so mark edge for refinement
        edge_markers[*e] = true;
      }
    }
  }

  std::shared_ptr<Mesh> refined_mesh(new Mesh);
  std::shared_ptr<MeshHierarchy> refined_hierarchy(new MeshHierarchy);
  std::shared_ptr<MeshRelation> refined_relation(new MeshRelation);

  // Refine with no redistribution
  PlazaRefinementND::refine(*refined_mesh, parent_mesh,
                            edge_markers, true, *refined_relation);

  refined_hierarchy->_meshes = _parent->_meshes;
  refined_hierarchy->_meshes.push_back(refined_mesh);

  refined_hierarchy->_parent = _parent;

  refined_hierarchy->_relation = refined_relation;

  return refined_hierarchy;
}
//-----------------------------------------------------------------------------
