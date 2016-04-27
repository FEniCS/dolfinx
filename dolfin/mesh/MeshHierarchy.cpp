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

#include <dolfin/parameter/GlobalParameters.h>
#include <dolfin/mesh/DistributedMeshTools.h>
#include <dolfin/mesh/LocalMeshData.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshEntityIterator.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/MeshPartitioning.h>
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
  std::shared_ptr<const Mesh> mesh = _meshes.back();
  dolfin_assert(mesh);

  // Make sure there is a parent MeshHierarchy
  dolfin_assert(_parent != NULL);
  std::shared_ptr<const Mesh> parent_mesh = _parent->_meshes.back();
  dolfin_assert(parent_mesh);

  // Make sure markers are on finest mesh
  dolfin_assert(coarsen_markers.mesh()->id() == mesh->id());

  // FIXME: copy across boundaries in parallel
  std::set<std::size_t> coarsening_vertices;
  if (coarsen_markers.dim() == 0)
  {
    for (VertexIterator v(*mesh); !v.end(); ++v)
      if (coarsen_markers[*v])
        coarsening_vertices.insert(v->global_index());
  }
  else
  {
    // FIXME: assumes "OR"-like behaviour, i.e. if any
    // entity around a vertex is marked, then the vertex is
    // marked. Should this be "AND"-like behaviour, i.e. require
    // all surrounding entities to be marked?
    for (MeshEntityIterator c(*mesh, coarsen_markers.dim());
         !c.end(); ++c)
    {
      if (coarsen_markers[*c])
        for (VertexIterator v(*c); !v.end(); ++v)
          coarsening_vertices.insert(v->global_index());
    }
  }

  // Set up refinement markers to re-refine the parent mesh
  EdgeFunction<bool> edge_markers(parent_mesh, false);
  const std::map<std::size_t, std::size_t>& edge_to_vertex
    = *(_relation->edge_to_global_vertex);

  // Find edges which were previously refined, but now only mark them
  // if not a parent of a "coarsening" vertex
  for (EdgeIterator e(*parent_mesh); !e.end(); ++e)
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

  auto refined_mesh = std::make_shared<Mesh>();
  auto refined_hierarchy = std::make_shared<MeshHierarchy>();
  auto refined_relation = std::make_shared<MeshRelation>();

  // Refine with no redistribution
  PlazaRefinementND::refine(*refined_mesh, *parent_mesh,
                            edge_markers, true, *refined_relation);

  refined_hierarchy->_meshes = _parent->_meshes;
  refined_hierarchy->_meshes.push_back(refined_mesh);

  refined_hierarchy->_parent = _parent;

  refined_hierarchy->_relation = refined_relation;

  return refined_hierarchy;
}
//-----------------------------------------------------------------------------
std::vector<std::size_t> MeshHierarchy::weight() const
{
  // Assign each fine cell a weight of 1.
  // FIXME? Not all fine cells are the same size - possibly weight by size
  std::vector<std::size_t> cell_weights(finest()->num_cells(), 1);

  for (std::size_t level = size() - 1; level > 0; --level)
  {
    const Mesh& mesh = *_meshes[level];
    const Mesh& parent_mesh = *_meshes[level - 1];
    const std::vector<std::size_t> parent_cell
      = mesh.data().array("parent_cell", mesh.topology().dim());
    dolfin_assert(parent_cell.size() == cell_weights.size());
    std::vector<std::size_t> parent_cell_weights(parent_mesh.num_cells(), 0);
    for (unsigned int i = 0; i != cell_weights.size(); ++i)
      parent_cell_weights[parent_cell[i]] += cell_weights[i];

    cell_weights = parent_cell_weights;
  }

  return cell_weights;
}
//-----------------------------------------------------------------------------
std::shared_ptr<Mesh> MeshHierarchy::rebalance() const
{
  // Make a new MeshHierarchy, with the same meshes, but rebalanced across
  // processes.

  // FIXME: this needs to be extended to all meshes in the Hierarchy
  // and reconstruction of the MeshRelations between them... work in progress

#ifndef HAS_SCOTCH
  dolfin_error("MeshHierarchy.cpp",
               "rebalance MeshHierarchy",
               "Rebalancing requires SCOTCH library at present");
#endif

  const Mesh& coarse_mesh = *coarsest();
  if (MPI::size(coarse_mesh.mpi_comm()) == 1)
    dolfin_error("MeshHierarchy.cpp",
                 "rebalance MeshHierarchy", "Not applicable in serial");

  LocalMeshData local_mesh_data(coarse_mesh.mpi_comm());
  local_mesh_data.cell_weight = weight();

  const std::size_t tdim = coarse_mesh.topology().dim();
  local_mesh_data.tdim = tdim;
  const std::size_t gdim = coarse_mesh.geometry().dim();
  local_mesh_data.geometry.dim = gdim;
  local_mesh_data.num_vertices_per_cell = tdim + 1;

  // Cells

  local_mesh_data.num_global_cells = coarse_mesh.size_global(tdim);
  const std::size_t num_local_cells = coarse_mesh.size(tdim);
  local_mesh_data.global_cell_indices.resize(num_local_cells);
  local_mesh_data.cell_vertices.resize(boost::extents[num_local_cells]
                               [local_mesh_data.num_vertices_per_cell]);

  for (CellIterator c(coarse_mesh); !c.end(); ++c)
  {
    const std::size_t cell_index = c->index();
    local_mesh_data.global_cell_indices[cell_index] = c->global_index();
    for (VertexIterator v(*c); !v.end(); ++v)
      local_mesh_data.cell_vertices[cell_index][v.pos()] = v->global_index();
  }

  // Vertices - must be reordered into global order

  const std::size_t num_local_vertices = coarse_mesh.size(0);
  local_mesh_data.geometry.num_global_vertices = coarse_mesh.size_global(0);
  local_mesh_data.geometry.vertex_indices.resize(num_local_vertices);
  for (VertexIterator v(coarse_mesh); !v.end(); ++v)
    local_mesh_data.geometry.vertex_indices[v->index()] = v->global_index();
  local_mesh_data.geometry.vertex_coordinates.resize(boost::extents[num_local_vertices][gdim]);

  const std::vector<double> vertex_coords =
    DistributedMeshTools::reorder_vertices_by_global_indices(coarse_mesh);
  std::copy(vertex_coords.begin(), vertex_coords.end(),
            local_mesh_data.geometry.vertex_coordinates.data());

  std::shared_ptr<Mesh> mesh(new Mesh(coarse_mesh.mpi_comm()));
  const std::string ghost_mode = dolfin::parameters["ghost_mode"];
  MeshPartitioning::build_distributed_mesh(*mesh, local_mesh_data, ghost_mode);

  return mesh;
}
//-----------------------------------------------------------------------------
