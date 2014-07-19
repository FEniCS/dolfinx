// Copyright (C) 2008-2013 Anders Logg, Ola Skavhaug and Garth N. Wells
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
// Modified by Niclas Jansson 2009
// Modified by Garth N. Wells 2010-2012
// Modified by Mikael Mortensen, 2012.
// Modified by Joachim B Haga, 2012
// Modified by Martin Alnaes, 2013
//
// First added:  2008-08-12
// Last changed: 2013-01-08

#include <cstdlib>
#include <random>
#include <utility>
#include <ufc.h>

#include <dolfin/common/Timer.h>
#include <dolfin/graph/BoostGraphOrdering.h>
#include <dolfin/graph/GraphBuilder.h>
#include <dolfin/graph/SCOTCH.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/DistributedMeshTools.h>
#include <dolfin/mesh/Facet.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshEntityIterator.h>
#include <dolfin/mesh/PeriodicBoundaryComputation.h>
#include <dolfin/mesh/Restriction.h>
#include <dolfin/mesh/SubDomain.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/parameter/GlobalParameters.h>
#include "DofMap.h"
#include "DofMapBuilder.h"

#include <dolfin/common/utils.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
void DofMapBuilder::build(DofMap& dofmap, const Mesh& mesh,
                          std::shared_ptr<const SubDomain> constrained_domain,
                          std::shared_ptr<const Restriction> restriction)
{
  dolfin_assert(dofmap._ufc_dofmap);

  // Start timer for dofmap initialization
  Timer t0("Init dofmap");

  // Check that mesh has been ordered
  if (!mesh.ordered())
  {
     dolfin_error("DofMapBuiler.cpp",
                  "create mapping of degrees of freedom",
                  "Mesh is not ordered according to the UFC numbering convention. "
                  "Consider calling mesh.order()");
  }

  // Check if dofmap is distributed (based on mesh MPI communicator)
  const bool distributed = MPI::size(mesh.mpi_comm()) > 1;

  // Check if UFC dofmap should not be re-ordered (only applicable in serial)
  const bool reorder_ufc = dolfin::parameters["reorder_dofs_serial"];
  const bool reorder = (distributed or reorder_ufc) ? true : false;

  // Sanity checks on UFC dofmap
  const std::size_t D = mesh.topology().dim();
  dolfin_assert(dofmap._ufc_dofmap);
  dolfin_assert(dofmap._ufc_dofmap->geometric_dimension()
                == mesh.geometry().dim());
  dolfin_assert(dofmap._ufc_dofmap->topological_dimension() == D);

  // Determine and set dof block size (block size must be 1 if UFC map
  // is not re-ordered)
  const std::size_t bs = reorder ? compute_blocksize(*dofmap._ufc_dofmap) : 1;
  dofmap.block_size = bs;

  // Compute a 'node' dofmap based on a UFC dofmap. Returns:
  // - node dofmap (node_dofmap)
  // - local-to-global node indices (node_local_to_global)
  // - UFC local indices to new local indices
  //   (node_ufc_local_to_local, constrained maps only)
  // - Number of mesh entities (global), which may differ from that
  //   from the mesh if the dofmap is constrained
  std::vector<std::size_t> node_local_to_global0;
  std::vector<std::vector<la_index>> node_graph0;
  std::vector<int> node_ufc_local_to_local0;
  std::shared_ptr<const ufc::dofmap> ufc_node_dofmap;
  if (!constrained_domain)
  {
    ufc_node_dofmap
      = build_ufc_node_graph(node_graph0, node_local_to_global0,
                              dofmap._num_mesh_entities_global,
                              dofmap._ufc_dofmap,
                              mesh, constrained_domain, bs);
    dolfin_assert(ufc_node_dofmap);
  }
  else
  {
    ufc_node_dofmap
      = build_ufc_node_graph_constrained(node_graph0,
                                          node_local_to_global0,
                                          node_ufc_local_to_local0,
                                          dofmap._num_mesh_entities_global,
                                          dofmap._ufc_dofmap,
                                          mesh, constrained_domain,
                                          bs);
  }

  // Set global dimension
  dofmap._global_dimension
    = dofmap._ufc_dofmap->global_dimension(dofmap._num_mesh_entities_global);

  // Compute local UFC indices of any 'global' dofs, and re-map if
  // required, e.g., in case that dofmap is periodic
  const std::vector<std::size_t> num_mesh_entities_local
    = compute_num_mesh_entities_local(mesh, *dofmap._ufc_dofmap);
  std::set<std::size_t> global_nodes0
    = compute_global_dofs(ufc_node_dofmap, num_mesh_entities_local);
  if (!node_ufc_local_to_local0.empty())
  {
    std::set<std::size_t> remapped_global_nodes;
    for (auto node : global_nodes0)
    {
      dolfin_assert(node < node_ufc_local_to_local0.size());
      remapped_global_nodes.insert(node_ufc_local_to_local0[node]);
    }
    global_nodes0 = remapped_global_nodes;
  }

  // Re-order and switch to local indexing in dofmap when distributed
  // for process locality and set local_range
  if (reorder)
  {
    // Mark boundary nodes. Boundary nodes are assigned a random
    // positive integer, interior nodes are marked as -1.
    std::vector<int> boundary_nodes;
    compute_boundary_nodes(boundary_nodes, node_graph0,
                           node_local_to_global0.size(), *ufc_node_dofmap,
                           mesh, MPI::rank(mesh.mpi_comm()));

    // Compute
    // (a) owned and shared nodes (and owned and un-owned):
    //    -1: unowned, 0: owned and shared, 1: owned and not shared;
    // (b) map from shared node to sharing processes; and
    // (c) set of all processes that share dofs with this process
    std::vector<short int> node_ownership0;
    std::unordered_map<int, std::vector<int>> shared_node_to_processes0;
    const int num_owned_nodes
      = compute_node_ownership(node_ownership0,
                               shared_node_to_processes0,
                               dofmap._neighbours,
                               node_graph0,
                               boundary_nodes, global_nodes0,
                               node_local_to_global0, mesh);

    // Set global offset for dofs owned by this process, and the local
    // ownership size
    const std::size_t node_offset_global = MPI::global_offset(mesh.mpi_comm(),
                                                              num_owned_nodes,
                                                              true);
    dofmap._global_offset = bs*node_offset_global;
    dofmap._local_ownership_size = bs*num_owned_nodes;

    // Sanity check
    dolfin_assert(MPI::sum(mesh.mpi_comm(),
                           (std::size_t) dofmap._local_ownership_size)
                  == dofmap._global_dimension);

    // Compute node re-ordering for process index locality and spatial
    // locality within a process, including
    // (a) Old-to-new node indices (local)
    // (b) Owning process for nodes that are not owned by this process
    // (c) New local node index to new global node index
    // (d) Old local node index to new local node index
    std::vector<int> node_old_to_new_local;
    std::vector<std::size_t> node_local_to_global1;
    compute_node_reodering(dofmap._local_to_global_unowned,
                           dofmap._off_process_owner,
                           node_local_to_global1,
                           node_old_to_new_local,
                           node_local_to_global0,
                           node_graph0, node_ownership0, global_nodes0,
                           mesh.mpi_comm());

    // Update UFC-locla-to-local map to account for re-ordering
    if (constrained_domain)
    {
      const std::size_t num_ufc_node_local = node_ufc_local_to_local0.size();
      dofmap._ufc_local_to_local.resize(num_ufc_node_local);
      for (std::size_t i = 0; i < num_ufc_node_local; ++i)
      {
        const int old_node = node_ufc_local_to_local0[i];
        const int new_node = node_old_to_new_local[old_node];
        dofmap._ufc_local_to_local[i] = new_node;
      }
    }
    else
    {
      // UFC dofmap was not altered, old_to_new is same as UFC-to-new
      dofmap._ufc_local_to_local = node_old_to_new_local;
    }

    // Update shared_nodes for node reordering
    dofmap._shared_nodes.clear();
    for (auto it = shared_node_to_processes0.begin();
         it != shared_node_to_processes0.end(); ++ it)
    {
      const int new_node = node_old_to_new_local[it->first];
      dofmap._shared_nodes[new_node] = it->second;
    }

    // Build dofmap from original node 'dof' map and applying the
    // 'old_to_new_local' map for the re-ordered node indices
    build_dofmap(dofmap._dofmap, node_graph0, node_old_to_new_local, bs);
  }
  else
  {
    // UFC dofmap has not been re-ordered
    dolfin_assert(!distributed);
    dofmap._dofmap = node_graph0;
    dofmap._ufc_local_to_local = node_ufc_local_to_local0;
    if (dofmap._ufc_local_to_local.empty() && dofmap._ufc_dofmap->num_sub_dofmaps() > 0)
    {
      dofmap._ufc_local_to_local.resize(dofmap._global_dimension);
      for (std::size_t i = 0; i < dofmap._ufc_local_to_local.size(); ++i)
        dofmap._ufc_local_to_local[i] = i;
    }

    dofmap._global_offset = 0;
    dofmap._local_ownership_size = dofmap._global_dimension;

    dofmap._local_to_global_unowned.clear();
    dofmap._off_process_owner.clear();
    dofmap._shared_nodes.clear();
 }

  // Clear ufc_local-to-local map if dofmap has no sub-maps
  if (dofmap._ufc_dofmap->num_sub_dofmaps() == 0)
    std::vector<int>().swap(dofmap._ufc_local_to_local);
}
//-----------------------------------------------------------------------------
void
DofMapBuilder::build_sub_map_view(DofMap& sub_dofmap,
                                  const DofMap& parent_dofmap,
                                  const std::vector<std::size_t>& component,
                                  const Mesh& mesh)
{
  // Note: Ownership range is set to zero since dofmap is a view
  dolfin_assert(!component.empty());

  // Convenience reference to parent UFC dofmap
  dolfin_assert(parent_dofmap._ufc_dofmap);
  const ufc::dofmap& parent_ufc_dofmap = *parent_dofmap._ufc_dofmap;

  // Generate and number required mesh entities for local UFC map
  const std::vector<std::size_t> num_mesh_entities_local
    = compute_num_mesh_entities_local(mesh, parent_ufc_dofmap);

  // Initialise UFC offset from parent
  std::size_t ufc_offset = parent_dofmap._ufc_offset;

  // Extract local UFC sub-dofmap from parent and update offset
  sub_dofmap._ufc_dofmap
    = extract_ufc_sub_dofmap(parent_ufc_dofmap,
                             ufc_offset, component,
                             num_mesh_entities_local);
  dolfin_assert(sub_dofmap._ufc_dofmap);

  // Set UFC sub-dofmap offset
  sub_dofmap._ufc_offset = ufc_offset;

  // Build local UFC-based dof map for sub-dofmap
  build_local_ufc_dofmap(sub_dofmap._dofmap, *sub_dofmap._ufc_dofmap, mesh);

  // Add offset to local UFC dofmap
  for (std::size_t i = 0; i < sub_dofmap._dofmap.size(); ++i)
    for (std::size_t j = 0; j < sub_dofmap._dofmap[i].size(); ++j)
      sub_dofmap._dofmap[i][j] += ufc_offset;

  // Store number of global mesh entities and set global dimension
  sub_dofmap._num_mesh_entities_global=parent_dofmap._num_mesh_entities_global;
  dolfin_assert(!sub_dofmap._num_mesh_entities_global.empty());
  sub_dofmap._global_dimension
    = sub_dofmap._ufc_dofmap->global_dimension(sub_dofmap._num_mesh_entities_global);

  // Copy data from parent
  sub_dofmap._local_to_global_unowned = parent_dofmap._local_to_global_unowned;
  sub_dofmap._off_process_owner = parent_dofmap._off_process_owner;
  sub_dofmap._shared_nodes = parent_dofmap._shared_nodes;
  sub_dofmap._neighbours = parent_dofmap._neighbours;
  sub_dofmap.block_size = parent_dofmap.block_size;

  //sub_dofmap._global_offset = parent_dofmap._global_offset;
  //sub_dofmap._local_ownership_size = parent_dofmap._local_ownership_size;

  // Store UFC local to re-ordered local if submap has any submaps
  if (sub_dofmap._ufc_dofmap->num_sub_dofmaps() > 0)
    sub_dofmap._ufc_local_to_local = parent_dofmap._ufc_local_to_local;
  else
    sub_dofmap._ufc_local_to_local.clear();

  std::cout << "Size of local-to-local: " << parent_dofmap._ufc_local_to_local.size() << std::endl;
  if (parent_dofmap._ufc_local_to_local.empty())
  {
    dolfin_error("DofMapBuilder.cpp",
                 "build sub-dofmap view",
                 "Re-ordering map not available. It may be been cleared by the user");
  }

  // Map to re-ordered dofs
  const std::vector<int>& local_to_local = parent_dofmap._ufc_local_to_local;
  const std::size_t bs = parent_dofmap.block_size;
  for (auto cell_map = sub_dofmap._dofmap.begin();
       cell_map != sub_dofmap._dofmap.end(); ++cell_map)
  {
    for (auto dof = cell_map->begin(); dof != cell_map->end(); ++dof)
    {
      const std::div_t  div = std::div((int) *dof, (int) local_to_local.size());
      const std::size_t node = div.rem;
      const std::size_t component = div.quot;

        // Get dof from UFC local-to-local map
      dolfin_assert(node < local_to_local.size());
      const std::size_t current_dof = bs*local_to_local[node] + component;

      // Set dof index in transformed dofmap
      *dof = current_dof;
    }
  }
}
//-----------------------------------------------------------------------------
std::size_t DofMapBuilder::build_constrained_vertex_indices(
  const Mesh& mesh,
  const std::map<unsigned int,
  std::pair<unsigned int, unsigned int>>& slave_to_master_vertices,
  std::vector<std::size_t>& modified_vertex_indices_global)
{
  // MPI communicator
  const MPI_Comm mpi_comm = mesh.mpi_comm();

  // Get vertex sharing information (local index, [(sharing process p,
  // local index on p)])
  const std::unordered_map<unsigned int, std::
                             vector<std::pair<unsigned int, unsigned int>>>
    shared_vertices = DistributedMeshTools::compute_shared_entities(mesh, 0);

   // Mark shared vertices
  std::vector<bool> vertex_shared(mesh.num_vertices(), false);
  std::unordered_map<unsigned int, std::vector<std::pair<unsigned int,
      unsigned int>>>::const_iterator shared_vertex;
  for (shared_vertex = shared_vertices.begin();
       shared_vertex != shared_vertices.end(); ++shared_vertex)
  {
    dolfin_assert(shared_vertex->first < vertex_shared.size());
    vertex_shared[shared_vertex->first] = true;
  }

  // Mark slave vertices
  std::vector<bool> slave_vertex(mesh.num_vertices(), false);
  std::map<unsigned int, std::pair<unsigned int,
                                   unsigned int>>::const_iterator slave;
  for (slave = slave_to_master_vertices.begin();
       slave != slave_to_master_vertices.end(); ++slave)
  {
    dolfin_assert(slave->first < slave_vertex.size());
    slave_vertex[slave->first] = true;
  }

  // MPI process number
  const std::size_t proc_num = MPI::rank(mesh.mpi_comm());

  // Communication data structures
  std::vector<std::vector<std::size_t>>
    new_shared_vertex_indices(MPI::size(mesh.mpi_comm()));

  // Compute modified global vertex indices
  std::size_t new_index = 0;
  modified_vertex_indices_global
    = std::vector<std::size_t>(mesh.num_vertices(),
                               std::numeric_limits<std::size_t>::max());
  for (VertexIterator vertex(mesh); !vertex.end(); ++vertex)
  {
    const std::size_t local_index = vertex->index();
    if (slave_vertex[local_index])
    {
      // Do nothing, will get new master index later
    }
    else if (vertex_shared[local_index])
    {
      // If shared, let lowest rank process number the vertex
      auto it = shared_vertices.find(local_index);
      dolfin_assert(it != shared_vertices.end());
      const std::vector<std::pair<unsigned int, unsigned int>>& sharing_procs
        = it->second;

      // Figure out if this is the lowest rank process sharing the
      // vertex
      std::vector<std::pair<unsigned int, unsigned int>>::const_iterator
       min_sharing_rank = std::min_element(sharing_procs.begin(),
                                           sharing_procs.end());
      std::size_t _min_sharing_rank = proc_num + 1;
      if (min_sharing_rank != sharing_procs.end())
        _min_sharing_rank = min_sharing_rank->first;

      if (proc_num <= _min_sharing_rank)
      {
        // Re-number vertex
        modified_vertex_indices_global[vertex->index()] = new_index;

        // Add to list to communicate
        std::vector<std::pair<unsigned int, unsigned int>>::const_iterator p;
        for (p = sharing_procs.begin(); p != sharing_procs.end(); ++p)
        {
          dolfin_assert(p->first < new_shared_vertex_indices.size());

          // Local index on remote process
          new_shared_vertex_indices[p->first].push_back(p->second);

          // Modified global index
          new_shared_vertex_indices[p->first].push_back(new_index);
        }

        new_index++;
      }
    }
    else
      modified_vertex_indices_global[vertex->index()] = new_index++;
  }

  // Send number of owned entities to compute offeset
  std::size_t offset = MPI::global_offset(mpi_comm, new_index, true);

  // Add process offset to modified indices
  for (std::size_t i = 0; i < modified_vertex_indices_global.size(); ++i)
    modified_vertex_indices_global[i] += offset;

  // Add process offset to shared vertex indices before sending
  for (std::size_t p = 0; p < new_shared_vertex_indices.size(); ++p)
    for (std::size_t i = 1; i < new_shared_vertex_indices[p].size(); i += 2)
      new_shared_vertex_indices[p][i] += offset;

  // Send/receive new indices for shared vertices
  std::vector<std::vector<std::size_t>> received_vertex_data;
  MPI::all_to_all(mesh.mpi_comm(), new_shared_vertex_indices,
                  received_vertex_data);

  // Set index for shared vertices that have been numbered by another
  // process
  for (std::size_t p = 0; p < received_vertex_data.size(); ++p)
  {
    const std::vector<std::size_t>& received_vertex_data_p
      = received_vertex_data[p];
    for (std::size_t i = 0; i < received_vertex_data_p.size(); i += 2)
    {
      const unsigned int local_index = received_vertex_data_p[i];
      const std::size_t recv_new_index = received_vertex_data_p[i + 1];

      dolfin_assert(local_index < modified_vertex_indices_global.size());
      modified_vertex_indices_global[local_index] = recv_new_index;
    }
  }

  // Request master vertex index from master owner
  std::vector<std::vector<std::size_t>> master_send_buffer(MPI::size(mpi_comm));
  std::vector<std::vector<std::size_t>> local_slave_index(MPI::size(mpi_comm));
  std::map<unsigned int,
           std::pair<unsigned int, unsigned int>>::const_iterator master;
  for (master = slave_to_master_vertices.begin();
       master != slave_to_master_vertices.end(); ++master)
  {
    const unsigned int local_index = master->first;
    const unsigned int master_proc = master->second.first;
    const unsigned int remote_master_local_index = master->second.second;
    dolfin_assert(master_proc < local_slave_index.size());
    dolfin_assert(master_proc < master_send_buffer.size());
    local_slave_index[master_proc].push_back(local_index);
    master_send_buffer[master_proc].push_back(remote_master_local_index);
  }

  // Send/receive master local indices for slave vertices
  std::vector<std::vector<std::size_t>> received_slave_vertex_indices;
  MPI::all_to_all(mpi_comm, master_send_buffer,
                  received_slave_vertex_indices);

  // Send back new master vertex index
  std::vector<std::vector<std::size_t>> master_vertex_indices(MPI::size(mpi_comm));
  for (std::size_t p = 0; p < received_slave_vertex_indices.size(); ++p)
  {
    const std::vector<std::size_t>& local_master_indices
      = received_slave_vertex_indices[p];
    for (std::size_t i = 0; i < local_master_indices.size(); ++i)
    {
      std::size_t master_local_index = local_master_indices[i];
      dolfin_assert(master_local_index < modified_vertex_indices_global.size());
      const std::size_t new_index = modified_vertex_indices_global[master_local_index];
      master_vertex_indices[p].push_back(new_index);
    }
  }

  // Send/receive new global master indices for slave vertices
  std::vector<std::vector<std::size_t>> received_new_slave_vertex_indices;
  MPI::all_to_all(mpi_comm, master_vertex_indices,
                  received_new_slave_vertex_indices);

  // Set index for slave vertices
  for (std::size_t p = 0; p < received_new_slave_vertex_indices.size(); ++p)
  {
    const std::vector<std::size_t>& new_indices
      = received_new_slave_vertex_indices[p];
    const std::vector<std::size_t>& local_indices = local_slave_index[p];
    for (std::size_t i = 0; i < new_indices.size(); ++i)
    {
      const std::size_t local_index = local_indices[i];
      const std::size_t new_global_index   = new_indices[i];

      dolfin_assert(local_index < modified_vertex_indices_global.size());
      modified_vertex_indices_global[local_index] = new_global_index;
    }
  }

  // Send new indices to process that share a vertex but were not
  // responsible for re-numbering
  return MPI::sum(mpi_comm, new_index);
}
//-----------------------------------------------------------------------------
void DofMapBuilder::build_local_ufc_dofmap(
  std::vector<std::vector<dolfin::la_index>>& dofmap,
  const ufc::dofmap& ufc_dofmap,
  const Mesh& mesh)
{
  // Topological dimension
  const std::size_t D = mesh.topology().dim();

  // Generate and number required mesh entities (locally)
  std::vector<std::size_t> num_mesh_entities(D + 1, 0);
  for (std::size_t d = 0; d <= D; ++d)
  {
    if (ufc_dofmap.needs_mesh_entities(d))
    {
      mesh.init(d);
      num_mesh_entities[d] = mesh.num_entities(d);
    }
  }

  // Create UFC cell and allocate memory
  ufc::cell ufc_cell;
  ufc_cell.entity_indices.resize(D + 1);

  // Build dofmap from ufc::dofmap
  dofmap.resize(mesh.num_cells(),
                std::vector<la_index>(ufc_dofmap.local_dimension()));
  std::vector<std::size_t> dof_holder(ufc_dofmap.local_dimension());
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    const std::size_t cell_index = cell->index();
    ufc_cell.orientation = cell->mesh().cell_orientations()[cell_index];
    ufc_cell.topological_dimension = D;

    for (std::size_t d = 0; d < D; d++)
    {
      ufc_cell.entity_indices[d].resize(cell->num_entities(d));
      for (std::size_t i = 0; i < cell->num_entities(d); ++i)
        ufc_cell.entity_indices[d][i] = cell->entities(d)[i];
    }

    // Cell index
    ufc_cell.entity_indices[D].resize(1);
    ufc_cell.entity_indices[D][0] = cell_index;

    // Local cell index short-cut
    ufc_cell.index = ufc_cell.entity_indices[D][0];

    // Tabulate dofs for cell
    ufc_dofmap.tabulate_dofs(dof_holder.data(), num_mesh_entities,
                             ufc_cell);
    std::copy(dof_holder.begin(), dof_holder.end(),
              dofmap[cell_index].begin());
  }
}
//-----------------------------------------------------------------------------
int
DofMapBuilder::compute_node_ownership(
  std::vector<short int>& node_ownership,
  std::unordered_map<int, std::vector<int>>& shared_node_to_processes,
  std::set<int>& neighbours,
  const std::vector<std::vector<la_index>>& dofmap,
  const std::vector<int>& boundary_nodes,
  const std::set<std::size_t>& global_nodes,
  const std::vector<std::size_t>& local_to_global,
  const Mesh& mesh)
{
  log(TRACE, "Determining node ownership for parallel dof map");

  // Get number of nodes
  const std::size_t num_nodes_local = local_to_global.size();

  // Global-to-local node map for nodes on boundary
  std::map<std::size_t, int> global_to_local;

  // Initialise node ownership array, assuming all nodes are owned and
  // not shared (1)
  node_ownership.resize(num_nodes_local);

  // Communication buffer
  std::vector<std::size_t> send_buffer;

  // Loop over nodes and buffer nodes on process boundaries
  for (std::size_t i = 0; i < num_nodes_local; ++i)
  {
    if (boundary_nodes[i] >= 0)
    {
      // Mark as provisionally owned and shared (0)
      node_ownership[i] = 0;

      // Buffer global index and 'vote'
      send_buffer.push_back(local_to_global[i]);
      send_buffer.push_back(boundary_nodes[i]);
      global_to_local.insert(std::make_pair(local_to_global[i], i));
    }
    else
      node_ownership[i] = 1;
  }

  // FIXME: The below algortihm can be improved (made more scalable)
  //        by distributing (dof, process) pairs to 'owner' range owner,
  //        then letting each process get the sharing process list. This
  //        will avoid interleaving communication and computation.

  // Decide ownership of shared nodes
  // Get MPI communicator
  const MPI_Comm mpi_comm = mesh.mpi_comm();
  const std::size_t num_prococesses = MPI::size(mpi_comm);
  const std::size_t process_number = MPI::rank(mpi_comm);
  std::vector<std::size_t> recv_buffer;
  for (std::size_t k = 1; k < num_prococesses; ++k)
  {
    const std::size_t src
      = (process_number - k + num_prococesses) % num_prococesses;
    const std::size_t dest = (process_number + k) % num_prococesses;
    MPI::send_recv(mpi_comm, send_buffer, dest, recv_buffer, src);
    for (std::size_t i = 0; i < recv_buffer.size(); i += 2)
    {
      const std::size_t received_node_global = recv_buffer[i];
      const int received_vote = recv_buffer[i + 1];

      // Get (global, local) node pair. If we have this node, process
      // for ownership
      auto it = global_to_local.find(received_node_global);
      if (it != global_to_local.end())
      {
        // Get local node index
        const int received_node_local = it->second;

        // If received node is shared, decide ownership
        if (node_ownership[received_node_local] == 0)
        {
          // Move dofs with higher ownership votes from shared to shared
          // but not owned
          if (received_vote < boundary_nodes[received_node_local])
            node_ownership[received_node_local] = -1;
          else if (received_vote == boundary_nodes[received_node_local]
                   && process_number > src)
          {
            // If votes are equal, let lower rank process take ownership
            node_ownership[received_node_local] = -1;
          }
          else
            node_ownership[received_node_local] = 0;

          // Store the process sharing of the node
          shared_node_to_processes[received_node_local].push_back(src);
        }
        else if (node_ownership[received_node_local] == -1)
        {
          // Store the process sharing of the node
          shared_node_to_processes[received_node_local].push_back(src);
        }
      }
    }
  }

  // Build set of neighbouring processes
  neighbours.clear();
  for (auto it = shared_node_to_processes.begin();
       it != shared_node_to_processes.end(); ++it)
  {
    neighbours.insert(it->second.begin(), it->second.end());
  }

  // Count number of owned nodes
  int num_owned_nodes = 0;
  for (std::size_t i = 0; i < node_ownership.size(); ++i)
  {
    if (node_ownership[i] >= 0)
      ++num_owned_nodes;
  }

  // Add/remove global dofs to/from relevant sets (last process owns
  // global nodes)
  for (std::set<std::size_t>::const_iterator node = global_nodes.begin();
       node != global_nodes.end(); ++node)
  {
    dolfin_assert(*node < node_ownership.size());
    if (process_number == num_prococesses - 1)
      node_ownership[*node] = 0;
    else
    {
      node_ownership[*node] = -1;
      --num_owned_nodes;
    }
  }

  log(TRACE, "Finished determining dof ownership for parallel dof map");
  return num_owned_nodes;
}
//-----------------------------------------------------------------------------
std::set<std::size_t>
DofMapBuilder::compute_global_dofs(std::shared_ptr<const ufc::dofmap> ufc_dofmap,
                                   const std::vector<std::size_t>& num_mesh_entities_local)
{
  // Compute global dof indices
  std::size_t offset_local = 0;
  std::set<std::size_t> global_dof_indices;
  compute_global_dofs(global_dof_indices,
                      offset_local,
                      ufc_dofmap,
                      num_mesh_entities_local);

  return global_dof_indices;
}
//-----------------------------------------------------------------------------
void DofMapBuilder::compute_global_dofs(
  std::set<std::size_t>& global_dofs,
  std::size_t& offset_local,
  const std::shared_ptr<const ufc::dofmap> ufc_dofmap,
  const std::vector<std::size_t>& num_mesh_entities_local)
{
  dolfin_assert(ufc_dofmap);

  if (ufc_dofmap->num_sub_dofmaps() == 0)
  {
    // Check if dofmap is for global dofs
    bool global_dof = true;
    for (std::size_t d = 0; d < num_mesh_entities_local.size(); ++d)
    {
      if (ufc_dofmap->needs_mesh_entities(d))
      {
        global_dof = false;
        break;
      }
    }

    if (global_dof)
    {
      // Check that we have just one dof
      if (ufc_dofmap->global_dimension(num_mesh_entities_local) != 1)
      {
        dolfin_error("DofMapBuilder.cpp",
                     "compute global degrees of freedom",
                     "Global degree of freedom has dimension != 1");
      }

      // Create dummy cell argument to tabulate single global dof
      std::unique_ptr<ufc::cell> ufc_cell(new ufc::cell);
      std::size_t dof_local = 0;
      ufc_dofmap->tabulate_dofs(&dof_local, num_mesh_entities_local,
                                *ufc_cell);

      // Insert global dof index
      std::pair<std::set<std::size_t>::iterator, bool> ret
        = global_dofs.insert(dof_local + offset_local);
      if (!ret.second)
      {
        dolfin_error("DofMapBuilder.cpp",
                     "compute global degrees of freedom",
                     "Global degree of freedom already exists");
      }
    }
  }
  else
  {
    // Loop through sub-dofmap looking for global dofs
    for (std::size_t i = 0; i < ufc_dofmap->num_sub_dofmaps(); ++i)
    {
      // Extract sub-dofmap and intialise
      std::shared_ptr<ufc::dofmap>
        sub_dofmap(ufc_dofmap->create_sub_dofmap(i));
      compute_global_dofs(global_dofs,
                          offset_local,
                          sub_dofmap,
                          num_mesh_entities_local);

      // Get offset
      if (sub_dofmap->num_sub_dofmaps() == 0)
        offset_local  += sub_dofmap->global_dimension(num_mesh_entities_local);
    }
  }
}
//-----------------------------------------------------------------------------
std::shared_ptr<ufc::dofmap>
DofMapBuilder::extract_ufc_sub_dofmap(
  const ufc::dofmap& ufc_dofmap,
  std::size_t& offset,
  const std::vector<std::size_t>& component,
  const std::vector<std::size_t>& num_mesh_entities)
{
  // Check if there are any sub systems
  if (ufc_dofmap.num_sub_dofmaps() == 0)
  {
    dolfin_error("DofMap.cpp",
                 "extract subsystem of degree of freedom mapping",
                 "There are no subsystems");
  }

  // Check that a sub system has been specified
  if (component.empty())
  {
    dolfin_error("DofMap.cpp",
                 "extract subsystem of degree of freedom mapping",
                 "No system was specified");
  }

  // Check the number of available sub systems
  if (component[0] >= ufc_dofmap.num_sub_dofmaps())
  {
    dolfin_error("DofMap.cpp",
                 "extract subsystem of degree of freedom mapping",
                 "Requested subsystem (%d) out of range [0, %d)",
                 component[0], ufc_dofmap.num_sub_dofmaps());
  }

  // Add to offset if necessary
  for (std::size_t i = 0; i < component[0]; i++)
  {
    // Extract sub dofmap
    std::unique_ptr<ufc::dofmap>
      ufc_tmp_dofmap(ufc_dofmap.create_sub_dofmap(i));
    dolfin_assert(ufc_tmp_dofmap);

    // Check dimensional consistency between UFC dofmap and the mesh
    //check_dimensional_consistency(ufc_dofmap, mesh);

    // Get offset
    offset += ufc_tmp_dofmap->global_dimension(num_mesh_entities);
  }

  // Create UFC sub-system
  std::shared_ptr<ufc::dofmap>
    sub_dofmap(ufc_dofmap.create_sub_dofmap(component[0]));
  dolfin_assert(sub_dofmap);

  // Return sub-system if sub-sub-system should not be extracted,
  // otherwise recursively extract the sub sub system
  if (component.size() == 1)
    return sub_dofmap;
  else
  {
    std::vector<std::size_t> sub_component;
    for (std::size_t i = 1; i < component.size(); ++i)
      sub_component.push_back(component[i]);

    std::shared_ptr<ufc::dofmap> sub_sub_dofmap
        = extract_ufc_sub_dofmap(*sub_dofmap, offset, sub_component,
                                 num_mesh_entities);

    return sub_sub_dofmap;
  }
}
//-----------------------------------------------------------------------------
std::size_t DofMapBuilder::compute_blocksize(const ufc::dofmap& ufc_dofmap)
{
  bool has_block_structure = false;
  if (ufc_dofmap.num_sub_dofmaps() > 1)
  {
    // Create UFC first sub-dofmap
    std::unique_ptr<ufc::dofmap>
      ufc_sub_dofmap0(ufc_dofmap.create_sub_dofmap(0));
    dolfin_assert(ufc_sub_dofmap0);

    // Create UFC sub-dofmaps and check that all sub dofmaps have the
    // same number of dofs per entity
    if (ufc_sub_dofmap0->num_sub_dofmaps() != 0)
      has_block_structure = false;
    else
    {
      // Assume dof map has block structure, then check
      has_block_structure = true;

      // Create UFC sub-dofmaps and check that all sub dofmaps have the
      // same number of dofs per entity
      for (std::size_t i = 1; i < ufc_dofmap.num_sub_dofmaps(); ++i)
      {
        std::unique_ptr<ufc::dofmap>
          ufc_sub_dofmap(ufc_dofmap.create_sub_dofmap(i));
        dolfin_assert(ufc_sub_dofmap);
        for (std::size_t d = 0; d <= ufc_dofmap.topological_dimension(); ++d)
        {
          if (ufc_sub_dofmap->num_entity_dofs(d)
              != ufc_sub_dofmap0->num_entity_dofs(d))
          {
            has_block_structure = false;
            break;
          }
        }
      }
    }
  }

  if (has_block_structure)
    return ufc_dofmap.num_sub_dofmaps();
  else
    return 1;
}
//-----------------------------------------------------------------------------
std::shared_ptr<const ufc::dofmap> DofMapBuilder::build_ufc_node_graph(
  std::vector<std::vector<la_index>>& node_dofmap,
  std::vector<std::size_t>& node_local_to_global,
  std::vector<std::size_t>& num_mesh_entities_global,
  std::shared_ptr<const ufc::dofmap> ufc_dofmap,
  const Mesh& mesh,
  std::shared_ptr<const SubDomain> constrained_domain,
  const std::size_t block_size)
{
  dolfin_assert(ufc_dofmap);

  // Start timer for dofmap initialization
  Timer t0("Init dofmap from UFC dofmap");

  // Topological dimension
  const std::size_t D = mesh.topology().dim();

  // Generate and number required mesh entities (local & global, and
  // constrained global)
  std::vector<std::size_t> num_mesh_entities_local(D + 1, 0);
  std::vector<std::size_t> num_mesh_entities_global_unconstrained(D + 1, 0);
  for (std::size_t d = 0; d <= D; ++d)
  {
    if (ufc_dofmap->needs_mesh_entities(d))
    {
      mesh.init(d);
      DistributedMeshTools::number_entities(mesh, d);
      num_mesh_entities_local[d]  = mesh.num_entities(d);
      num_mesh_entities_global_unconstrained[d] = mesh.size_global(d);
    }
  }

  // Extract sub-dofmaps
  std::vector<std::shared_ptr<const ufc::dofmap>> dofmaps(block_size);
  std::vector<std::size_t> offset_local(block_size + 1, 0);
  if (block_size > 1)
  {
    std::vector<std::size_t> component(1);
    std::size_t _offset_local = 0;
    for (std::size_t i = 0; i < block_size; ++i)
    {
      component[0] = i;
      dofmaps[i] = extract_ufc_sub_dofmap(*ufc_dofmap, _offset_local, component,
                                         num_mesh_entities_local);
      offset_local[i] = _offset_local;
    }
  }
  else
    dofmaps[0] = ufc_dofmap;
  offset_local[block_size] = ufc_dofmap->global_dimension(num_mesh_entities_local);

  num_mesh_entities_global = num_mesh_entities_global_unconstrained;

  // Allocate space for dof map
  node_dofmap.clear();
  node_dofmap.resize(mesh.num_cells());

  // Holder for UFC 64-bit dofmap integers
  std::vector<std::size_t> ufc_nodes_global, ufc_nodes_local;

  // Get standard local elem2ent dimension
  const std::size_t local_dim = dofmaps[0]->local_dimension();

  // Create UFC cell and allocate memory
  ufc::cell ufc_cell_global, ufc_cell_local;
  ufc_cell_global.entity_indices.resize(D + 1);
  ufc_cell_local.entity_indices.resize(D + 1);

  // Resize local-to-global map
  node_local_to_global.resize(offset_local[1]);

  // Build dofmaps from ufc::dofmap
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Set cell orientation
    const int cell_orientation
      = cell->mesh().cell_orientations()[cell->index()];
    ufc_cell_global.orientation = cell_orientation;
    ufc_cell_local.orientation = cell_orientation;

    // Get cell data (local)
    for (std::size_t d = 0; d < D; ++d)
    {
      // Get cell local entities
      ufc_cell_local.entity_indices[d].resize(cell->num_entities(d));
      for (std::size_t i = 0; i < cell->num_entities(d); ++i)
        ufc_cell_local.entity_indices[d][i] = cell->entities(d)[i];
    }
    ufc_cell_local.entity_indices[D].resize(1);
    ufc_cell_local.entity_indices[D][0] = cell->index();
    ufc_cell_local.index = cell->index();

    // Get global cell entities/data (global)
    cell->get_cell_data(ufc_cell_global);
    cell->get_cell_topology(ufc_cell_global);

    // Get reference to container for cell dofs
    std::vector<la_index>& cell_nodes = node_dofmap[cell->index()];
    cell_nodes.resize(local_dim);

    // Tabulate standard UFC dof map for first space (local)
    ufc_nodes_local.resize(local_dim);
    dofmaps[0]->tabulate_dofs(ufc_nodes_local.data(),
                              num_mesh_entities_local,
                              ufc_cell_local);
    std::copy(ufc_nodes_local.begin(), ufc_nodes_local.end(),
              cell_nodes.begin());

    // Tabulate standard UFC dof map for first space (global)
    ufc_nodes_global.resize(local_dim);
    dofmaps[0]->tabulate_dofs(ufc_nodes_global.data(),
                              num_mesh_entities_global_unconstrained,
                              ufc_cell_global);

    // Build local-to-global map for nodes
    for (std::size_t i = 0; i < local_dim; ++i)
      node_local_to_global[ufc_nodes_local[i]] = ufc_nodes_global[i];
  }

  return dofmaps[0];
}
//-----------------------------------------------------------------------------
std::shared_ptr<const ufc::dofmap>
DofMapBuilder::build_ufc_node_graph_constrained(
  std::vector<std::vector<la_index>>& node_dofmap,
  std::vector<std::size_t>& node_local_to_global,
  std::vector<int>& node_ufc_local_to_local,
  std::vector<std::size_t>& num_mesh_entities_global,
  std::shared_ptr<const ufc::dofmap> ufc_dofmap,
  const Mesh& mesh,
  std::shared_ptr<const SubDomain> constrained_domain,
  const std::size_t block_size)
{
  dolfin_assert(ufc_dofmap);
  dolfin_assert(constrained_domain);

  // Start timer for dofmap initialization
  Timer t0("Init dofmap from UFC dofmap");

  // Topological dimension
  const std::size_t D = mesh.topology().dim();

  // Generate and number required mesh entities (local & global, and
  // constrained global)
  std::vector<std::size_t> num_mesh_entities_local(D + 1, 0);
  std::vector<std::size_t> num_mesh_entities_global_unconstrained(D + 1, 0);
  std::vector<bool> required_mesh_entities(D + 1, false);
  for (std::size_t d = 0; d <= D; ++d)
  {
    if (ufc_dofmap->needs_mesh_entities(d))
    {
      required_mesh_entities[d] = true;
      mesh.init(d);
      DistributedMeshTools::number_entities(mesh, d);
      num_mesh_entities_local[d]  = mesh.num_entities(d);
      num_mesh_entities_global_unconstrained[d] = mesh.size_global(d);
    }
  }

  // Get constrained mesh entities
  std::vector<std::vector<std::size_t>> global_entity_indices;
  compute_constrained_mesh_indices(global_entity_indices,
                                   num_mesh_entities_global,
                                   required_mesh_entities,
                                   mesh, *constrained_domain);

  // Extract sub-dofmaps
  std::vector<std::shared_ptr<const ufc::dofmap>> dofmaps(block_size);
  std::vector<std::size_t> offset_local(block_size + 1, 0);
  if (block_size > 1)
  {
    std::vector<std::size_t> component(1);
    std::size_t _offset_local = 0;
    for (std::size_t i = 0; i < block_size; ++i)
    {
      component[0] = i;
      dofmaps[i] = extract_ufc_sub_dofmap(*ufc_dofmap, _offset_local,
                                          component,
                                         num_mesh_entities_local);
      offset_local[i] = _offset_local;
    }
  }
  else
    dofmaps[0] = ufc_dofmap;

  offset_local[block_size]
    = ufc_dofmap->global_dimension(num_mesh_entities_local);

  //num_mesh_entities_global = num_mesh_entities_global_unconstrained;

  // Allocate space for dof map
  node_dofmap.clear();
  node_dofmap.resize(mesh.num_cells());

  // Holder for UFC 64-bit dofmap integers
  std::vector<std::size_t> ufc_nodes_local, ufc_nodes_global,
    ufc_nodes_global_constrained;

  // Get standard local element dimension
  const std::size_t local_dim = dofmaps[0]->local_dimension();

  // Create UFC cell and allocate memory
  ufc::cell ufc_cell;

  // Resize local-to-global map
  std::vector<std::size_t> node_local_to_global_constrained(offset_local[1]);
  node_local_to_global.resize(offset_local[1]);

  // Build dofmaps from ufc::dofmap
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Get reference to container for cell dofs
    std::vector<la_index>& cell_nodes = node_dofmap[cell->index()];
    cell_nodes.resize(local_dim);

    // Tabulate standard UFC dof map for first space (local)
    ufc_nodes_local.resize(local_dim);
    get_cell_data_local(ufc_cell, *cell);
    dofmaps[0]->tabulate_dofs(ufc_nodes_local.data(),
                              num_mesh_entities_local,
                              ufc_cell);
    std::copy(ufc_nodes_local.begin(), ufc_nodes_local.end(),
              cell_nodes.begin());

    // Tabulate standard UFC dof map for first space (global)
    ufc_nodes_global.resize(local_dim);
    cell->get_cell_data(ufc_cell);
    cell->get_cell_topology(ufc_cell);
    dofmaps[0]->tabulate_dofs(ufc_nodes_global.data(),
                              num_mesh_entities_global_unconstrained,
                              ufc_cell);

    // Tabulate standard UFC dof map for first space (global, constrained)
    ufc_nodes_global_constrained.resize(local_dim);
    get_cell_data_global_constrained(ufc_cell, *cell,
                                     global_entity_indices);
    dofmaps[0]->tabulate_dofs(ufc_nodes_global_constrained.data(),
                              num_mesh_entities_global,
                              ufc_cell);

    // Build local-to-global map for nodes
    for (std::size_t i = 0; i < local_dim; ++i)
    {
      node_local_to_global[ufc_nodes_local[i]]
        = ufc_nodes_global_constrained[i];
    }
  }

  // Modify for constraints
  std::map<std::size_t, int> global_to_local;
  std::vector<std::size_t> node_local_to_global_mod(offset_local[1]);
  node_ufc_local_to_local.resize(offset_local[1]);
  int counter = 0;
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Get nodes (local) on cell
    std::vector<la_index>& cell_nodes = node_dofmap[cell->index()];
    for (std::size_t i = 0; i < cell_nodes.size(); ++i)
    {
      dolfin_assert(cell_nodes[i] < (int) node_local_to_global.size());
      const std::size_t node_index_global
        = node_local_to_global[cell_nodes[i]];
      auto it = global_to_local.insert(std::make_pair(node_index_global,
                                                      counter));
      if (it.second)
      {
        dolfin_assert(counter < (int) node_local_to_global_mod.size());
        node_local_to_global_mod[counter] = node_local_to_global[cell_nodes[i]];

        dolfin_assert(cell_nodes[i] < (int) node_ufc_local_to_local.size());
        node_ufc_local_to_local[cell_nodes[i]] = counter;

        cell_nodes[i] = counter;
        counter++;
      }
      else
      {
          node_local_to_global_mod[it.first->second]
            = node_local_to_global[cell_nodes[i]];
          //if (has_submaps)
          node_ufc_local_to_local[cell_nodes[i]] = it.first->second;
          cell_nodes[i] = it.first->second;
      }
    }
  }
  node_local_to_global_mod.resize(counter);
  node_local_to_global = node_local_to_global_mod;

  // Build maps
  //const bool has_submaps = ufc_dofmap->num_sub_dofmaps() > 0 ? true : false;

  // Build UFC local-to-global map if map has sub-dofmaps
  /*
  if (has_submaps)
  {
    ufc_local_to_local.resize(ufc_dofmap->global_dimension(num_mesh_entities_local),
                              0);
    for (std::size_t i = 0; i <  ufc_local_to_local.size(); ++i)
      ufc_local_to_local[i] = i;
  }
  else
    ufc_local_to_local.clear();
  */

  return dofmaps[0];
}
//-----------------------------------------------------------------------------
void DofMapBuilder::compute_constrained_mesh_indices(
  std::vector<std::vector<std::size_t>>& global_entity_indices,
  std::vector<std::size_t>& num_mesh_entities_global,
  const std::vector<bool>& needs_mesh_entities,
  const Mesh& mesh,
  const SubDomain& constrained_domain)
{
  // Topological dimension
  const std::size_t D = mesh.topology().dim();
  dolfin_assert(needs_mesh_entities.size() == (D + 1));

  // Compute slave-master pairs
  std::map<unsigned int,
           std::map<unsigned int, std::pair<unsigned int, unsigned int>>>
    slave_master_mesh_entities;
  for (std::size_t d = 0; d <= D; ++d)
  {
    if (needs_mesh_entities[d])
    {
      slave_master_mesh_entities.insert(std::make_pair(d,
        PeriodicBoundaryComputation::compute_periodic_pairs(mesh,
                                                            constrained_domain,
                                                            d)));
    }
  }

  global_entity_indices.resize(D + 1);
  num_mesh_entities_global.resize(D + 1);
  std::fill(num_mesh_entities_global.begin(),
            num_mesh_entities_global.end(), 0);

  // Compute number of mesh entities
  for (std::size_t d = 0; d <= D; ++d)
  {
    if (needs_mesh_entities[d])
    {
      // Get master-slave map
      dolfin_assert(slave_master_mesh_entities.find(d) != slave_master_mesh_entities.end());
      const auto& slave_to_master_mesh_entities
        = slave_master_mesh_entities.find(d)->second;
      if (d == 0)
      {
        // Compute modified global vertex indices
        const std::size_t num_vertices
          = build_constrained_vertex_indices(mesh,
                                             slave_to_master_mesh_entities,
                                             global_entity_indices[0]);
        num_mesh_entities_global[0] = num_vertices;
      }
      else
      {
        // Get number of entities
        std::map<unsigned int, std::set<unsigned int>> shared_entities;
        const std::size_t num_entities
          = DistributedMeshTools::number_entities(mesh,
                                                  slave_to_master_mesh_entities,
                                                  global_entity_indices[d],
                                                  shared_entities, d);
        num_mesh_entities_global[d] = num_entities;
      }
    }
  }
}
//-----------------------------------------------------------------------------
void
DofMapBuilder::compute_boundary_nodes(std::vector<int>& boundary_nodes,
                                      const std::vector<std::vector<la_index>>& node_dofmap,
                                      const std::size_t num_nodes_local,
                                      const ufc::dofmap& ufc_dofmap,
                                      const Mesh& mesh,
                                      const std::size_t seed)
{
  // Initialise mesh
  const std::size_t D = mesh.topology().dim();
  mesh.init(D - 1);
  mesh.init(D - 1, D);

  // Allocate data and initialise all facets to -1
  boundary_nodes.resize(num_nodes_local);
  std::fill(boundary_nodes.begin(), boundary_nodes.end(), -1);

  // Create a random number generator for ownership 'voting'
  std::mt19937 engine(seed);
  std::uniform_int_distribution<> distribution(0, std::numeric_limits<int>::max());

  // Mark nodes on inter-process boundary
  std::vector<std::size_t> facet_nodes(ufc_dofmap.num_facet_dofs());
  for (FacetIterator f(mesh); !f.end(); ++f)
  {
    // Skip if facet is internal on this partition (i.e., cannot be on
    // process boundary)
    if (f->num_entities(D) == 2)
      continue;

    // Get cell to which facet belongs (pick first)
    const Cell c(mesh, f->entities(mesh.topology().dim())[0]);

    // Tabulate dofs (local) on cell
    const std::vector<la_index>& cell_nodes = node_dofmap[c.index()];

    // Tabulate which dofs are on the facet
    ufc_dofmap.tabulate_facet_dofs(facet_nodes.data(), c.index(*f));

    // Mark boundary nodes and insert into map
    for (std::size_t i = 0; i < facet_nodes.size(); ++i)
    {
      // Get facet node local index and assign votes (positive integer)
      size_t facet_node_local = cell_nodes[facet_nodes[i]];
      if (boundary_nodes[facet_node_local] < 0)
        boundary_nodes[facet_node_local] = distribution(engine);
    }
  }
}
//-----------------------------------------------------------------------------
void DofMapBuilder::compute_node_reodering(
  std::vector<std::size_t>& local_to_global_unowned,
  std::vector<int>& off_process_owner,
  std::vector<std::size_t>& local_to_global,
  std::vector<int>& old_to_new_local,
  const std::vector<std::size_t>& old_local_to_global,
  const std::vector<std::vector<la_index>>& node_dofmap,
  const std::vector<short int>& node_ownership,
  const std::set<std::size_t>& global_nodes,
  MPI_Comm mpi_comm)
{
  // Count number of locally owned nodes
  std::size_t owned_local_size = 0;
  std::size_t unowned_local_size = 0;
  for (auto node = node_ownership.begin(); node != node_ownership.end(); ++node)
  {
    if (*node >= 0)
      ++owned_local_size;
    else if (*node == -1)
      ++unowned_local_size;
    else
      error("Invalid node ownership index.");
  }
  dolfin_assert((unowned_local_size+owned_local_size) == node_ownership.size());
  dolfin_assert((unowned_local_size+owned_local_size) == old_local_to_global.size());

  // Create global-to-local index map for local un-owned nodes
  std::vector<std::pair<std::size_t, int>> node_pairs;
  node_pairs.reserve(unowned_local_size);
  for (std::size_t i = 0; i < node_ownership.size(); ++i)
  {
    if (node_ownership[i] == -1)
      node_pairs.push_back(std::make_pair(old_local_to_global[i] , i));
  }
  std::map<std::size_t, int>
    global_to_local_nodes_unowned(node_pairs.begin(), node_pairs.end());
  std::vector<std::pair<std::size_t, int>>().swap(node_pairs);

  // Build graph for re-ordering. Below block is scoped to clear
  // working data structures once graph is constructed.
  Graph graph(owned_local_size);

  // Create contiguous local numbering for locally owned dofs
  std::size_t my_counter = 0;
  std::vector<int> old_to_contiguous_node_index(node_ownership.size(), -1);
  for (std::size_t i = 0; i < node_ownership.size(); ++i)
  {
    if (node_ownership[i] >= 0)
      old_to_contiguous_node_index[i] = my_counter++;
  }

  // Build local graph, based on old dof map, with contiguous
  // numbering
  for (std::size_t cell = 0; cell < node_dofmap.size(); ++cell)
  {
    // Cell dofmaps with old local indices
    const std::vector<la_index>& nodes0 = node_dofmap[cell];
    const std::vector<la_index>& nodes1 = node_dofmap[cell];

    // Loop over nodes0
    for (std::size_t i = 0; i < nodes0.size(); ++i)
    {
      if (global_nodes.find(nodes0[i]) != global_nodes.end())
        continue;

      // Old node index (0)
      const int n0_old = nodes0[i];

      // New node index (0)
      dolfin_assert(n0_old < (int) old_to_contiguous_node_index.size());
      const int n0_local = old_to_contiguous_node_index[n0_old];

      // Add to graph if node n0_local is owned
      if (n0_local != -1)
      {
        dolfin_assert(n0_local < (int) graph.size());
        for (std::size_t j = 0; j < nodes1.size(); ++j)
        {
          if (global_nodes.find(nodes1[j]) != global_nodes.end())
            continue;

          // Old node index (1)
          const int n1_old = nodes1[j];

          // New node index (1)
          dolfin_assert(n1_old < (int) old_to_contiguous_node_index.size());
          const int n1_local = old_to_contiguous_node_index[n1_old];

          // Add edge graph if node n1_local is owned
          if (n1_local != -1)
          {
            if (n0_local != n1_local)
              graph[n0_local].insert(n1_local);
          }
        }
      }
    }
  }

  // Reorder nodes
  const std::string ordering_library
    = dolfin::parameters["dof_ordering_library"];
  std::vector<int> node_remap;
  if (ordering_library == "Boost")
    node_remap = BoostGraphOrdering::compute_cuthill_mckee(graph, true);
  else if (ordering_library == "SCOTCH")
    node_remap = SCOTCH::compute_gps(graph);
  else if (ordering_library == "random")
  {
    // NOTE: Randomised dof ordering should only be used for
    // testing/benchmarking
    node_remap.resize(graph.size());
    for (std::size_t i = 0; i < node_remap.size(); ++i)
      node_remap[i] = i;
    std::random_shuffle(node_remap.begin(), node_remap.end());
  }
  else
  {
    dolfin_error("DofMapBuilder.cpp",
                 "reorder degrees of freedom",
                 "The requested ordering library '%s' is unknown",
                 ordering_library.c_str());
  }

  // Compute offset for owned nodes
  const std::size_t process_offset
    = MPI::global_offset(mpi_comm, owned_local_size, true);

  // Allocate space
  old_to_new_local.clear();
  old_to_new_local.resize(node_ownership.size(), -1);
  local_to_global.resize(node_ownership.size());

  // Renumber owned nodes, and buffer nodes that are owned but shared
  // with another process
  std::vector<std::size_t> send_buffer;
  std::size_t counter = 0;
  for (std::size_t old_node_index_local = 0;
       old_node_index_local < node_ownership.size();
       ++old_node_index_local)
  {
    // Skip nodes that are not owned (will receive global index later)
    if (node_ownership[old_node_index_local] < 0)
      continue;

    // Set new node number
    dolfin_assert(counter < node_remap.size());
    dolfin_assert(old_node_index_local < old_to_new_local.size());
    dolfin_assert(node_remap[counter] < (int) local_to_global.size());
    old_to_new_local[old_node_index_local] = node_remap[counter];
    local_to_global[node_remap[counter]] = node_remap[counter] + process_offset ;

    // If this node is shared and owned, buffer old and new (global)
    // node index for sending
    if (node_ownership[old_node_index_local] == 0)
    {
      // Buffer old and new global indices t send
      send_buffer.push_back(old_local_to_global[old_node_index_local]);
      send_buffer.push_back(process_offset + node_remap[counter]);
    }

    ++counter;
  }

  // FIXME: The below algorithm can be improved (made more scalable)
  //        by distributing (dof, process) pairs to 'owner' range
  //        owner, then letting each process get the sharing process
  //        list. This will avoid interleaving communication and
  //        computation.
  std::map<std::size_t , std::size_t> received_node_indices;

  // Exchange new node numbers for nodes that are shared
  const std::size_t num_processes = MPI::size(mpi_comm);
  const std::size_t process_number = MPI::rank(mpi_comm);
  std::vector<std::size_t> recv_buffer;

  local_to_global_unowned.resize(unowned_local_size);
  off_process_owner.resize(unowned_local_size);

  std::size_t off_process_node_counter = 0;
  std::set<std::size_t> test_set;

  for (std::size_t k = 1; k < num_processes; ++k)
  {
    const std::size_t src
      = (process_number - k + num_processes) % num_processes;
    const std::size_t dest = (process_number + k) % num_processes;
    MPI::send_recv(mpi_comm, send_buffer, dest, recv_buffer, src);

    // Add nodes renumbered by another process to the old-to-new map
    for (std::size_t i = 0; i < recv_buffer.size(); i += 2)
    {
      const std::size_t received_old_node_index_global = recv_buffer[i];
      const std::size_t received_new_node_index_global = recv_buffer[i + 1];

      // Get local node index (if on this process), and check if this
      // process has shared node (and is not the owner)
      auto it
        = global_to_local_nodes_unowned.find(received_old_node_index_global);
      if (it != global_to_local_nodes_unowned.end())
      {
        // Get local index for received dof
        const int received_old_node_index_local = it->second;
        dolfin_assert(received_old_node_index_local
                      < (int) node_ownership.size());
        dolfin_assert(node_ownership[received_old_node_index_local] == -1);

        // Testing
        if (test_set.find(received_old_node_index_global) != test_set.end())
         error("Oops, problem in new dofmap code.");
        test_set.insert(received_old_node_index_global);
        //dolfin_assert(local_to_global_unowned.size() == off_process_node_counter);

        // Assign local index to remotely numbered node
        //if (MPI::rank(MPI_COMM_WORLD) ==0)
        //  std::cout << "** Insert: " << local_to_global_unowned.size() << ", "
        //            << received_new_node_index_global << std::endl;
        local_to_global_unowned[off_process_node_counter]
          = received_new_node_index_global;
        off_process_owner[off_process_node_counter] = src;

        // Add to old-to-new node map
        const int new_index_local = owned_local_size + off_process_node_counter;
        old_to_new_local[received_old_node_index_local] = new_index_local;
        local_to_global[new_index_local] = received_new_node_index_global;

        off_process_node_counter++;
      }
    }
  }

  // Sanity check
  for (auto it : old_to_new_local)
  {
      dolfin_assert(it != -1);
  }

}
//-----------------------------------------------------------------------------
void DofMapBuilder::build_dofmap(
  std::vector<std::vector<la_index>>& dofmap,
  const std::vector<std::vector<la_index>>& node_dofmap,
  const std::vector<int>& old_to_new_node_local,
  const std::size_t block_size)
{
  // Build dofmap looping over nodes
  dofmap.resize(node_dofmap.size());
  for (std::size_t i = 0; i < node_dofmap.size(); ++i)
  {
    const std::size_t local_dim0 = node_dofmap[i].size();
    dofmap[i].resize(block_size*local_dim0);
    for (std::size_t j = 0; j < local_dim0; ++j)
    {
      const int old_node = node_dofmap[i][j];
      dolfin_assert(old_node < (int)  old_to_new_node_local.size());
      const int new_node = old_to_new_node_local[old_node];
      for (std::size_t block = 0; block < block_size; ++block)
      {
        dolfin_assert((block*local_dim0 + j) < dofmap[i].size());
        dofmap[i][block*local_dim0 + j] = block_size*new_node + block;
      }
    }
  }
}
//-----------------------------------------------------------------------------
void DofMapBuilder::get_cell_data_local(ufc::cell& ufc_cell,
                                        const Cell& cell)
{
  const std::size_t D = cell.mesh().topology().dim();
  ufc_cell.entity_indices.resize(D + 1);

  // Set cell orientation
  ufc_cell.orientation = cell.orientation();

  // Set data
  for (std::size_t d = 0; d < D; ++d)
  {
    // Get cell local entities
    ufc_cell.entity_indices[d].resize(cell.num_entities(d));
    for (std::size_t i = 0; i < cell.num_entities(d); ++i)
      ufc_cell.entity_indices[d][i] = cell.entities(d)[i];
  }
  ufc_cell.entity_indices[D].resize(1);
  ufc_cell.entity_indices[D][0] = cell.index();
  ufc_cell.index = cell.index();
}
//-----------------------------------------------------------------------------
void DofMapBuilder::get_cell_data_global_constrained(
  ufc::cell& ufc_cell, const Cell& cell,
  const std::vector<std::vector<std::size_t>>& global_entity_indices)
{
  const std::size_t D = cell.mesh().topology().dim();
  ufc_cell.entity_indices.resize(D + 1);

  for (std::size_t d = 0; d < D; ++d)
  {
    ufc_cell.entity_indices[d].resize(cell.num_entities(d));
    if (!global_entity_indices[d].empty())
    {
      ufc_cell.entity_indices[d].resize(cell.num_entities(d));
      for (std::size_t i = 0; i < cell.num_entities(d); ++i)
      {
        ufc_cell.entity_indices[d][i]
          = global_entity_indices[d][cell.entities(d)[i]];
      }
    }
  }
  ufc_cell.entity_indices[D].resize(1);
  ufc_cell.entity_indices[D][0] = cell.index();
  ufc_cell.index = cell.index();
}
//-----------------------------------------------------------------------------
std::vector<std::size_t>
DofMapBuilder::compute_num_mesh_entities_local(const Mesh& mesh,
                                               const ufc::dofmap& ufc_dofmap)
{
  const std::size_t D = mesh.topology().dim();
  std::vector<std::size_t> num_mesh_entities_local(D + 1);
  for (std::size_t d = 0; d <= D; ++d)
  {
    if (ufc_dofmap.needs_mesh_entities(d))
    {
      mesh.init(d);
      num_mesh_entities_local[d] = mesh.num_entities(d);
    }
  }

  return num_mesh_entities_local;
}
//-----------------------------------------------------------------------------
