// Copyright (C) 2008-2012 Anders Logg and Ola Skavhaug
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
// Modified by Niclas Jansson 2009.
// Modified by Garth N. Wells 2010.
// Modified by Joachim B Haga, 2012.
// Modified by Mikael Mortensen, 2012.
// Modified by Niclas Jansson 2009
// Modified by Garth N. Wells 2010-2012
// Modified by Joachim B Haga, 2012
//
// First added:  2008-08-12
// Last changed: 2012-11-05

#include <ufc.h>
#include <boost/random.hpp>
#include <boost/unordered_map.hpp>
#include <boost/serialization/map.hpp>
#include <dolfin/common/tuple_serialization.h>

#include <dolfin/common/Timer.h>
#include <dolfin/common/constants.h>
#include <dolfin/graph/BoostGraphOrdering.h>
#include <dolfin/graph/GraphBuilder.h>
#include <dolfin/graph/SCOTCH.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/BoundaryMesh.h>
#include <dolfin/mesh/Facet.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/Restriction.h>
#include "DofMap.h"
#include "UFCCell.h"
#include "UFCMesh.h"
#include "DofMapBuilder.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void DofMapBuilder::build(DofMap& dofmap,
                          const Mesh& dolfin_mesh,
                          const UFCMesh& ufc_mesh,
                          bool reorder,
                          bool distributed)
{
  // Start timer for dofmap initialization
  Timer t0("Init dofmap");

  // Create space for dof map
  dofmap._dofmap.resize(dolfin_mesh.num_cells());
  dofmap._off_process_owner.clear();
  dolfin_assert(dofmap._ufc_dofmap);

  // Temporary holder until UFC supporte 64-bit integers
  std::vector<uint> tmp_dofs;

  // Build dofmap from ufc::dofmap
  dolfin::UFCCell ufc_cell(dolfin_mesh);
  for (dolfin::CellIterator cell(dolfin_mesh); !cell.end(); ++cell)
  {
    // Update UFC cell
    ufc_cell.update(*cell);

    // Get standard local dimension
    const uint local_dim = dofmap._ufc_dofmap->local_dimension(ufc_cell);

    // Get container for cell dofs
    std::vector<DolfinIndex>& cell_dofs = dofmap._dofmap[cell->index()];
    cell_dofs.resize(local_dim);
    tmp_dofs.resize(local_dim);

    // Tabulate standard UFC dof map
    // Temporary fix until UFC supporte 64-bit integers
    dofmap._ufc_dofmap->tabulate_dofs(&tmp_dofs[0],
                                      ufc_mesh, ufc_cell);
    std::copy(tmp_dofs.begin(), tmp_dofs.end(), cell_dofs.begin());
  }

  // Set global dimension
  dofmap._global_dimension = dofmap._ufc_dofmap->global_dimension();

  // Build (re-order) dofmap when running in parallel
  if (distributed)
  {
    // Build set of global dofs
    const set global_dofs = compute_global_dofs(dofmap, dolfin_mesh);

    // Build distributed dof map
    build_distributed(dofmap, global_dofs, dolfin_mesh);
  }
  else
  {
    if (reorder)
    {
      // Build graph
      Graph graph(dofmap.global_dimension());
      for (CellIterator cell(dolfin_mesh); !cell.end(); ++cell)
      {
        const std::vector<DolfinIndex>& dofs0 = dofmap.cell_dofs(cell->index());
        const std::vector<DolfinIndex>& dofs1 = dofmap.cell_dofs(cell->index());
        std::vector<DolfinIndex>::const_iterator node;
        for (node = dofs0.begin(); node != dofs0.end(); ++node)
          graph[*node].insert(dofs1.begin(), dofs1.end());
      }

      // Reorder graph (reverse Cuthill-McKee)
      const std::vector<std::size_t> dof_remap
          = BoostGraphOrdering::compute_cuthill_mckee(graph, true);

      // Reorder dof map
      dolfin_assert(dofmap.ufc_map_to_dofmap.empty());
      for (std::size_t i = 0; i < dofmap.global_dimension(); ++i)
        dofmap.ufc_map_to_dofmap[i] = dof_remap[i];

      // Re-number dofs for cell
      std::vector<std::vector<DolfinIndex> >::iterator cell_map;
      std::vector<DolfinIndex>::iterator dof;
      for (cell_map = dofmap._dofmap.begin(); cell_map != dofmap._dofmap.end(); ++cell_map)
        for (dof = cell_map->begin(); dof != cell_map->end(); ++dof)
          *dof = dof_remap[*dof];
    }
    dofmap._ownership_range = std::make_pair(0, dofmap.global_dimension());
  }
  
  // Periodic modification. Compute master-slave pairs and eliminate slaves 
  // Could probably be done prior to compute_ownership etc?? such that we don't need
  // to modify all maps after making the periodic modification.
  if (dolfin_mesh.is_periodic())
    periodic_modification(dofmap, dolfin_mesh);

}
//-----------------------------------------------------------------------------
void DofMapBuilder::build(DofMap& dofmap,
                          const Mesh& dolfin_mesh,
                          const UFCMesh& ufc_mesh,
                          const Restriction& restriction,
                          bool reorder,
                          bool distributed)
{
  info("Building restricted dofmap.");

  // Start timer for dofmap initialization
  Timer t0("Init dofmap");

  // Note: We store a local dof map for each cell but for cells not
  // included in the restriction, the local size will be zero.

  // Create space for dof map
  dofmap._dofmap.resize(dolfin_mesh.num_cells());
  dofmap._off_process_owner.clear();
  dolfin_assert(dofmap._ufc_dofmap);

  // Use a map to renumber dofs on restricted mesh
  map restricted_dofs;

  // Temporary holder until UFC supporte 64-bit integers
  std::vector<uint> tmp_dofs;

  // Build dofmap from ufc::dofmap
  dolfin::UFCCell ufc_cell(dolfin_mesh);
  for (dolfin::CellIterator cell(dolfin_mesh); !cell.end(); ++cell)
  {
    // Skip cells not included in restriction
    if (!restriction.contains(*cell))
      continue;

    // Update UFC cell
    ufc_cell.update(*cell);

    // Get standard local dimension
    const unsigned int local_dim = dofmap._ufc_dofmap->local_dimension(ufc_cell);
    dofmap._dofmap[cell->index()].resize(local_dim);

    // Get container for cell dofs
    std::vector<DolfinIndex>& cell_dofs = dofmap._dofmap[cell->index()];
    cell_dofs.resize(local_dim);
    tmp_dofs.resize(local_dim);

    // Tabulate standard UFC dof map
    // Temporary fix until UFC supporte 64-bit integers
    dofmap._ufc_dofmap->tabulate_dofs(&tmp_dofs[0],
                                      ufc_mesh, ufc_cell);
    std::copy(tmp_dofs.begin(), tmp_dofs.end(), cell_dofs.begin());

    // Renumber by counting, starting at zero
    for (uint i = 0; i < cell_dofs.size(); i++)
    {
      map_iterator it = restricted_dofs.find(cell_dofs[i]);
      if (it == restricted_dofs.end())
      {
        const uint dof = restricted_dofs.size();
        restricted_dofs[cell_dofs[i]] = dof;
        cell_dofs[i] = dof;
      }
      else
        cell_dofs[i] = it->second;
    }
  }

  // FIXME: Debugging
  cout << "Number of restricted dofs: " << restricted_dofs.size() << endl;
  for (map_iterator it = restricted_dofs.begin(); it != restricted_dofs.end(); ++it)
    cout << "  " << it->first << " --> " << it->second << endl;
  dofmap._global_dimension = dofmap._ufc_dofmap->global_dimension();

  // FIXME: Improve code reuse between this one and default build above

  // Set global dimension
  dofmap._global_dimension = restricted_dofs.size();

  // Build (re-order) dofmap when running in parallel
  if (distributed)
  {
    // Build set of global dofs
    const set global_dofs = compute_global_dofs(dofmap, dolfin_mesh);

    // Build distributed dof map
    build_distributed(dofmap, global_dofs, dolfin_mesh);
  }
  else
  {
    if (reorder)
    {
      // Build graph
      Graph graph(dofmap.global_dimension());
      for (CellIterator cell(dolfin_mesh); !cell.end(); ++cell)
      {
        const std::vector<DolfinIndex>& dofs0 = dofmap.cell_dofs(cell->index());
        const std::vector<DolfinIndex>& dofs1 = dofmap.cell_dofs(cell->index());
        std::vector<DolfinIndex>::const_iterator node;
        for (node = dofs0.begin(); node != dofs0.end(); ++node)
          graph[*node].insert(dofs1.begin(), dofs1.end());
      }

      // Reorder graph (reverse Cuthill-McKee)
      const std::vector<std::size_t> dof_remap
          = BoostGraphOrdering::compute_cuthill_mckee(graph, true);

      // Reorder dof map
      dolfin_assert(dofmap.ufc_map_to_dofmap.empty());
      for (std::size_t i = 0; i < dofmap.global_dimension(); ++i)
        dofmap.ufc_map_to_dofmap[i] = dof_remap[i];

      // Re-number dofs for cell
      std::vector<std::vector<DolfinIndex> >::iterator cell_map;
      std::vector<DolfinIndex>::iterator dof;
      for (cell_map = dofmap._dofmap.begin(); cell_map != dofmap._dofmap.end(); ++cell_map)
        for (dof = cell_map->begin(); dof != cell_map->end(); ++dof)
          *dof = dof_remap[*dof];
    }
    dofmap._ownership_range = std::make_pair(0, dofmap.global_dimension());
  }
  
}
//-----------------------------------------------------------------------------
void DofMapBuilder::build_distributed(DofMap& dofmap,
                                      const DofMapBuilder::set& global_dofs,
                                      const Mesh& mesh)
{
  // Create data structures
  DofMapBuilder::set owned_dofs, shared_owned_dofs, shared_unowned_dofs;
  DofMapBuilder::vec_map shared_dof_processes;

  // Computed owned and shared dofs (and owned and un-owned)
  compute_ownership(owned_dofs, shared_owned_dofs, shared_unowned_dofs,
                    shared_dof_processes, dofmap, global_dofs, mesh);

  // Renumber owned dofs and receive new numbering for unowned shared dofs
  parallel_renumber(owned_dofs, shared_owned_dofs, shared_unowned_dofs,
                    shared_dof_processes, dofmap, mesh);
}
//-----------------------------------------------------------------------------
void DofMapBuilder::compute_ownership(set& owned_dofs, set& shared_owned_dofs,
                                      set& shared_unowned_dofs,
                                      vec_map& shared_dof_processes,
                                      const DofMap& dofmap,
                                      const DofMapBuilder::set& global_dofs,
                                      const Mesh& mesh)
{
  log(TRACE, "Determining dof ownership for parallel dof map");

  // Create a radom number generator for ownership 'voting'
  boost::mt19937 engine(MPI::process_number());
  boost::uniform_int<> distribution(0, 100000000);
  boost::variate_generator<boost::mt19937&, boost::uniform_int<> > rng(engine, distribution);

  // Clear data structures
  owned_dofs.clear();
  shared_owned_dofs.clear();
  shared_unowned_dofs.clear();

  // Data structures for computing ownership
  boost::unordered_map<std::size_t, std::size_t> dof_vote;
  std::vector<uint> facet_dofs(dofmap.num_facet_dofs());

  // Communication buffer
  std::vector<std::size_t> send_buffer;

  // Extract the interior boundary
  BoundaryMesh interior_boundary;
  interior_boundary.init_interior_boundary(mesh);

  // Build set of dofs on process boundary (assume all are owned by this process)
  const MeshFunction<std::size_t>& cell_map = interior_boundary.cell_map();
  if (!cell_map.empty())
  {
    for (CellIterator bc(interior_boundary); !bc.end(); ++bc)
    {
      // Get boundary facet
      Facet f(mesh, cell_map[*bc]);

      // Get cell to which facet belongs (pick first)
      Cell c(mesh, f.entities(mesh.topology().dim())[0]);

      // Tabulate dofs on cell
      const std::vector<DolfinIndex>& cell_dofs = dofmap.cell_dofs(c.index());

      // Tabulate which dofs are on the facet
      dofmap.tabulate_facet_dofs(&facet_dofs[0], c.index(f));

      // Insert shared dofs into set and assign a 'vote'
      for (std::size_t i = 0; i < dofmap.num_facet_dofs(); i++)
      {
        if (shared_owned_dofs.find(cell_dofs[facet_dofs[i]]) == shared_owned_dofs.end())
        {
          shared_owned_dofs.insert(cell_dofs[facet_dofs[i]]);
          dof_vote[cell_dofs[facet_dofs[i]]] = rng();

          send_buffer.push_back(cell_dofs[facet_dofs[i]]);
          send_buffer.push_back(dof_vote[cell_dofs[facet_dofs[i]]]);
        }
      }
    }
  }

  // Decide ownership of shared dofs
  const uint num_proc = MPI::num_processes();
  const uint proc_num = MPI::process_number();
  std::vector<std::size_t> recv_buffer;
  for (uint k = 1; k < MPI::num_processes(); ++k)
  {
    const uint src  = (proc_num - k + num_proc) % num_proc;
    const uint dest = (proc_num + k) % num_proc;
    MPI::send_recv(send_buffer, dest, recv_buffer, src);

    for (std::size_t i = 0; i < recv_buffer.size(); i += 2)
    {
      const std::size_t received_dof  = recv_buffer[i];
      const std::size_t received_vote = recv_buffer[i + 1];

      if (shared_owned_dofs.find(received_dof) != shared_owned_dofs.end())
      {
        // Move dofs with higher ownership votes from shared to shared
        // but not owned
        if (received_vote < dof_vote[received_dof])
        {
          shared_unowned_dofs.insert(received_dof);
          shared_owned_dofs.erase(received_dof);
        }
        else if (received_vote == dof_vote[received_dof] && proc_num > src)
        {
          // If votes are equal, let lower rank process take ownership
          shared_unowned_dofs.insert(received_dof);
          shared_owned_dofs.erase(received_dof);
        }

        // Remember the sharing of the dof
        shared_dof_processes[received_dof].push_back(src);
      }
      else if (shared_unowned_dofs.find(received_dof) != shared_unowned_dofs.end())
      {
        // Remember the sharing of the dof
        shared_dof_processes[received_dof].push_back(src);
      }
    }
  }

  // Add/remove global dofs to relevant sets (process 0 owns global dofs)
  if (MPI::process_number() == 0)
  {
    shared_owned_dofs.insert(global_dofs.begin(), global_dofs.end());
    for (set::const_iterator dof = global_dofs.begin(); dof != global_dofs.begin(); ++dof)
    {
      set::const_iterator _dof = shared_unowned_dofs.find(*dof);
      if (_dof != shared_unowned_dofs.end())
        shared_unowned_dofs.erase(_dof);
    }
  }
  else
  {
    shared_unowned_dofs.insert(global_dofs.begin(), global_dofs.end());
    for (set::const_iterator dof = global_dofs.begin(); dof != global_dofs.begin(); ++dof)
    {
      set::const_iterator _dof = shared_owned_dofs.find(*dof);
      if (_dof != shared_owned_dofs.end())
        shared_owned_dofs.erase(_dof);
    }
  }

  // Mark all shared and owned dofs as owned by the processes
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    const std::vector<DolfinIndex>& cell_dofs = dofmap.cell_dofs(cell->index());
    const uint cell_dimension = dofmap.cell_dimension(cell->index());
    for (uint i = 0; i < cell_dimension; ++i)
    {
      // Mark dof as owned if in unowned set
      if (shared_unowned_dofs.find(cell_dofs[i]) == shared_unowned_dofs.end())
        owned_dofs.insert(cell_dofs[i]);
    }
  }

  // Check that sum of locally owned dofs is equal to global dimension
  const std::size_t _owned_dim = owned_dofs.size();
  dolfin_assert(MPI::sum(_owned_dim) == dofmap._ufc_dofmap->global_dimension());

  log(TRACE, "Finished determining dof ownership for parallel dof map");
}
//-----------------------------------------------------------------------------
void DofMapBuilder::parallel_renumber(const set& owned_dofs,
                             const set& shared_owned_dofs,
                             const set& shared_unowned_dofs,
                             const vec_map& shared_dof_processes,
                             DofMap& dofmap, const Mesh& mesh)
{
  log(TRACE, "Renumber dofs for parallel dof map");

  // FIXME: Handle double-renumbered dof map
  if (!dofmap.ufc_map_to_dofmap.empty())
  {
    dolfin_error("DofMapBuilder.cpp",
                 "compute parallel renumbering of degrees of freedom",
                 "The degree of freedom mapping cannot (yet) be renumbered twice");
  }

  const std::vector<std::vector<DolfinIndex> >& old_dofmap = dofmap._dofmap;
  std::vector<std::vector<DolfinIndex> > new_dofmap(old_dofmap.size());
  dolfin_assert(old_dofmap.size() == mesh.num_cells());

  // Compute offset for owned and non-shared dofs
  const std::size_t process_offset = MPI::global_offset(owned_dofs.size(), true);

  // Clear some data
  dofmap._off_process_owner.clear();

  // Build vector of owned dofs
  const std::vector<std::size_t> my_dofs(owned_dofs.begin(), owned_dofs.end());

  // Create contiguous local numbering for locally owned dofs
  std::size_t my_counter = 0;
  boost::unordered_map<std::size_t, std::size_t> my_old_to_new_dof_index;
  for (set_iterator owned_dof = owned_dofs.begin(); owned_dof != owned_dofs.end(); ++owned_dof, my_counter++)
    my_old_to_new_dof_index[*owned_dof] = my_counter;

  // Build local graph based on old dof map with contiguous numbering
  Graph graph(owned_dofs.size());
  for (std::size_t cell = 0; cell < old_dofmap.size(); ++cell)
  {
    const std::vector<DolfinIndex>& dofs0 = dofmap.cell_dofs(cell);
    const std::vector<DolfinIndex>& dofs1 = dofmap.cell_dofs(cell);
    std::vector<DolfinIndex>::const_iterator node0, node1;
    for (node0 = dofs0.begin(); node0 != dofs0.end(); ++node0)
    {
      boost::unordered_map<std::size_t, std::size_t>::const_iterator _node0
          = my_old_to_new_dof_index.find(*node0);
      if (_node0 != my_old_to_new_dof_index.end())
      {
        const std::size_t local_node0 = _node0->second;
        dolfin_assert(local_node0 < graph.size());
        for (node1 = dofs1.begin(); node1 != dofs1.end(); ++node1)
        {
          boost::unordered_map<std::size_t, std::size_t>::const_iterator
                _node1 = my_old_to_new_dof_index.find(*node1);
          if (_node1 != my_old_to_new_dof_index.end())
          {
            const std::size_t local_node1 = _node1->second;
            graph[local_node0].insert(local_node1);
          }
        }
      }
    }
  }

  // Reorder dofs locally
  const std::vector<std::size_t> dof_remap
      = BoostGraphOrdering::compute_cuthill_mckee(graph, true);

  // Map from old to new index for dofs
  boost::unordered_map<std::size_t, std::size_t> old_to_new_dof_index;

  // Renumber owned dofs and buffer dofs that are owned but shared with
  // another process
  std::size_t counter = 0;
  std::vector<std::size_t> send_buffer;
  for (set_iterator owned_dof = owned_dofs.begin(); owned_dof != owned_dofs.end(); ++owned_dof, counter++)
  {
    // Set new dof number
    old_to_new_dof_index[*owned_dof] = process_offset + dof_remap[counter];

    // Update UFC-to-renumbered map for new number
    dofmap.ufc_map_to_dofmap[*owned_dof] = process_offset + dof_remap[counter];

    // If this dof is shared and owned, buffer old and new index for sending
    if (shared_owned_dofs.find(*owned_dof) != shared_owned_dofs.end())
    {
      send_buffer.push_back(*owned_dof);
      send_buffer.push_back(process_offset + dof_remap[counter]);
    }
  }

  // Exchange new dof numbers for dofs that are shared
  const uint num_proc = MPI::num_processes();
  const uint proc_num = MPI::process_number();
  std::vector<std::size_t> recv_buffer;
  for (uint k = 1; k < MPI::num_processes(); ++k)
  {
    const uint src  = (proc_num - k + num_proc) % num_proc;
    const uint dest = (proc_num + k) % num_proc;
    MPI::send_recv(send_buffer, dest, recv_buffer, src);

    // Add dofs renumbered by another process to the old-to-new map
    for (std::size_t i = 0; i < recv_buffer.size(); i += 2)
    {
      const std::size_t received_old_dof_index = recv_buffer[i];
      const std::size_t received_new_dof_index = recv_buffer[i + 1];

      // Check if this process has shared dof (and is not the owner)
      if (shared_unowned_dofs.find(received_old_dof_index) != shared_unowned_dofs.end())
      {
        // Add to old-to-new dof map
        old_to_new_dof_index[received_old_dof_index] = received_new_dof_index;

        // Store map from off-process dof to owner
        dofmap._off_process_owner[received_new_dof_index] = src;

        // Update UFC-to-renumbered map
        dofmap.ufc_map_to_dofmap[received_old_dof_index] = received_new_dof_index;
      }
    }
  }

  // Insert the shared-dof-to-process mapping into the dofmap, renumbering
  // as necessary
  for (vec_map::const_iterator it = shared_dof_processes.begin();
            it != shared_dof_processes.end(); ++it)
  {
    boost::unordered_map<std::size_t, std::size_t>::const_iterator new_index = old_to_new_dof_index.find(it->first);
    if (new_index == old_to_new_dof_index.end())
      dofmap._shared_dofs.insert(*it);
    else
      dofmap._shared_dofs.insert(std::make_pair(new_index->second, it->second));
    dofmap._neighbours.insert(it->second.begin(), it->second.end());
  }

  // Build new dof map
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    const std::size_t cell_index = cell->index();
    const uint cell_dimension = dofmap.cell_dimension(cell_index);

    // Resize cell map and insert dofs
    new_dofmap[cell_index].resize(cell_dimension);
    for (uint i = 0; i < cell_dimension; ++i)
    {
      const std::size_t old_index = old_dofmap[cell_index][i];
      new_dofmap[cell_index][i] = old_to_new_dof_index[old_index];
    }
  }

  // Set new dof map
  dofmap._dofmap = new_dofmap;

  // Set ownership range
  dofmap._ownership_range = std::make_pair<std::size_t, std::size_t>(process_offset,
                                          process_offset + owned_dofs.size());

  log(TRACE, "Finished renumbering dofs for parallel dof map");
}
//-----------------------------------------------------------------------------
DofMapBuilder::set DofMapBuilder::compute_global_dofs(const DofMap& dofmap,
                                                       const Mesh& dolfin_mesh)
{
  // Wrap UFC dof map
  boost::shared_ptr<const ufc::dofmap> _dofmap(dofmap._ufc_dofmap.get(),
                                               NoDeleter());

  // Create UFCMesh
  const UFCMesh ufc_mesh(dolfin_mesh);

  // Compute global dof indices
  std::size_t offset = 0;
  set global_dof_indices;
  compute_global_dofs(global_dof_indices, offset, _dofmap, dolfin_mesh, ufc_mesh);

  return global_dof_indices;
}
//-----------------------------------------------------------------------------
void DofMapBuilder::compute_global_dofs(DofMapBuilder::set& global_dofs,
                            std::size_t& offset,
                            boost::shared_ptr<const ufc::dofmap> dofmap,
                            const Mesh& dolfin_mesh, const UFCMesh& ufc_mesh)
{
  dolfin_assert(dofmap);
  const uint D = dolfin_mesh.topology().dim();

  if (dofmap->num_sub_dofmaps() == 0)
  {
    // Check if dofmap is for global dofs
    bool global_dof = true;
    for (std::size_t d = 0; d <= D; ++d)
    {
      if (dofmap->needs_mesh_entities(d))
      {
        global_dof = false;
        break;
      }
    }

    if (global_dof)
    {
      // Check that we have just one dof
      if (dofmap->global_dimension() != 1)
      {
        dolfin_error("DofMapBuilder.cpp",
                     "compute global degrees of freedom",
                     "Global degree of freedom has dimension != 1");
      }

      boost::scoped_ptr<ufc::mesh> ufc_mesh(new ufc::mesh);
      boost::scoped_ptr<ufc::cell> ufc_cell(new ufc::cell);
      //std::size_t dof = 0;
      uint dof = 0;
      dofmap->tabulate_dofs(&dof, *ufc_mesh, *ufc_cell);

      // Insert global dof index
      std::pair<DofMapBuilder::set::iterator, bool> ret = global_dofs.insert(dof + offset);
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
    for (std::size_t i = 0; i < dofmap->num_sub_dofmaps(); ++i)
    {
      // Extract sub-dofmap and intialise
      boost::shared_ptr<ufc::dofmap> sub_dofmap(dofmap->create_sub_dofmap(i));
      DofMap::init_ufc_dofmap(*sub_dofmap, ufc_mesh, dolfin_mesh);

      compute_global_dofs(global_dofs, offset, sub_dofmap, dolfin_mesh,
                          ufc_mesh);

      // Get offset
      if (sub_dofmap->num_sub_dofmaps() == 0)
        offset += sub_dofmap->global_dimension();
    }
  }
}
//-----------------------------------------------------------------------------
void DofMapBuilder::extract_dof_pairs(const DofMap& dofmap, const Mesh& mesh, 
                                      dof_data_map& _slave_master_map,
                                      std::pair<std::size_t, std::size_t> ownership_range)
{
  Timer t0("Extracting dof pairs");
  
  const uint num_sub_dofmaps = dofmap._ufc_dofmap->num_sub_dofmaps();
  if (num_sub_dofmaps > 0)
  {
    // Call recursively for all sub_dofmaps
    std::vector<uint> component(1);
    for (uint i=0; i<num_sub_dofmaps; i++)
    {
      component[0] = i;
      DofMap* sub_dofmap = dofmap.extract_sub_dofmap(component, mesh);
      extract_dof_pairs(*sub_dofmap, mesh, _slave_master_map, ownership_range);
    }
    return;
  }
    
  // Get dimensions
  const uint tdim = mesh.topology().dim();
  const uint gdim = mesh.geometry().dim();
  
  // Arrays used for mapping coordinates
  std::vector<double> x(gdim);
  std::vector<double> y(gdim);
  std::vector<double> dx(gdim);    
  
  const uint process_number = MPI::process_number();
  
  // Declare some variables used to hold information on each facet
  std::vector<uint> facet_dofs(dofmap.num_facet_dofs());
  boost::multi_array<double, 2> facet_coors(boost::extents[dofmap.max_cell_dimension()][gdim]);
      
  // First we send all relevant information on the slave facets to adjoining master.
  // Create a type to hold all info that will be sent. The info is:
  //    (periodic facet id, global slave dofs and coordinates of all slave dofs)
  typedef boost::tuples::tuple<std::size_t, std::vector<std::size_t>, std::vector<std::vector<double> > > facet_info_type;
  typedef std::vector<facet_info_type> facets_info_type;
  typedef std::map<uint, facets_info_type> facet_info_map_type;    
  typedef std::map<uint, std::vector<std::vector<double> > > coor_map_type;
    
  // Use a master_slave map for faster master search
  std::map<std::size_t, std::size_t> _current_master_slave_map;
    
  // Run over periodic domains and build the global _slave_master_map
  for (uint periodic_domain = 0; periodic_domain < mesh.num_periodic_domains(); periodic_domain++)
  {
    // Get periodic info
    facet_pair_type facet_pairs = mesh.get_periodic_facet_pairs(periodic_domain);
    dx = mesh.get_periodic_distance(periodic_domain);  // Distance between periodic domains
    const uint num_periodic_faces = facet_pairs.size(); 
    facet_info_map_type facet_info_map;    
    std::set<uint> communicating_processors;

    // Run over periodic facets and collect all info that should be sent
    for (uint i = 0; i < num_periodic_faces; i++)
    {   
      const uint master_process = facet_pairs[i].first.second;
      const uint slave_process = facet_pairs[i].second.second;
            
      if (master_process == process_number)
        communicating_processors.insert(slave_process);
      
      if (slave_process == process_number)
      {
        // Get dofs and dof-coordinates from slave facet
        const Facet facet(mesh, facet_pairs[i].second.first); 
        const Cell cell(mesh, facet.entities(tdim)[0]);
        const std::vector<DolfinIndex> global_dofs = dofmap.cell_dofs(cell.index());
        dofmap.tabulate_coordinates(facet_coors, cell);
        dofmap.tabulate_facet_dofs(&facet_dofs[0], cell.index(facet));
        communicating_processors.insert(master_process);
        
        // Get dofs and coordinates of dofs on slave facet
        std::vector<std::vector<double> > coors_of_dofs;    
        std::vector<std::size_t> dofs_on_facet;
        for (uint k = 0; k < dofmap.num_facet_dofs(); k++)
        {
          std::copy(facet_coors[facet_dofs[k]].begin(),
                    facet_coors[facet_dofs[k]].end(), y.begin());
          coors_of_dofs.push_back(y);
          dofs_on_facet.push_back(global_dofs[facet_dofs[k]]);
        }
                
        // Put info in type used for communicating with master
        if (facet_info_map.find(master_process) == facet_info_map.end())
        {
          facet_info_type facet_info = facet_info_type(i, dofs_on_facet, coors_of_dofs);
          facets_info_type facets_info;
          facets_info.push_back(facet_info);
          facet_info_map[master_process] = facets_info;
        }
        else
          facet_info_map[master_process].push_back(facet_info_type(i, dofs_on_facet, coors_of_dofs));
      }      
    }  
    
    // Send slave info from all slaves to all masters     
    facet_info_map_type received_info;
    MPI::distribute(communicating_processors, facet_info_map, received_info);

    // Put info from slave facets into new variables
    coor_map_type coors_on_slave;
    vec_map slave_dofs;
    for (facet_info_map_type::const_iterator proc_it = received_info.begin(); 
             proc_it != received_info.end(); ++proc_it)
    {
      facets_info_type info_list = proc_it->second;
      for (uint j = 0; j < info_list.size(); j++)
      {
        std::size_t i = info_list[j].get<0>();  // The periodic facet number
        slave_dofs[i] = info_list[j].get<1>();
        coors_on_slave[i] = info_list[j].get<2>();
      }
    }
    
    // Declare map used to hold global matching pairs of dofs on this process
    std::map<std::size_t, std::size_t> matching_dofs;
    
    // Run over periodic facets and locate matching dof pairs
    for (uint i = 0; i < num_periodic_faces; i++)
    {   
      const uint master_process = facet_pairs[i].first.second;
      
      // Do work only on master process
      if (master_process == process_number)
      {         
        // Get info from master facet: cell, dofs, coordinates
        const Facet facet(mesh, facet_pairs[i].first.first);
        const Cell cell(mesh, facet.entities(tdim)[0]);        
        const std::vector<DolfinIndex> global_dofs = dofmap.cell_dofs(cell.index());
        dofmap.tabulate_coordinates(facet_coors, cell);
        dofmap.tabulate_facet_dofs(&facet_dofs[0], cell.index(facet)); 
        
//         //////////////////////////////////////////////////////////////
//           // Faster search, but this is not really a timeconsuming process anyway...
//         if (dofmap.num_facet_dofs() == 0)
//         {
//           continue;
//         }
//         else if (dofmap.num_facet_dofs() == 1)
//         {
//           std::size_t master_dof = global_dofs[facet_dofs[0]];
//           std::size_t slave_dof = slave_dofs[i][0];
//           if (master_dof >= ownership_range.first && master_dof < ownership_range.second)
//             matching_dofs[master_dof] = slave_dof;
//         }
//         else
//         {
//           // Get global master dof and coordinates of first local dof
//           std::size_t master_dof = global_dofs[facet_dofs[0]];
//           std::copy(facet_coors[facet_dofs[0]].begin(),
//                     facet_coors[facet_dofs[0]].end(), x.begin());      
//           // Look for a match in coordinates of the first local slave dof
//           y = coors_on_slave[i][0];    
//           double error = 0.;
//           for(uint l = 0; l < gdim; l++) 
//             error += std::abs(x[l] - y[l] + dx[l]);
//             
//           if (error < 1.0e-12)    // Match! Assuming the dofs are laid out in the same order 
//                                            // on the facet the remaining are simply copied without control
//           {  
//             for (uint j = 0; j < dofmap.num_facet_dofs(); j++)
//             {
//               master_dof = global_dofs[facet_dofs[j]];
//               if (master_dof >= ownership_range.first && master_dof < ownership_range.second)
//                 matching_dofs[master_dof] = slave_dofs[i][j];
//             }
//           }
//           else  // If local dofs 0/0 don't match the order must be opposite
//           {
//             for (uint j = 0; j < dofmap.num_facet_dofs(); j++)
//             {
//               master_dof = global_dofs[facet_dofs[j]];
//               if (master_dof >= ownership_range.first && master_dof < ownership_range.second)
//                 matching_dofs[master_dof] = slave_dofs[i][dofmap.num_facet_dofs()-j-1];
//             }
//           }
//         }
//         ////////////////////////////////////////////////////////
        
        // Match master and slave dofs and put pair in map
        for (uint j = 0; j < dofmap.num_facet_dofs(); j++)
        {
          //Get global master dof and coordinates
          std::size_t master_dof = global_dofs[facet_dofs[j]];
         
          // Only handle dofs owned by this process
          if (master_dof >= ownership_range.first && master_dof < ownership_range.second)
          {            
            // Check new master_dofs only
            if (matching_dofs.find(master_dof) == matching_dofs.end())
            {
              std::copy(facet_coors[facet_dofs[j]].begin(),
                        facet_coors[facet_dofs[j]].end(), x.begin());

              for (uint k = 0; k < dofmap.num_facet_dofs(); k++)
              {
                // Look for a match in coordinates
                y = coors_on_slave[i][k];                      
                double error = 0.;
                for(uint l = 0; l < gdim; l++) 
                  error += std::abs(x[l] - y[l] + dx[l]);
                  
                if (error < 1.0e-12) // Match! Store master and slave in global_matching_pairs
                {  
                  matching_dofs[master_dof] = slave_dofs[i][k];
                  break;
                }
                if (k == dofmap.num_facet_dofs()-1)
                  dolfin_error("DofMapBuilder.cpp",
                               "extracting dof pairs",
                               "Could not find a pair of matching degrees of freedom");
              }
            }
          }
        }  // Finished on facet. Move to next periodic facet pair
      }                  
    }   // Finished all periodic pairs on periodic domain
    
    // At this point there should be a match between dofs in matching_dofs. 
    // Put the matching dof pairs on all processes    
    std::vector<std::map<std::size_t, std::size_t> > all_dof_pairs;
    MPI::all_gather(matching_dofs, all_dof_pairs);      
    
    typedef std::map<std::size_t, std::size_t>::const_iterator pair_iterator;
    // Add to the global _slave_master_map
    for (uint i = 0; i < all_dof_pairs.size(); i++)
    {
      std::map<std::size_t, std::size_t> matching_dofs = all_dof_pairs[i];            
      for (pair_iterator it = matching_dofs.begin();
                         it != matching_dofs.end(); ++it)
      {
        std::size_t master_dof = it->first; 
        std::size_t slave_dof  = it->second;
        dof_data pair(master_dof, i); // i is dof owner
                
        if (periodic_domain == 0)     // First periodic direction
        {   
          _slave_master_map[slave_dof] = pair;
          _current_master_slave_map[master_dof] = slave_dof;
        }
        else
        {
          // At this point we need to do something clever in case of more than one 
          // periodic direction. For example, a rectangle mesh with two periodic 
          // directions will have four corners that should be equal. In that case 
          // we will here end up with one master and three slaves of the same master. 
          // A 3D Cube with 8 corners should similarily have 7 slaves of the same master. 
          if (_slave_master_map.find(slave_dof) == _slave_master_map.end())
          {
            // If the slave does not exist, then add to slave_master map. But check also
            // if the slave has been used as a master before.
            _slave_master_map[slave_dof] = pair;
            _current_master_slave_map[master_dof] = slave_dof;
    //         The search below is slow, but perhaps more intuitive than the other.
    //         The _current_master_slave_map is used solely for the purpose of speeding up this search.
    //         for (dof_data_map_iterator it = _current_slave_master_map.begin();
    //                                   it != _current_slave_master_map.end(); ++it)
    //         {
    //           if (it->second.first == slave_dof) // Has been previously used as master
    //           {
    //             _slave_master_map[it->first] = pair; // Use latest master value for previous as well
    //             break;
    //           }
    //         }
            if (_current_master_slave_map.find(slave_dof) != _current_master_slave_map.end())
            {
              _slave_master_map[_current_master_slave_map[slave_dof]] = pair;
              _current_master_slave_map.erase(slave_dof);
            }
          }
          else // The slave_dof exists as slave from before
          {
            // Check if the master_dof has been used previously as a slave
            // In that case use previous master for the current slave as well
            if (_slave_master_map.find(master_dof) != _slave_master_map.end())
            {
              _current_master_slave_map.erase(_slave_master_map[slave_dof].first);
              _slave_master_map[slave_dof] = _slave_master_map[master_dof]; 
              _current_master_slave_map[_slave_master_map[master_dof].first] = slave_dof;
            }
          }
        }
      }
    }
  }
  
//     cout << "Map" << endl;
//     for (dof_data_map_iterator it = _slave_master_map.begin();
//                                     it != _slave_master_map.end(); ++it)
//     {
//       cout << "   " << it->first << " " << it->second.first << endl;
//     }
//     cout << "Master Map" << endl;
//     for (std::map<std::size_t, std::size_t>::iterator it = _current_master_slave_map.begin();
//                                     it != _current_master_slave_map.end(); ++it)
//     {
//       cout << "   " << it->first << " " << it->second << endl;
//     }
}

void DofMapBuilder::periodic_modification(DofMap& dofmap, const Mesh& mesh)
{
  Timer t0("Periodic dofmap modification");
    
  // Recursively extract a map from slaves to master dofs for all sub-dofmaps of dofmap.
  dof_data_map _slave_master_map;
  extract_dof_pairs(dofmap, mesh, _slave_master_map, dofmap._ownership_range);
      
  // Get topological dimension
  const uint tdim = mesh.topology().dim();  
  
  // Eliminate all slaves from the dofmap by placing the master in all locations 
  // where a slave is found. For efficiency first find all cells that could contain a slave.  
  set cells_with_slave;
  for (uint periodic_domain = 0; periodic_domain < mesh.num_periodic_domains(); periodic_domain++)
  {
    // Get periodic facet-to-facet map
    facet_pair_type facet_pairs = mesh.get_periodic_facet_pairs(periodic_domain);
    
    for (uint i = 0; i < facet_pairs.size(); i++)
    {   
      const uint slave_process = facet_pairs[i].second.second;
      if (slave_process == MPI::process_number())
      {
        // Get all cells with vertex on the periodic facet.
        const Facet facet(mesh, facet_pairs[i].second.first); 
        const std::size_t* facet_vertices = facet.entities(0);
        if (tdim == 1) // Special treatment of 1D
        {
          const Cell cell(mesh, facet.entities(1)[0]);
          cells_with_slave.insert(cell.index());
        }
        else
        {
          for (uint j = 0; j < facet.num_entities(0); j++)
          {
            const Vertex v(mesh, facet_vertices[j]);
            const std::size_t* vertex_cells = v.entities(tdim);          
            for (uint k = 0; k < v.num_entities(tdim); k++)
              cells_with_slave.insert(vertex_cells[k]);
          }
        }
      }
    }
  }
  
  // Run over cells with potential slave and eliminate slaves from dofmap
  for (set_iterator it = cells_with_slave.begin();
                    it != cells_with_slave.end(); ++it)
  {
    const std::vector<DolfinIndex> global_dofs = dofmap.cell_dofs(*it);
    for (uint j = 0; j < dofmap.max_cell_dimension(); j++)
    {
      const std::size_t dof = global_dofs[j];
      if (_slave_master_map.find(dof) != _slave_master_map.end())
      {
        dofmap._dofmap[*it][j] = _slave_master_map[dof].first; // Switch slave for master
        
        // Modify _off_process_owner if master is not owned by current process
        if (_slave_master_map[dof].second != MPI::process_number())
          dofmap._off_process_owner[_slave_master_map[dof].first] = _slave_master_map[dof].second; 
        dofmap._off_process_owner.erase(dof);
      }      
    }
  }  
  
  // Replace slaves by master in ufc_map_to_dofmap
  for (boost::unordered_map<std::size_t, std::size_t>::const_iterator op = dofmap.ufc_map_to_dofmap.begin();
       op != dofmap.ufc_map_to_dofmap.end(); ++op)
  {
    if (_slave_master_map.find(op->second) != _slave_master_map.end())
      dofmap.ufc_map_to_dofmap[op->first] = _slave_master_map[op->second].first;
  }  
  
  // Remove slaves from _shared_dofs
  for (dof_data_map_iterator it = _slave_master_map.begin();
                             it != _slave_master_map.end(); ++it)
    dofmap._shared_dofs.erase(it->first);
  
  // At this point the slaves should be completely removed from the dofmap
  // and the global dimension of the dofmap can now be reduced.
  // To do this:
  //   1) Compute the total number of slaves that has been eliminated
  //   2) Renumber all dofs by subtracting current dof-number with the number 
  //        of eliminated slaves with a number less than the current
  //   3) Recompute ownership_range
  //   4) Recompute global_dimension (set _global_dimension)
  //   5) Modify all variables due to the new dof numbering

  // Count slaves on this process and alltogether
  std::vector<std::size_t> _slaves_on_process;
  std::vector<std::size_t> _all_slaves;
  for (dof_data_map_iterator it = _slave_master_map.begin();
                             it != _slave_master_map.end(); ++it)
  {
    if (it->first >= dofmap._ownership_range.first && it->first < dofmap._ownership_range.second)
      _slaves_on_process.push_back(it->first);
    
    _all_slaves.push_back(it->first);
  }
  std::size_t slaves_on_this_process = _slaves_on_process.size();
  
  // Compute the new global dimension of dofmap
  dofmap._global_dimension = dofmap._ufc_dofmap->global_dimension() - _all_slaves.size();
  
  // Put the counted slaves on this process on all processes
  std::vector<std::size_t> slaves_on_all_processes(MPI::num_processes()); 
  MPI::all_gather(slaves_on_this_process, slaves_on_all_processes);
  
  // Count the accumulated number of slaves on each process (all slaves on processes with lower process_number than current)
  std::size_t accumulated_slaves = 0;
  for (uint i = 0; i < MPI::process_number(); i++)
    accumulated_slaves += slaves_on_all_processes[i];
  
  // Modify ownership_range due to deleted slave dofs.
  dofmap._ownership_range.first -= accumulated_slaves;
  dofmap._ownership_range.second -= (accumulated_slaves + slaves_on_this_process);
  
  // Renumber all dofs due to deleted slave dofs
  std::sort(_all_slaves.begin(), _all_slaves.end());
  std::vector<std::size_t>::iterator it;  
  for (std::size_t i = 0; i < dofmap._dofmap.size(); i++)
  {
    const std::vector<DolfinIndex>& global_dofs = dofmap.cell_dofs(i); 
    for (uint j = 0; j < dofmap.max_cell_dimension(); j++)
    {
      const std::size_t dof = global_dofs[j];
      
      // lower_bound returns the location of the first item bigger than dof. As such 
      // it counts the number of slaves with dof number smaller than current.
      it = std::lower_bound(_all_slaves.begin(), _all_slaves.end(), dof);
      const std::size_t new_dof = dof - std::size_t(it - _all_slaves.begin());
      dofmap._dofmap[i][j] = new_dof;
    }
  }
  
  // Modify _off_process_owner due to renumbering
  boost::unordered_map<std::size_t, uint> new_off_process_owner;
  for (boost::unordered_map<std::size_t, uint>::iterator op_dof = dofmap._off_process_owner.begin();
      op_dof != dofmap._off_process_owner.end(); ++op_dof)
  {
    const std::size_t old_dof = op_dof->first;
    it = std::lower_bound(_all_slaves.begin(), _all_slaves.end(), old_dof);
    const std::size_t new_dof = old_dof - std::size_t(it - _all_slaves.begin());
    new_off_process_owner[new_dof] = op_dof->second;
  }
  dofmap._off_process_owner = new_off_process_owner;    
  
  // Modify ufc_map_to_dofmap due to renumbering
  boost::unordered_map<std::size_t, std::size_t> new_ufc_map_to_dofmap;
  for (boost::unordered_map<std::size_t, std::size_t>::iterator op = dofmap.ufc_map_to_dofmap.begin();
      op != dofmap.ufc_map_to_dofmap.end(); ++op)
  {
    const std::size_t old_dof = op->second;
    it = std::lower_bound(_all_slaves.begin(), _all_slaves.end(), old_dof);
    const std::size_t new_dof = old_dof - std::size_t(it - _all_slaves.begin());
    new_ufc_map_to_dofmap[op->first] = new_dof;
  }
  dofmap.ufc_map_to_dofmap = new_ufc_map_to_dofmap;  
  
  // Modify _shared_dofs due to renumbering
  boost::unordered_map<std::size_t, std::vector<std::size_t> > new_shared_dofs;
  for (boost::unordered_map<std::size_t, std::vector<std::size_t> >::iterator op = dofmap._shared_dofs.begin();
       op != dofmap._shared_dofs.end(); ++op)
  {
    const std::size_t old_dof = op->first;
    it = std::lower_bound(_all_slaves.begin(), _all_slaves.end(), old_dof);
    const std::size_t new_dof = old_dof - std::size_t(it - _all_slaves.begin());
    new_shared_dofs[new_dof] = op->second;
  }
  dofmap._shared_dofs = new_shared_dofs;  
  
}
