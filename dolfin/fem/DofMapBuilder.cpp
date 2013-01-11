// Copyright (C) 2008-2013 Anders Logg and Ola Skavhaug
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

#include <ufc.h>
#include <boost/random.hpp>
#include <boost/unordered_map.hpp>
#include <boost/serialization/map.hpp>
#include <dolfin/common/tuple_serialization.h>

#include <dolfin/common/Timer.h>
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
#include "DofMapBuilder.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void DofMapBuilder::build(DofMap& dofmap, const Mesh& mesh,
  boost::shared_ptr<const Restriction> restriction,
  const std::map<std::size_t, std::pair<std::size_t, std::size_t> > slave_to_master_facets)
{
  // Sanity checks
  dolfin_assert(dofmap._ufc_dofmap);
  dolfin_assert(dofmap._ufc_dofmap->geometric_dimension() == mesh.geometry().dim());
  dolfin_assert(dofmap._ufc_dofmap->topological_dimension() == mesh.topology().dim());

  // Check that mesh has been ordered
  if (!mesh.ordered())
  {
     dolfin_error("DofMapBuiler.cpp",
                  "create mapping of degrees of freedom",
                  "Mesh is not ordered according to the UFC numbering convention. "
                  "Consider calling mesh.order()");
  }

  // Generate and number required mesh entities
  const bool distributed = MPI::num_processes() > 1;
  const std::size_t D = mesh.topology().dim();
  for (std::size_t d = 1; d <= D; ++d)
  {
    if (dofmap._ufc_dofmap->needs_mesh_entities(d) || (distributed && d == (D - 1)))
    {
      mesh.init(d);
      if (distributed)
        MeshDistributed::number_entities(mesh, d);
    }
  }

  // Build dof map
  const bool reorder = dolfin::parameters["reorder_dofs_serial"];
  build_old(dofmap, mesh, restriction, reorder);
}
//-----------------------------------------------------------------------------
void DofMapBuilder::build_old(DofMap& dofmap,
                          const Mesh& mesh,
                          boost::shared_ptr<const Restriction> restriction,
                          bool reorder)
{
  // Start timer for dofmap initialization
  Timer t0("Init dofmap");

  // Build dofmap based on UFC-provided map. This function does not
  // set local_range
  map restricted_dofs_inverse;
  build_ufc(dofmap, restricted_dofs_inverse, mesh, restriction);

  // Re-order dofmap when running in parallel for process locality
  if (MPI::num_processes() > 1)
  {

    // Periodic modification of the UFC-numbered dofmap.
    // Computes slave-master map and eliminates slaves from dofmap.
    // Computes processes that share master dofs. Recomputes _global_dimension
    //if (mesh.is_periodic())
    //  periodic_modification(dofmap, mesh, global_dofs);

    // Build distributed dof map
    reorder_distributed(dofmap, mesh, restriction, restricted_dofs_inverse);
  }
  else
  {
    //set global_dofs;
    //if (mesh.is_periodic())
    //  periodic_modification(dofmap, mesh, global_dofs);

    // Re-ordering of local dofmap for spatial locality
    if (reorder)
      reorder_local(dofmap, mesh);

    // Set local dof ownbership range
    dofmap._ownership_range = std::make_pair(0, dofmap.global_dimension());
  }
}
//-----------------------------------------------------------------------------
void DofMapBuilder::reorder_local(DofMap& dofmap, const Mesh& mesh)
{
  // Build graph
  Graph graph(dofmap.global_dimension());
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    const std::vector<dolfin::la_index>& dofs0 = dofmap.cell_dofs(cell->index());
    const std::vector<dolfin::la_index>& dofs1 = dofmap.cell_dofs(cell->index());
    std::vector<dolfin::la_index>::const_iterator node;
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

  // Re-number dofs for each cell
  std::vector<std::vector<dolfin::la_index> >::iterator cell_map;
  std::vector<dolfin::la_index>::iterator dof;
  for (cell_map = dofmap._dofmap.begin(); cell_map != dofmap._dofmap.end(); ++cell_map)
    for (dof = cell_map->begin(); dof != cell_map->end(); ++dof)
      *dof = dof_remap[*dof];
}
//-----------------------------------------------------------------------------
void DofMapBuilder::build_ufc(DofMap& dofmap,
                             DofMapBuilder::map& restricted_dofs_inverse,
                             const Mesh& mesh,
                             boost::shared_ptr<const Restriction> restriction)
{
  // Start timer for dofmap initialization
  Timer t0("Init dofmap from UFC dofmap");

  // Allocate space for dof map
  dofmap._dofmap.resize(mesh.num_cells());
  dofmap._off_process_owner.clear();
  dolfin_assert(dofmap._ufc_dofmap);

  // FIXME: Remove restricted_dofs_inverse if not needed

  // Maps used to renumber dofs for restricted meshes
  map restricted_dofs;         // map from old to new dof
  //map restricted_dofs_inverse; // map from new to old dof

  // Store bumber of global entities
  std::vector<std::size_t> num_global_mesh_entities(mesh.topology().dim() + 1);
  for (std::size_t d = 0; d < num_global_mesh_entities.size(); d++)
    num_global_mesh_entities[d] = mesh.size_global(d);

  // Holder for UFC support 64-bit integers
  std::vector<std::size_t> ufc_dofs;

  // Build dofmap from ufc::dofmap
  dolfin::UFCCell ufc_cell(mesh);
  for (dolfin::CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Skip cells not included in restriction
    if (restriction && !restriction->contains(*cell))
      continue;

    // Update UFC cell
    ufc_cell.update(*cell);

    // Get standard local dimension
    const std::size_t local_dim = dofmap._ufc_dofmap->local_dimension(ufc_cell);

    // Get container for cell dofs
    std::vector<dolfin::la_index>& cell_dofs = dofmap._dofmap[cell->index()];
    cell_dofs.resize(local_dim);

    // Tabulate standard UFC dof map
    ufc_dofs.resize(local_dim);
    dofmap._ufc_dofmap->tabulate_dofs(&ufc_dofs[0],
                                      num_global_mesh_entities, ufc_cell);
    std::copy(ufc_dofs.begin(), ufc_dofs.end(), cell_dofs.begin());

    // Renumber dofs if mesh is restricted
    if (restriction)
    {
      for (std::size_t i = 0; i < cell_dofs.size(); i++)
      {
        map_iterator it = restricted_dofs.find(cell_dofs[i]);
        if (it == restricted_dofs.end())
        {
          const std::size_t dof = restricted_dofs.size();
          restricted_dofs[cell_dofs[i]] = dof;
          restricted_dofs_inverse[dof] = cell_dofs[i];
          cell_dofs[i] = dof;
        }
        else
          cell_dofs[i] = it->second;
      }
    }
  }

  // Set global dimension
  if (restriction)
    dofmap._global_dimension = restricted_dofs.size();
  else
  {
    dofmap._global_dimension
      = dofmap._ufc_dofmap->global_dimension(num_global_mesh_entities);
  }

  //dofmap._ownership_range = std::make_pair(0, dofmap.global_dimension());
}
//-----------------------------------------------------------------------------
void DofMapBuilder::build_sub_map(DofMap& sub_dofmap,
                                  const DofMap& parent_dofmap,
                                  const std::vector<std::size_t>& component,
                                  const Mesh& mesh)
{
  // Note: Ownership range is set to zero since dofmap is a view

  dolfin_assert(!component.empty());

  // Collect global mesh entity dimensions in a vector
  std::vector<std::size_t> num_global_mesh_entities(mesh.topology().dim() + 1);
  for (std::size_t d = 0; d < num_global_mesh_entities.size(); d++)
    num_global_mesh_entities[d] = mesh.size_global(d);

  // Initialise offset from parent
  std::size_t offset = parent_dofmap._ufc_offset;

  // Get parent UFC dof map
  const ufc::dofmap& parent_ufc_dofmap = *(parent_dofmap._ufc_dofmap);

  // Extract ufc sub-dofmap from parent and get offset
  sub_dofmap._ufc_dofmap.reset(extract_ufc_sub_dofmap(parent_ufc_dofmap,
                                                      offset,
                                                      component, mesh));
  dolfin_assert(sub_dofmap._ufc_dofmap);

  // Check for dimensional consistency between the dofmap and mesh
  //check_dimensional_consistency(*_ufc_dofmap, mesh);

  // Set UFC offset
  sub_dofmap._ufc_offset = offset;

  // Check dimensional consistency between UFC dofmap and the mesh
  //check_provided_entities(*_ufc_dofmap, mesh);

  // Resize dofmap data structure
  sub_dofmap._dofmap.resize(mesh.num_cells());

  // Set to hold slave dofs on current processor
  //std::set<std::size_t> slave_dofs;

  // Store original _slave_master_map on this sub_dofmap
  //_slave_master_map = parent_dofmap._slave_master_map;

  // Holder for copying UFC std::size_t dof maps into the a dof map that
  // is consistent with the linear algebra backend
  std::vector<std::size_t> dof_holder;

  // Build sub-map based on UFC dofmap
  UFCCell ufc_cell(mesh);
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    const std::size_t cell_index = cell->index();

    // Update to current cell
    ufc_cell.update(*cell);

    // Resize list for cell
    sub_dofmap._dofmap[cell_index].resize(sub_dofmap._ufc_dofmap->local_dimension(ufc_cell));
    dof_holder.resize(sub_dofmap._ufc_dofmap->local_dimension(ufc_cell));

    // Tabulate sub-dofs on cell (using UFC map)
    sub_dofmap._ufc_dofmap->tabulate_dofs(&dof_holder[0], num_global_mesh_entities, ufc_cell);

    // Add UFC offset
    for (unsigned int i=0; i < dof_holder.size(); i++)
      dof_holder[i] += offset;

    /*
    if (mesh.is_periodic() && !_slave_master_map.empty())
    {
      // Check for slaves and modify
      std::map<std::size_t, std::size_t>::const_iterator slave_it;
      for (unsigned int i = 0; i < dof_holder.size();ck i++)
      {
        const std::size_t dof = dof_holder[i];
        slave_it = _slave_master_map.find(dof);
        if (slave_it != _slave_master_map.end())
        {
          dof_holder[i] = slave_it->second; // Replace slave with master
          slave_dofs.insert(slave_it->first);
        }
      }
    }
    */
    std::copy(dof_holder.begin(), dof_holder.end(), sub_dofmap._dofmap[cell_index].begin());
  }

  /*
  if (mesh.is_periodic() && !_slave_master_map.empty())
  {
    // Periodic meshes need to renumber UFC-numbered dofs due to elimination of slave dofs
    // For faster search get a vector of all slaves on parent dofmap (or parent of parent, aka the owner)
    std::vector<std::size_t> parent_slaves;
    for (std::map<std::size_t, std::size_t>::const_iterator it = _slave_master_map.begin();
                              it != _slave_master_map.end(); ++it)
    {
      parent_slaves.push_back(it->first);
    }

    std::vector<std::size_t>::iterator it;
    std::vector<std::vector<dolfin::la_index> >::iterator cell_map;
    std::vector<dolfin::la_index>::iterator dof;
    for (cell_map = _dofmap.begin(); cell_map != _dofmap.end(); ++cell_map)
    {
      for (dof = cell_map->begin(); dof != cell_map->end(); ++dof)
      {
        it = std::lower_bound(parent_slaves.begin(), parent_slaves.end(), *dof);
        *dof -= std::size_t(it - parent_slaves.begin());
      }
    }

    // Reduce the local slaves onto all processes to eliminate duplicates
    std::vector<std::set<std::size_t> > all_slave_dofs;
    MPI::all_gather(slave_dofs, all_slave_dofs);
    for (std::size_t i = 0; i < all_slave_dofs.size(); i++)
      if (i != MPI::process_number())
        slave_dofs.insert(all_slave_dofs[i].begin(), all_slave_dofs[i].end());

    // Set global dimension
    _global_dimension = _ufc_dofmap->global_dimension(num_global_mesh_entities) - slave_dofs.size();

  }
  else
  */
  {
    // Set global dimension
    sub_dofmap._global_dimension
      = sub_dofmap._ufc_dofmap->global_dimension(num_global_mesh_entities);
  }

  // Modify dofmap for non-UFC numbering
  sub_dofmap.ufc_map_to_dofmap.clear();
  sub_dofmap._off_process_owner.clear();
  sub_dofmap._shared_dofs.clear();
  sub_dofmap._neighbours.clear();
  if (!parent_dofmap.ufc_map_to_dofmap.empty())
  {
    boost::unordered_map<std::size_t, std::size_t>::const_iterator ufc_to_current_dof;
    std::vector<std::vector<dolfin::la_index> >::iterator cell_map;
    std::vector<dolfin::la_index>::iterator dof;
    for (cell_map = sub_dofmap._dofmap.begin();
        cell_map != sub_dofmap._dofmap.end(); ++cell_map)
    {
      for (dof = cell_map->begin(); dof != cell_map->end(); ++dof)
      {
        // Get dof index
        ufc_to_current_dof = parent_dofmap.ufc_map_to_dofmap.find(*dof);
        dolfin_assert(ufc_to_current_dof != parent_dofmap.ufc_map_to_dofmap.end());

        // Add to ufc-to-current dof map
        sub_dofmap.ufc_map_to_dofmap.insert(*ufc_to_current_dof);

        // Set dof index
        *dof = ufc_to_current_dof->second;

        // Add to off-process dof owner map
        boost::unordered_map<std::size_t, std::size_t>::const_iterator
          parent_off_proc = parent_dofmap._off_process_owner.find(*dof);
        if (parent_off_proc != parent_dofmap._off_process_owner.end())
          sub_dofmap._off_process_owner.insert(*parent_off_proc);

        // Add to shared-dof process map, and update the set of neighbours
        boost::unordered_map<std::size_t, std::vector<std::size_t> >::const_iterator parent_shared
          = parent_dofmap._shared_dofs.find(*dof);
        if (parent_shared != parent_dofmap._shared_dofs.end())
        {
          sub_dofmap._shared_dofs.insert(*parent_shared);
          sub_dofmap._neighbours.insert(parent_shared->second.begin(), parent_shared->second.end());
        }
      }
    }
  }
}
//-----------------------------------------------------------------------------
void DofMapBuilder::reorder_distributed(DofMap& dofmap,
                                      const Mesh& mesh,
                                      boost::shared_ptr<const Restriction> restriction,
                                      const map& restricted_dofs_inverse)
{
  // Build set of global dofs
  set global_dofs = compute_global_dofs(dofmap, mesh);

  // Create data structures
  DofMapBuilder::set owned_dofs, shared_owned_dofs, shared_unowned_dofs;
  DofMapBuilder::vec_map shared_dof_processes;

  // Computed owned and shared dofs (and owned and un-owned)
  compute_dof_ownership(owned_dofs, shared_owned_dofs, shared_unowned_dofs,
                    shared_dof_processes, dofmap, global_dofs, mesh,
                    restriction, restricted_dofs_inverse);

  // Renumber owned dofs and receive new numbering for unowned shared dofs
  parallel_renumber(owned_dofs, shared_owned_dofs, shared_unowned_dofs,
                    shared_dof_processes, dofmap, mesh,
                    restriction, restricted_dofs_inverse);
}
//-----------------------------------------------------------------------------
void DofMapBuilder::compute_dof_ownership(set& owned_dofs, set& shared_owned_dofs,
                                      set& shared_unowned_dofs,
                                      vec_map& shared_dof_processes,
                                      DofMap& dofmap,
                                      const DofMapBuilder::set& global_dofs,
                                      const Mesh& mesh,
                                      boost::shared_ptr<const Restriction> restriction,
                                      const map& restricted_dofs_inverse)
{
  log(TRACE, "Determining dof ownership for parallel dof map");

  // Create a random number generator for ownership 'voting'
  boost::mt19937 engine(MPI::process_number());
  boost::uniform_int<> distribution(0, 100000000);
  boost::variate_generator<boost::mt19937&, boost::uniform_int<> >
    rng(engine, distribution);

  // Clear data structures
  owned_dofs.clear();
  shared_owned_dofs.clear();
  shared_unowned_dofs.clear();

  // Data structures for computing ownership
  boost::unordered_map<std::size_t, std::size_t> dof_vote;
  std::vector<std::size_t> facet_dofs(dofmap.num_facet_dofs());

  // Communication buffer
  std::vector<std::size_t> send_buffer;

  // Extract the interior boundary
  BoundaryMesh interior_boundary;
  interior_boundary.init_interior_boundary(mesh);

  // Build set of dofs on process boundary (assume all are owned by this process)
  const MeshFunction<std::size_t>& cell_map = interior_boundary.cell_map();
  if (!cell_map.empty())
  {
    for (CellIterator _f(interior_boundary); !_f.end(); ++_f)
    {
      // Get boundary facet
      Facet f(mesh, cell_map[*_f]);

      // Get cell to which facet belongs (pick first)
      Cell c(mesh, f.entities(mesh.topology().dim())[0]);

      // Skip cells not included in restriction
      if (restriction && !restriction->contains(c))
        continue;

      // Tabulate dofs on cell
      const std::vector<dolfin::la_index>& cell_dofs = dofmap.cell_dofs(c.index());

      // Tabulate which dofs are on the facet
      dofmap.tabulate_facet_dofs(facet_dofs, c.index(f));

      // Insert shared dofs into set and assign a 'vote'
      for (std::size_t i = 0; i < dofmap.num_facet_dofs(); i++)
      {
        // Get facet dof
        size_t facet_dof = cell_dofs[facet_dofs[i]];

        // Map back to original (and common) numbering for restricted space
        if (restriction)
        {
          const map_iterator it = restricted_dofs_inverse.find(facet_dof);
          dolfin_assert(it != restricted_dofs_inverse.end());
          facet_dof = it->second;
        }

        // Add to list of shared dofs
        if (shared_owned_dofs.find(facet_dof) == shared_owned_dofs.end())
        {
          shared_owned_dofs.insert(facet_dof);
          dof_vote[facet_dof] = rng();

          send_buffer.push_back(facet_dof);
          send_buffer.push_back(dof_vote[facet_dof]);
        }
      }
    }
  }

  // Periodic contribution because the boundary between periodic domains
  // is not captured by the interior_boundary of a BoundaryMesh
  /*
  std::map<std::size_t, boost::unordered_set<std::size_t> >::const_iterator map_it;
  for (map_it = dofmap._master_processes.begin();
       map_it != dofmap._master_processes.end(); ++map_it)
  {
    std::size_t master_dof = map_it->first;
    for (boost::unordered_set<std::size_t>::const_iterator sit = map_it->second.begin();
          sit != map_it->second.end(); ++sit)
    {
      if (*sit == MPI::process_number())
      {
        if (shared_owned_dofs.find(master_dof) == shared_owned_dofs.end())
        {
          shared_owned_dofs.insert(master_dof);
          dof_vote[master_dof] = rng();

          send_buffer.push_back(master_dof);
          send_buffer.push_back(dof_vote[master_dof]);
        }
      }
    }
  }
  */

  // Decide ownership of shared dofs
  const std::size_t num_proc = MPI::num_processes();
  const std::size_t proc_num = MPI::process_number();
  std::vector<std::size_t> recv_buffer;
  for (std::size_t k = 1; k < MPI::num_processes(); ++k)
  {
    const std::size_t src  = (proc_num - k + num_proc) % num_proc;
    const std::size_t dest = (proc_num + k) % num_proc;
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
    const std::vector<dolfin::la_index>& cell_dofs = dofmap.cell_dofs(cell->index());
    const std::size_t cell_dimension = dofmap.cell_dimension(cell->index());
    for (std::size_t i = 0; i < cell_dimension; ++i)
    {
      // Get cell dof
      size_t cell_dof = cell_dofs[i];

      // Map back to original (and common) numbering for restricted space
      if (restriction)
      {
        const map_iterator it = restricted_dofs_inverse.find(cell_dof);
        dolfin_assert(it != restricted_dofs_inverse.end());
        cell_dof = it->second;
      }

      // Mark dof as owned if in unowned set
      if (shared_unowned_dofs.find(cell_dof) == shared_unowned_dofs.end())
        owned_dofs.insert(cell_dof);
    }
  }

  // Check or set global dimension
  if (restriction)
  {
    // Global dimension for restricted space needs to be computed here
    // since it is not know by the UFC dof map.
    const std::size_t _owned_dim = owned_dofs.size();
    const std::size_t _global_dimension = MPI::sum(_owned_dim);
    dofmap._global_dimension = _global_dimension;
  }
  else
  {
    const std::size_t _owned_dim = owned_dofs.size();
    dolfin_assert(MPI::sum(_owned_dim) == dofmap.global_dimension());
  }

  log(TRACE, "Finished determining dof ownership for parallel dof map");
}
//-----------------------------------------------------------------------------
void DofMapBuilder::parallel_renumber(const set& owned_dofs,
                                      const set& shared_owned_dofs,
                                      const set& shared_unowned_dofs,
                                      const vec_map& shared_dof_processes,
                                      DofMap& dofmap,
                                      const Mesh& mesh,
                                      boost::shared_ptr<const Restriction> restriction,
                                      const map& restricted_dofs_inverse)
{
  log(TRACE, "Renumber dofs for parallel dof map");

  // FIXME: Handle double-renumbered dof map
  if (!dofmap.ufc_map_to_dofmap.empty())
  {
    dolfin_error("DofMapBuilder.cpp",
                 "compute parallel renumbering of degrees of freedom",
                 "The degree of freedom mapping cannot (yet) be renumbered twice");
  }

  const std::vector<std::vector<dolfin::la_index> >& old_dofmap = dofmap._dofmap;
  std::vector<std::vector<dolfin::la_index> > new_dofmap(old_dofmap.size());
  dolfin_assert(old_dofmap.size() == mesh.num_cells());

  // Compute offset for owned and non-shared dofs
  const std::size_t process_offset = MPI::global_offset(owned_dofs.size(), true);

  // Clear some data
  dofmap._off_process_owner.clear();

  // Create graph
  Graph graph(owned_dofs.size());

  // Build graph for re-ordering. Below block is scoped to clear working
  // data structures once graph is constructed.
  {
    // Create contiguous local numbering for locally owned dofs
    std::size_t my_counter = 0;
    boost::unordered_map<std::size_t, std::size_t> my_old_to_new_dof_index;
    for (set_iterator owned_dof = owned_dofs.begin(); owned_dof != owned_dofs.end(); ++owned_dof, my_counter++)
      my_old_to_new_dof_index[*owned_dof] = my_counter;

    // Build local graph based on old dof map with contiguous numbering
    for (std::size_t cell = 0; cell < old_dofmap.size(); ++cell)
    {
      const std::vector<dolfin::la_index>& dofs0 = dofmap.cell_dofs(cell);
      const std::vector<dolfin::la_index>& dofs1 = dofmap.cell_dofs(cell);
      std::vector<dolfin::la_index>::const_iterator node0, node1;
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
  const std::size_t num_proc = MPI::num_processes();
  const std::size_t proc_num = MPI::process_number();
  std::vector<std::size_t> recv_buffer;
  for (std::size_t k = 1; k < MPI::num_processes(); ++k)
  {
    const std::size_t src  = (proc_num - k + num_proc) % num_proc;
    const std::size_t dest = (proc_num + k) % num_proc;
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
    boost::unordered_map<std::size_t, std::size_t>::const_iterator
      new_index = old_to_new_dof_index.find(it->first);
    if (new_index == old_to_new_dof_index.end())
      dofmap._shared_dofs.insert(*it);
    else
      dofmap._shared_dofs.insert(std::make_pair(new_index->second, it->second));
    dofmap._neighbours.insert(it->second.begin(), it->second.end());
  }

  // Build new dof map
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Skip cells not included in restriction
    if (restriction && !restriction->contains(*cell))
      continue;

    // Get cell index and dimension
    const std::size_t cell_index = cell->index();
    const std::size_t cell_dimension = dofmap.cell_dimension(cell_index);

    // Resize cell map and insert dofs
    new_dofmap[cell_index].resize(cell_dimension);
    for (std::size_t i = 0; i < cell_dimension; ++i)
    {
      // Get old dof
      std::size_t old_index = old_dofmap[cell_index][i];

      // Map back to original (and common) numbering for restricted space
      if (restriction)
      {
        const map_iterator it = restricted_dofs_inverse.find(old_index);
        dolfin_assert(it != restricted_dofs_inverse.end());
        old_index = it->second;
      }

      // Insert dof
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
                                                      const Mesh& mesh)
{
  // Wrap UFC dof map
  boost::shared_ptr<const ufc::dofmap> _dofmap(dofmap._ufc_dofmap.get(),
                                               NoDeleter());

  // Compute global dof indices
  std::size_t offset = 0;
  set global_dof_indices;
  compute_global_dofs(global_dof_indices, offset, _dofmap, mesh);

  return global_dof_indices;
}
//-----------------------------------------------------------------------------
void DofMapBuilder::compute_global_dofs(DofMapBuilder::set& global_dofs,
                                        std::size_t& offset,
                                        boost::shared_ptr<const ufc::dofmap> dofmap,
                                        const Mesh& mesh)
{
  dolfin_assert(dofmap);
  const std::size_t D = mesh.topology().dim();

  // Store global entity dimensions in vector // TODO: These lines are repeated a few times
  std::vector<std::size_t> num_global_mesh_entities(mesh.topology().dim() + 1);
  for (std::size_t d = 0; d < num_global_mesh_entities.size(); d++)
    num_global_mesh_entities[d] = mesh.size_global(d);

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
      if (dofmap->global_dimension(num_global_mesh_entities) != 1)
      {
        dolfin_error("DofMapBuilder.cpp",
                     "compute global degrees of freedom",
                     "Global degree of freedom has dimension != 1");
      }

      // Create dummy cell argument to tabulate single global dof
      boost::scoped_ptr<ufc::cell> ufc_cell(new ufc::cell);
      std::size_t dof = 0;
      dofmap->tabulate_dofs(&dof, num_global_mesh_entities, *ufc_cell);

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
      DofMap::check_provided_entities(*sub_dofmap, mesh);

      compute_global_dofs(global_dofs, offset, sub_dofmap, mesh);

      // Get offset
      if (sub_dofmap->num_sub_dofmaps() == 0)
        offset += sub_dofmap->global_dimension(num_global_mesh_entities);
    }
  }
}
//-----------------------------------------------------------------------------
void DofMapBuilder::extract_dof_pairs(const DofMap& dofmap, const Mesh& mesh,
    periodic_map& _slave_master_map,
    std::map<std::size_t, boost::unordered_set<std::size_t> >& _master_processes)
{
  /*
  Timer t0("Extracting dof pairs");

  const std::size_t num_sub_dofmaps = dofmap._ufc_dofmap->num_sub_dofmaps();
  if (num_sub_dofmaps > 0)
  {
    // Call recursively for all sub_dofmaps
    std::vector<std::size_t> component(1);
    for (std::size_t i=0; i<num_sub_dofmaps; i++)
    {
      component[0] = i;
      DofMap* sub_dofmap = dofmap.extract_sub_dofmap(component, mesh);
      extract_dof_pairs(*sub_dofmap, mesh, _slave_master_map, _master_processes);
    }
    return;
  }

  // Get dimensions
  const std::size_t tdim = mesh.topology().dim();
  const std::size_t gdim = mesh.geometry().dim();

  // Arrays used for mapping coordinates
  std::vector<double> x(gdim);
  std::vector<double> y(gdim);
  std::vector<double> dx(gdim);

  const std::size_t process_number = MPI::process_number();

  // Declare some variables used to hold information on each facet
  std::vector<std::size_t> facet_dofs(dofmap.num_facet_dofs());
  boost::multi_array<double, 2> facet_coors(boost::extents[dofmap.max_cell_dimension()][gdim]);

  // First send all relevant information on the slave facets to adjoining master.
  // Create a type to hold all info that will be sent. The info is:
  //    (periodic facet id, global slave dofs and coordinates of all slave dofs)
  typedef boost::tuples::tuple<std::size_t, std::vector<std::size_t>,
                    std::vector<std::vector<double> > > facet_info_type;
  typedef std::vector<facet_info_type> facets_info_type;
  typedef std::map<std::size_t, facets_info_type> facet_info_map_type;
  typedef std::map<std::size_t, std::vector<std::vector<double> > > coor_map_type;

  // Use a master_slave map for faster master search. Only the slave_master_map is stored
  periodic_map _current_master_slave_map;
  periodic_map _current_slave_master_map;

  // Run over periodic domains and build the global _slave_master_map
  for (std::size_t periodic_domain = 0; periodic_domain < mesh.num_periodic_domains(); periodic_domain++)
  {
    // Get periodic info
    //facet_pair_type facet_pairs = mesh.get_periodic_facet_pairs(periodic_domain);
    //dx = mesh.get_periodic_distance(periodic_domain);  // Distance between periodic domains
    //const std::size_t num_periodic_faces = facet_pairs.size();

    // Map to hold all information being sent from slaves to masters
    facet_info_map_type facet_info_map;

    // Communicating processes
    std::set<std::size_t> communicating_processors;

    // Run over periodic facets and collect all info that should be sent
    for (std::size_t i = 0; i < num_periodic_faces; i++)
    {
      const std::size_t master_process = facet_pairs[i].first.second;
      const std::size_t slave_process = facet_pairs[i].second.second;

      if (master_process == process_number)
        communicating_processors.insert(slave_process);

      if (slave_process == process_number)
      {
        // Get dofs and dof-coordinates from slave facet
        const Facet facet(mesh, facet_pairs[i].second.first);
        const Cell cell(mesh, facet.entities(tdim)[0]);
        const std::vector<dolfin::la_index> global_dofs = dofmap.cell_dofs(cell.index());
        dofmap.tabulate_coordinates(facet_coors, cell);
        dofmap.tabulate_facet_dofs(facet_dofs, cell.index(facet));
        communicating_processors.insert(master_process);

        std::vector<std::vector<double> > coors_of_dofs;
        std::vector<std::size_t> dofs_on_facet;
        for (std::size_t k = 0; k < dofmap.num_facet_dofs(); k++)
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
      for (std::size_t j = 0; j < info_list.size(); j++)
      {
        std::size_t i = info_list[j].get<0>();  // The periodic facet number
        slave_dofs[i] = info_list[j].get<1>();
        coors_on_slave[i] = info_list[j].get<2>();
      }
    }

    // Declare map used to hold global matching pairs of dofs on this process
    periodic_map matching_dofs;

    // Map from master dof to processes sharing it for one single periodic direction
    std::map<std::size_t, std::pair<std::size_t, std::size_t> > master_processes;

    // Run over periodic facets and locate matching dof pairs
    for (std::size_t i = 0; i < num_periodic_faces; i++)
    {
      const std::size_t master_process = facet_pairs[i].first.second;
      const std::size_t slave_process = facet_pairs[i].second.second;

      // Compare only on master process
      if (master_process == process_number)
      {
        // Get info from master facet: cell, dofs, coordinates
        const Facet facet(mesh, facet_pairs[i].first.first);
        const Cell cell(mesh, facet.entities(tdim)[0]);
        const std::vector<dolfin::la_index> global_dofs = dofmap.cell_dofs(cell.index());
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
//           for(std::size_t l = 0; l < gdim; l++)
//             error += std::abs(x[l] - y[l] + dx[l]);
//
//           if (error < 1.0e-12)    // Match! Assuming the dofs are laid out in the same order
//                                            // on the facet the remaining are simply copied without control
//           {
//             for (std::size_t j = 0; j < dofmap.num_facet_dofs(); j++)
//             {
//               master_dof = global_dofs[facet_dofs[j]];
//               matching_dofs[master_dof] = slave_dofs[i][j];
//             }
//           }
//           else  // If local dofs 0/0 don't match the order must be opposite
//           {
//             for (std::size_t j = 0; j < dofmap.num_facet_dofs(); j++)
//             {
//               master_dof = global_dofs[facet_dofs[j]];
//               matching_dofs[master_dof] = slave_dofs[i][dofmap.num_facet_dofs()-j-1];
//             }
//           }
//         }
//         ////////////////////////////////////////////////////////

        // Match master and slave dofs and put pair in map
        for (std::size_t j = 0; j < dofmap.num_facet_dofs(); j++)
        {
          // Get global master dof and coordinates
          std::size_t master_dof = global_dofs[facet_dofs[j]];

          // Check new master_dofs only
          if (matching_dofs.find(master_dof) == matching_dofs.end())
          {
            std::copy(facet_coors[facet_dofs[j]].begin(),
                      facet_coors[facet_dofs[j]].end(), x.begin());

            for (std::size_t k = 0; k < dofmap.num_facet_dofs(); k++)
            {
              // Look for a match in coordinates
              y = coors_on_slave[i][k];
              double error = 0.;
              for(std::size_t l = 0; l < gdim; l++)
                error += std::abs(x[l] - y[l] + dx[l]);

              if (error < 1.0e-12) // Match! Store master and slave in matching_dofs
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
          // Store the processes that share the slave/master pair
          master_processes[master_dof] = std::make_pair(master_process, slave_process);

        }  // Finished on facet. Move to next periodic facet pair
      }
    }   // Finished all periodic pairs on periodic domain

    // At this point there should be a match between dofs in matching_dofs.
    // Put the matching dof pairs on all processes
    std::vector<periodic_map> all_dof_pairs;
    MPI::all_gather(matching_dofs, all_dof_pairs);

    // Store also on all processors the processes that share each master dof
    std::vector<std::map<std::size_t, std::pair<std::size_t, std::size_t> > > all_process_pairs;
    MPI::all_gather(master_processes, all_process_pairs);

    // Add to the global _slave_master_map and _master_processes
    for (std::size_t i = 0; i < all_dof_pairs.size(); i++)
    {
      periodic_map matching_dofs = all_dof_pairs[i];
      std::map<std::size_t, std::pair<std::size_t, std::size_t> > master_processes = all_process_pairs[i];

      for (periodic_map_iterator it = matching_dofs.begin();
                                 it != matching_dofs.end(); ++it)
      {
        std::size_t master_dof = it->first;
        std::size_t slave_dof = it->second;

        if (periodic_domain == 0)     // First periodic direction, just copy down
        {
          _current_slave_master_map[slave_dof] = master_dof;
          _current_master_slave_map[master_dof] = slave_dof;
          _master_processes[master_dof].insert(master_processes[master_dof].first);
          _master_processes[master_dof].insert(master_processes[master_dof].second);
        }
        else
        {
          // At this point we need to do something clever in case of more than one
          // periodic direction. For example, a rectangular mesh with two periodic
          // directions will have four corners that should be equal. In that case
          // we will here end up with one master and three slaves of the same master.
          // A 3D Cube with 8 corners should similarily have 7 slaves of the same master.
          if (_current_slave_master_map.find(slave_dof) == _current_slave_master_map.end())
          {
            // If the slave does not exist, then simply add to maps.
            _current_slave_master_map[slave_dof] = master_dof;
            _current_master_slave_map[master_dof] = slave_dof;
            _master_processes[master_dof].insert(master_processes[master_dof].first);
            _master_processes[master_dof].insert(master_processes[master_dof].second);

            // Need to check if this is a "corner dof" by checking
            // whether the slave has been used as a master previously.
            if (_current_master_slave_map.find(slave_dof) != _current_master_slave_map.end())
            {
              // The slave dof has been previously used as master
              // Get the old slave of slave_dof
              std::size_t old_slave = _current_master_slave_map[slave_dof];

              // Use latest master for previous slave as well
              _current_slave_master_map[old_slave] = master_dof;

              // slave_dof should no longer be used as a master.
              _current_master_slave_map.erase(slave_dof);

              // Get the processes that share slave_dof and add to the ultimate master
              _master_processes[master_dof].insert(_master_processes[slave_dof].begin(), _master_processes[slave_dof].end());
              _master_processes.erase(slave_dof);
            }
          }
          else // Corner dof. The slave_dof exists as slave from before
          {
            // Check if the master_dof has been used previously as a slave
            if (_current_slave_master_map.find(master_dof) != _current_slave_master_map.end())
            {
              // Get the dof that will be master for all corner dofs
              std::size_t ultimate_master = _current_slave_master_map[master_dof];

              // Get the previous master
              std::size_t old_master = _current_slave_master_map[slave_dof];

              // Use ultimate master for the current slave as well
              _current_slave_master_map[slave_dof] = ultimate_master;

              // Put the processes that shared the old master in the ultimate master set
              _master_processes[ultimate_master].insert(_master_processes[old_master].begin(), _master_processes[old_master].end());

              // Update _current_master_slave_map
              _current_master_slave_map[ultimate_master] = slave_dof;
            }
          }
        }
      }
    }
  } // Finished with all periodic domains

  // Update the global _slave_master_map (the map for all sub_dofmaps)
  _slave_master_map.insert(_current_slave_master_map.begin(), _current_slave_master_map.end());

//     cout << "Map" << endl;
//     for (periodic_map_iterator it = _slave_master_map.begin();
//                                it != _slave_master_map.end(); ++it)
//     {
//       cout << "   " << it->first << " " << it->second << endl;
//     }
//     for (std::map<std::size_t, set>::iterator it = _master_processes.begin();
//                                     it != _master_processes.end(); ++it)
//     {
//       cout << " Master " << it->first << endl;
//       for (set_iterator sit = it->second.begin(); sit != it->second.end(); ++sit)
//          cout << " " << *sit;
//       cout << endl;
//     }
*/
}

void DofMapBuilder::periodic_modification(DofMap& dofmap, const Mesh& mesh, set& global_dofs)
{
  /*
  Timer t0("Periodic dofmap modification");

  periodic_map _slave_master_map;
  std::map<std::size_t, boost::unordered_set<std::size_t> > _master_processes;

  // Recursively extract a map from slaves to master dofs for all sub-dofmaps of dofmap.
  // Create also a map of the processes that share the master dofs.
  extract_dof_pairs(dofmap, mesh, _slave_master_map, _master_processes);

  // Store the global _slave_master_map
  dofmap._slave_master_map = _slave_master_map;

  // Get topological dimension
  const std::size_t tdim = mesh.topology().dim();

  // Eliminate all slaves from the dofmap by placing the master in all locations
  // where a slave is found. For efficiency first find all cells that could contain a slave.
  set cells_with_slave;
  for (std::size_t periodic_domain = 0; periodic_domain < mesh.num_periodic_domains(); periodic_domain++)
  {
    // Get periodic facet-to-facet map
    facet_pair_type facet_pairs = mesh.get_periodic_facet_pairs(periodic_domain);

    for (std::size_t i = 0; i < facet_pairs.size(); i++)
    {
      const std::size_t slave_process = facet_pairs[i].second.second;
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
          for (std::size_t j = 0; j < facet.num_entities(0); j++)
          {
            const Vertex v(mesh, facet_vertices[j]);
            const std::size_t* vertex_cells = v.entities(tdim);
            for (std::size_t k = 0; k < v.num_entities(tdim); k++)
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
    const std::vector<dolfin::la_index> global_dofs = dofmap.cell_dofs(*it);
    for (std::size_t j = 0; j < dofmap.max_cell_dimension(); j++)
    {
      const std::size_t dof = global_dofs[j];
      if (_slave_master_map.find(dof) != _slave_master_map.end())
      {
        dofmap._dofmap[*it][j] = _slave_master_map[dof]; // Switch slave for master
      }
    }
  }

  // At this point the slaves should be completely removed from the dofmap
  // and the global dimension of the dofmap can be reduced.
  // To do this:
  //   1) Compute the total number of slaves that has been eliminated
  //   2) Renumber all UFC-numbering based dofs by subtracting current
  //        dof-number with the number of eliminated slaves with a number
  //        less than the current
  //   3) Recompute global_dimension (set _global_dimension)
  //   4) Renumber global_dofs and _master_processes

  // Get the keys of _slave_master_map for faster search (to avoid std::distance)
  std::vector<std::size_t> _all_slaves;
  for (periodic_map_iterator it = _slave_master_map.begin();
                             it != _slave_master_map.end(); ++it)
  {
    _all_slaves.push_back(it->first);
  }

  // Store global entity dimensions in vector // TODO: These lines are repeated a few times
  std::vector<std::size_t> num_global_mesh_entities(mesh.topology().dim() + 1);
  for (std::size_t d = 0; d < num_global_mesh_entities.size(); d++)
    num_global_mesh_entities[d] = mesh.size_global(d);

  // Compute the new global dimension of dofmap
  dofmap._global_dimension = dofmap._ufc_dofmap->global_dimension(num_global_mesh_entities) - _slave_master_map.size();

  // Renumber all UFC-numbering based dofs due to deleted slave dofs
  // Could here alternatively create a map
  std::vector<std::size_t>::iterator it;
  periodic_map_iterator pit;
  std::vector<std::vector<dolfin::la_index> >::iterator cell_map;
  std::vector<dolfin::la_index>::iterator dof;
  for (cell_map = dofmap._dofmap.begin(); cell_map != dofmap._dofmap.end(); ++cell_map)
  {
    for (dof = cell_map->begin(); dof != cell_map->end(); ++dof)
    {
      it = std::lower_bound(_all_slaves.begin(), _all_slaves.end(), *dof);
      *dof -= std::size_t(it - _all_slaves.begin());
//      pit = _slave_master_map.lower_bound(*dof);
//      *dof -= std::distance(_slave_master_map.begin(), pit); // slow
    }
  }

  // Renumber _master_processes and global_dofs
  std::map<std::size_t, boost::unordered_set<std::size_t> > new_master_processes;
  for (std::map<std::size_t, boost::unordered_set<std::size_t> >::iterator sit = _master_processes.begin();
                                            sit != _master_processes.end(); ++sit)
  {
    const std::size_t dof = sit->first;
    it = std::lower_bound(_all_slaves.begin(), _all_slaves.end(), dof);
    const std::size_t new_dof = dof - std::size_t(it - _all_slaves.begin());
    new_master_processes[new_dof] = sit->second;
  }
  dofmap._master_processes = new_master_processes;

  set new_global_dofs;
  for (set_iterator sit = global_dofs.begin(); sit != global_dofs.end(); ++sit)
  {
    it = std::lower_bound(_all_slaves.begin(), _all_slaves.end(), *sit);
    new_global_dofs.insert(*sit - std::size_t(it - _all_slaves.begin()));
  }
  global_dofs = new_global_dofs;
  */
}
//-----------------------------------------------------------------------------
ufc::dofmap* DofMapBuilder::extract_ufc_sub_dofmap(const ufc::dofmap& ufc_dofmap,
                                            std::size_t& offset,
                                            const std::vector<std::size_t>& component,
                                            const Mesh& mesh)
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

  // Store global entity dimensions in vector
  std::vector<std::size_t> num_global_mesh_entities(mesh.topology().dim() + 1);
  for (std::size_t d = 0; d < num_global_mesh_entities.size(); d++)
    num_global_mesh_entities[d] = mesh.size_global(d);

  // Add to offset if necessary
  for (std::size_t i = 0; i < component[0]; i++)
  {
    // Extract sub dofmap
    boost::scoped_ptr<ufc::dofmap> ufc_tmp_dofmap(ufc_dofmap.create_sub_dofmap(i));
    dolfin_assert(ufc_tmp_dofmap);

    // Check dimensional consistency between UFC dofmap and the mesh
    //check_dimensional_consistency(ufc_dofmap, mesh);

    // Get offset
    offset += ufc_tmp_dofmap->global_dimension(num_global_mesh_entities);
  }

  // Create UFC sub-system
  ufc::dofmap* sub_dofmap = ufc_dofmap.create_sub_dofmap(component[0]);
  dolfin_assert(sub_dofmap);

  // Return sub-system if sub-sub-system should not be extracted, otherwise
  // recursively extract the sub sub system
  if (component.size() == 1)
    return sub_dofmap;
  else
  {
    std::vector<std::size_t> sub_component;
    for (std::size_t i = 1; i < component.size(); ++i)
      sub_component.push_back(component[i]);

    ufc::dofmap* sub_sub_dofmap = extract_ufc_sub_dofmap(*sub_dofmap, offset,
                                                         sub_component,
                                                         mesh);
    delete sub_dofmap;
    return sub_sub_dofmap;
  }
}
//-----------------------------------------------------------------------------
