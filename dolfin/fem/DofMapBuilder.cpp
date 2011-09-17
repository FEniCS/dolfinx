// Copyright (C) 2008 Anders Logg and Ola Skavhaug
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
//
// First added:  2008-08-12
// Last changed: 2011-03-17

#include <ufc.h>
#include <boost/random.hpp>
#include <boost/unordered_map.hpp>

#include <dolfin/common/Timer.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/BoundaryMesh.h>
#include <dolfin/mesh/Facet.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Vertex.h>
#include "DofMap.h"
#include "UFCCell.h"
#include "UFCMesh.h"
#include "DofMapBuilder.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void DofMapBuilder::build(DofMap& dofmap, const Mesh& dolfin_mesh,
                          const UFCMesh& ufc_mesh, bool distributed)
{
  // Start timer for dofmap initialization
  Timer t0("Init dofmap");

  // Create space for dof map
  dofmap.dofmap.resize(dolfin_mesh.num_cells());

  dofmap._off_process_owner.clear();

  assert(dofmap._ufc_dofmap);

  // Build dofmap from ufc::dofmap
  dolfin::UFCCell ufc_cell(dolfin_mesh);
  for (dolfin::CellIterator cell(dolfin_mesh); !cell.end(); ++cell)
  {
    // Update UFC cell
    ufc_cell.update(*cell);

    // Get standard local dimension
    const unsigned int local_dim = dofmap._ufc_dofmap->local_dimension(ufc_cell);
    dofmap.dofmap[cell->index()].resize(local_dim);

    // Tabulate standard UFC dof map
    dofmap._ufc_dofmap->tabulate_dofs(&dofmap.dofmap[cell->index()][0], ufc_mesh, ufc_cell);
  }

  // Build (renumber) dofmap when running in parallel
  if (distributed)
  {
    // Build set of global dofs
    const set global_dofs = compute_global_dofs(dofmap, dolfin_mesh);

    // Build distributed dof map
    build_distributed(dofmap, global_dofs, dolfin_mesh);
  }
  else
    dofmap._ownership_range = std::make_pair(0, dofmap.global_dimension());
}
//-----------------------------------------------------------------------------
void DofMapBuilder::build_distributed(DofMap& dofmap,
                                      const DofMapBuilder::set& global_dofs,
                                      const Mesh& mesh)
{
  // Create data structures
  DofMapBuilder::set owned_dofs, shared_owned_dofs, shared_unowned_dofs;

  // Computed owned and shared dofs (and owned and un-owned)
  compute_ownership(owned_dofs, shared_owned_dofs, shared_unowned_dofs,
                    dofmap, global_dofs, mesh);

  // Renumber dofs owned dofs and received new numbering for unowned shared dofs
  parallel_renumber(owned_dofs, shared_owned_dofs, shared_unowned_dofs,
                    dofmap, mesh);
}
//-----------------------------------------------------------------------------
void DofMapBuilder::compute_ownership(set& owned_dofs, set& shared_owned_dofs,
                                      set& shared_unowned_dofs,
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
  boost::unordered_map<uint, uint> dof_vote;
  std::vector<uint> facet_dofs(dofmap.num_facet_dofs());

  // Communication buffer
  std::vector<uint> send_buffer;

  // Extract the interior boundary
  BoundaryMesh interior_boundary;
  interior_boundary.init_interior_boundary(mesh);

  // Build set of dofs on process boundary (assume all are owned by this process)
  const MeshFunction<unsigned int>& cell_map = interior_boundary.cell_map();
  if (cell_map.size() > 0)
  {
    for (CellIterator bc(interior_boundary); !bc.end(); ++bc)
    {
      // Get boundary facet
      Facet f(mesh, cell_map[*bc]);

      // Get cell to which facet belongs (pick first)
      Cell c(mesh, f.entities(mesh.topology().dim())[0]);

      // Tabulate dofs on cell
      const std::vector<uint>& cell_dofs = dofmap.cell_dofs(c.index());

      // Tabulate which dofs are on the facet
      dofmap.tabulate_facet_dofs(&facet_dofs[0], c.index(f));

      // Insert shared dofs into set and assign a 'vote'
      for (uint i = 0; i < dofmap.num_facet_dofs(); i++)
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
  const uint max_recv = MPI::max(send_buffer.size());
  std::vector<uint> recv_buffer(max_recv);
  for (uint k = 1; k < MPI::num_processes(); ++k)
  {
    uint src  = (proc_num - k + num_proc) % num_proc;
    uint dest = (proc_num +k) % num_proc;
    uint recv_count = MPI::send_recv(&send_buffer[0], send_buffer.size(), dest,
				                             &recv_buffer[0], max_recv, src);

    for (uint i = 0; i < recv_count; i += 2)
    {
      const uint received_dof  = recv_buffer[i];
      const uint received_vote = recv_buffer[i + 1];
      if (shared_owned_dofs.find(received_dof) != shared_owned_dofs.end())
      {
        // Move dofs with higher ownership votes from shared to shared but not owned
        if (received_vote < dof_vote[received_dof])
        {
          shared_unowned_dofs.insert(received_dof);
          shared_owned_dofs.erase(received_dof);
        }
        else if (received_vote == dof_vote[received_dof])
        {
          // FIXME: Eventually replace this with a more robust condition. It's
          // good for testing that ownership of shared dofs is spread roughly
          // equally
          error("Cannot decide on dof ownership. Votes are equal.");
        }
      }
    }
  }

  // Add/remove global dofs to relevant sets (process 0 owns local dofs)
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
    const std::vector<uint>& cell_dofs = dofmap.cell_dofs(cell->index());
    const uint cell_dimension = dofmap.cell_dimension(cell->index());
    for (uint i = 0; i < cell_dimension; ++i)
    {
      // Mark dof as owned if in unowned set
      if (shared_unowned_dofs.find(cell_dofs[i]) == shared_unowned_dofs.end())
        owned_dofs.insert(cell_dofs[i]);
    }
  }

  // Check that sum of locally owned dofs is equal to global dimension
  const uint _owned_dim = owned_dofs.size();
  assert(MPI::sum(_owned_dim) == dofmap.global_dimension());

  log(TRACE, "Finished determining dof ownership for parallel dof map");
}
//-----------------------------------------------------------------------------
void DofMapBuilder::parallel_renumber(const set& owned_dofs,
                             const set& shared_owned_dofs,
                             const set& shared_unowned_dofs,
                             DofMap& dofmap, const Mesh& mesh)
{
  log(TRACE, "Renumber dofs for parallel dof map");

  // FIXME: Handle double-renumbered dof map
  if (dofmap.ufc_map_to_dofmap.size() > 0)
    error("DofMaps cannot yet be renumbered twice.");

  const std::vector<std::vector<uint> >& old_dofmap = dofmap.dofmap;
  std::vector<std::vector<uint> > new_dofmap(old_dofmap.size());
  assert(old_dofmap.size() == mesh.num_cells());

  // Compute offset for owned and non-shared dofs
  const uint process_offset = MPI::global_offset(owned_dofs.size(), true);

  // Clear some data
  dofmap._off_process_owner.clear();

  // Map from old to new index for dofs
  boost::unordered_map<uint, uint> old_to_new_dof_index;

  // Renumber owned dofs and buffer dofs that are owned but shared with another
  // process
  uint counter = 0;
  std::vector<uint> send_buffer;
  for (set_iterator owned_dof = owned_dofs.begin(); owned_dof != owned_dofs.end(); ++owned_dof, counter++)
  {
    // Set new dof number
    old_to_new_dof_index[*owned_dof] = process_offset + counter;

    // Update UFC-to-renumbered map for new number
    dofmap.ufc_map_to_dofmap[*owned_dof] = process_offset + counter;

    // If this dof is shared and owned, buffer old and new index for sending
    if (shared_owned_dofs.find(*owned_dof) != shared_owned_dofs.end())
    {
      send_buffer.push_back(*owned_dof);
      send_buffer.push_back(process_offset + counter);
    }
  }

  // FIXME: Use MPI::distribute here instead of send_recv

  // Exchange new dof numbers for dofs that are shared
  const uint num_proc = MPI::num_processes();
  const uint proc_num = MPI::process_number();
  const uint max_recv = MPI::max(send_buffer.size());
  std::vector<uint> recv_buffer(max_recv);
  for (uint k = 1; k < MPI::num_processes(); ++k)
  {
    const uint src  = (proc_num - k + num_proc) % num_proc;
    const uint dest = (proc_num + k) % num_proc;
    const uint recv_count = MPI::send_recv(&send_buffer[0], send_buffer.size(),
                                           dest,
                                           &recv_buffer[0], max_recv, src);

    // Add dofs renumbered by another process to the old-to-new map
    for (uint i = 0; i < recv_count; i += 2)
    {
      const uint received_old_dof_index = recv_buffer[i];
      const uint received_new_dof_index = recv_buffer[i + 1];

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

  // Build new dof map
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    const uint cell_index = cell->index();
    const uint cell_dimension = dofmap.cell_dimension(cell_index);

    // Resize cell map and insert dofs
    new_dofmap[cell_index].resize(cell_dimension);
    for (uint i = 0; i < cell_dimension; ++i)
    {
      const uint old_index = old_dofmap[cell_index][i];
      new_dofmap[cell_index][i] = old_to_new_dof_index[old_index];
    }
  }

  // Set new dof map
  dofmap.dofmap = new_dofmap;

  // Set ownership range
  dofmap._ownership_range = std::make_pair<uint, uint>(process_offset, process_offset + owned_dofs.size());

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
  uint offset = 0;
  set global_dof_indices;
  compute_global_dofs(global_dof_indices, offset, _dofmap, dolfin_mesh, ufc_mesh);

  return global_dof_indices;
}
//-----------------------------------------------------------------------------
void DofMapBuilder::compute_global_dofs(DofMapBuilder::set& global_dofs,
                            uint& offset,
                            boost::shared_ptr<const ufc::dofmap> dofmap,
                            const Mesh& dolfin_mesh, const UFCMesh& ufc_mesh)
{
  assert(dofmap);
  const uint D = dolfin_mesh.topology().dim();

  if (dofmap->num_sub_dofmaps() == 0)
  {
    // Check if dofmap is for global dofs
    bool global_dof = true;
    for (uint d = 0; d <= D; ++d)
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
        error("Global dof has dimension != 1.");

      boost::scoped_ptr<ufc::mesh> ufc_mesh(new ufc::mesh);
      boost::scoped_ptr<ufc::cell> ufc_cell(new ufc::cell);
      uint dof = 0;
      dofmap->tabulate_dofs(&dof, *ufc_mesh, *ufc_cell);

      // Insert global dof index
      std::pair<DofMapBuilder::set::iterator, bool> ret = global_dofs.insert(dof + offset);
      if (!ret.second)
        error("Problem inserting global dof into set. Dof index aready exists.");
    }
  }
  else
  {
    for (uint i = 0; i < dofmap->num_sub_dofmaps(); ++i)
    {
      // Extract sub-dofmap and intialise
      boost::shared_ptr<ufc::dofmap> sub_dofmap(dofmap->create_sub_dofmap(i));
      DofMap::init_ufc_dofmap(*sub_dofmap, ufc_mesh, dolfin_mesh);

      compute_global_dofs(global_dofs, offset, sub_dofmap, dolfin_mesh,
                          ufc_mesh);

      // Get offset
      offset += sub_dofmap->global_dimension();
    }
  }
}
//-----------------------------------------------------------------------------
