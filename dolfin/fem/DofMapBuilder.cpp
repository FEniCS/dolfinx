// Copyright (C) 2008 Anders Logg and Ola Skavhaug.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Niclas Jansson 2009.
// Modified by Garth N. Wells 2010.
//
// First added:  2008-08-12
// Last changed: 2010-04-05

#include <iostream>
#include <algorithm>
#include <cstring>
#include <ctime>
#include <dolfin/log/log.h>
#include <dolfin/mesh/BoundaryMesh.h>
#include <dolfin/mesh/Edge.h>
#include <dolfin/mesh/Facet.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshData.h>
#include <dolfin/mesh/MeshPartitioning.h>
#include "UFCCell.h"
#include "DofMap.h"
#include "DofMapBuilder.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void DofMapBuilder::parallel_build(DofMap& dofmap, const Mesh& mesh)
{
  // Create data structures
  set owned_dofs, shared_dofs, forbidden_dofs;

  // Determine ownership
  compute_ownership(owned_dofs, shared_dofs, forbidden_dofs, dofmap, mesh);

  // Renumber dofs
  parallel_renumber(owned_dofs, shared_dofs, forbidden_dofs, dofmap, mesh);
}
//-----------------------------------------------------------------------------
void DofMapBuilder::compute_ownership(set& owned_dofs, set& shared_dofs,
                                      set& forbidden_dofs,
                                      const DofMap& dofmap, const Mesh& mesh)
{
  info(TRACE, "Determining dof ownership for parallel dof map");

  // Initialize random number generator differently on each process
  //srand((uint)time(0) + MPI::process_number());
  // FIXME: Temporary while debugging (to get same results in each run)
  srand(253*MPI::process_number() + 378);

  // Extract the interior boundary
  BoundaryMesh interior_boundary;
  interior_boundary.init_interior_boundary(mesh);

  // Decide ownership of shared dofs
  std::vector<uint> send_buffer;
  std::map<uint, uint> dof_vote;
  std::vector<uint> old_cell_dofs(dofmap.max_local_dimension());
  std::vector<uint> facet_dofs(dofmap.num_facet_dofs());

  MeshFunction<uint>* cell_map = interior_boundary.data().mesh_function("cell map");
  if (cell_map)
  {
    for (CellIterator bc(interior_boundary); !bc.end(); ++bc)
    {
      // Get boundary facet
      Facet f(mesh, (*cell_map)[*bc]);

      // Get cell to which facet belongs (pick first)
      Cell c(mesh, f.entities(mesh.topology().dim())[0]);

      // Tabulate dofs on cell
      dofmap.tabulate_dofs(&old_cell_dofs[0], c);

      // Tabulate which dofs are on the facet
      dofmap.tabulate_facet_dofs(&facet_dofs[0], c.index(f));

      for (uint i = 0; i < dofmap.num_facet_dofs(); i++)
      {
        if (shared_dofs.find(old_cell_dofs[facet_dofs[i]]) == shared_dofs.end())
        {
          shared_dofs.insert(old_cell_dofs[facet_dofs[i]]);
          dof_vote[old_cell_dofs[facet_dofs[i]]] = (uint) rand();
          send_buffer.push_back(old_cell_dofs[facet_dofs[i]]);
          send_buffer.push_back(dof_vote[old_cell_dofs[facet_dofs[i]]]);
        }
      }
    }
  }

  // Decide ownership of "shared" dofs
  const uint num_proc = MPI::num_processes();
  const uint proc_num = MPI::process_number();
  const uint max_recv = MPI::global_maximum(send_buffer.size());
  std::vector<uint> recv_buffer(max_recv);
  for (uint k = 1; k < MPI::num_processes(); ++k)
  {
    uint src  = (proc_num - k + num_proc) % num_proc;
    uint dest = (proc_num +k) % num_proc;
    uint recv_count = MPI::send_recv(&send_buffer[0], send_buffer.size(), dest,
				                             &recv_buffer[0], max_recv, src);

    for (uint i = 0; i < recv_count; i += 2)
    {
      if (shared_dofs.find(recv_buffer[i]) != shared_dofs.end())
      {
        // Move dofs with higher ownership votes from shared to forbidden
        if (recv_buffer[i+1] < dof_vote[recv_buffer[i]])
        {
          forbidden_dofs.insert(recv_buffer[i]);
          shared_dofs.erase(recv_buffer[i]);
        }
        else if (recv_buffer[i+1] == dof_vote[recv_buffer[i]])
          error("Cannot decide on dof ownership. Votes are equal.");
      }
    }
  }

  // Mark all non-forbidden dofs as owned by the processes
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    dofmap.tabulate_dofs(&old_cell_dofs[0], *cell);
    const uint cell_dimension = dofmap.dimension(cell->index());
    for (uint i = 0; i < cell_dimension; ++i)
    {
      // Mark dof as owned if not forbidden
      if (forbidden_dofs.find(old_cell_dofs[i]) == forbidden_dofs.end())
        owned_dofs.insert(old_cell_dofs[i]);
    }
  }

  info(TRACE, "Finished determining dof ownership for parallel dof map");
}
//-----------------------------------------------------------------------------
void DofMapBuilder::parallel_renumber(const set& owned_dofs,
                             const set& shared_dofs,
                             const set& forbidden_dofs,
                             DofMap& dofmap, const Mesh& mesh)
{
  info(TRACE, "Renumber dofs for parallel dof map");

  // FIXME: Handle double-renumbered dof map
  if (dofmap.ufc_map_to_dofmap.size() > 0)
    error("DofMaps cannot yet be renumbered twice.");

  const std::vector<std::vector<uint> >& old_dofmap = dofmap.dofmap;
  std::vector<std::vector<uint> > new_dofmap(old_dofmap.size());
  assert(old_dofmap.size() == mesh.num_cells());

  // Compute offset for owned and non-shared dofs
  const uint process_offset = MPI::global_offset(owned_dofs.size(), true);

  // Map from old to new index for dofs
  std::map<uint, uint> old_to_new_dof_index;

  // Compute renumber for dofs
  uint counter = 0;
  std::vector<uint> send_buffer;
  for (set_iterator owned_dof = owned_dofs.begin(); owned_dof != owned_dofs.end(); ++owned_dof, counter++)
  {
    // New dof number
    old_to_new_dof_index[*owned_dof] = process_offset + counter;

    // UFC to renumbered map
    dofmap.ufc_map_to_dofmap[*owned_dof] = process_offset + counter;

    // If this dof is shared buffer old and new index for sending
    if (shared_dofs.find(*owned_dof) != shared_dofs.end())
    {
      send_buffer.push_back(*owned_dof);
      send_buffer.push_back(process_offset + counter);
    }
  }

  // FIXME: Use MPI::distribute here instead of send_recv

  // Exchange new dof numbers for shared dofs
  const uint num_proc = MPI::num_processes();
  const uint proc_num = MPI::process_number();
  const uint max_recv = MPI::global_maximum(send_buffer.size());
  std::vector<uint> recv_buffer(max_recv);
  for (uint k = 1; k < MPI::num_processes(); ++k)
  {
    const uint src  = (proc_num - k + num_proc) % num_proc;
    const uint dest = (proc_num +k) % num_proc;
    const uint recv_count = MPI::send_recv(&send_buffer[0], send_buffer.size(),
                                           dest,
                                           &recv_buffer[0], max_recv, src);

    // Add dofs renumbered by another process to the old-to-new map
    for (uint i = 0; i < recv_count; i += 2)
    {
      old_to_new_dof_index[recv_buffer[i]] = recv_buffer[i+1];

      // UFC to renumbered map
      dofmap.ufc_map_to_dofmap[recv_buffer[i]] = recv_buffer[i+1];
    }
  }

  // Build new dof map
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    const uint cell_dimension = dofmap.dimension(cell->index());
    new_dofmap[cell->index()].resize(cell_dimension);

    for (uint i = 0; i < cell_dimension; ++i)
    {
      const uint old_index = old_dofmap[cell->index()][i];
      new_dofmap[cell->index()][i] = old_to_new_dof_index[old_index];
    }
  }

  // Set new dof map
  dofmap.dofmap = new_dofmap;

  info(TRACE, "Finished renumbering dofs for parallel dof map");
}
//-----------------------------------------------------------------------------
