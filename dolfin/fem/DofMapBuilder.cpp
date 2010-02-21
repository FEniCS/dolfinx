// Copyright (C) 2008 Anders Logg and Ola Skavhaug.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Niclas Jansson 2009.
// Modified by Garth N. Wells 2010.
//
// First added:  2008-08-12
// Last changed: 2010-02-08

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
#include "UFC.h"
#include "DofMap.h"
#include "DofMapBuilder.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void DofMapBuilder::parallel_build(DofMap& dofmap, const Mesh& mesh)
{
  // Check that dof map has not been built
  if (dofmap._map.get())
    error("Local-to-global mapping has already been computed.");

  // Create data structures
  set owned_dofs, shared_dofs, forbidden_dofs;
  std::map<uint, std::vector<uint> > dof2index;

  // Determine ownership
  compute_ownership(owned_dofs, shared_dofs, forbidden_dofs, dof2index,
                    dofmap, mesh);

  // Renumber dofs
  parallel_renumber(owned_dofs, shared_dofs, forbidden_dofs, dof2index,
                    dofmap, mesh);
}
//-----------------------------------------------------------------------------
void DofMapBuilder::compute_ownership(set& owned_dofs, set& shared_dofs,
                                      set& forbidden_dofs,
                                      std::map<uint, std::vector<uint> >& dof2index,
                                      const DofMap& dofmap, const Mesh& mesh)
{
  info("Determining dof ownership for parallel dof map");

  // Initialize random number generator differently on each process
  srand((uint)time(0) + MPI::process_number());
  // FIXME: Temporary while debugging (to get same results in each run)
  //srand(MPI::process_number() + 1);

  // Extract the interior boundary
  BoundaryMesh interior_boundary;
  interior_boundary.init_interior_boundary(mesh);

  // Decide ownership of shared dofs
  UFCCell ufc_cell(mesh);
  std::vector<uint> send_buffer;
  std::map<uint, uint> dof_vote;
  std::vector<uint> old_dofs(dofmap.max_local_dimension());
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
      ufc_cell.update(c);
      dofmap.tabulate_dofs(&old_dofs[0], ufc_cell, c.index());

      // Tabulate which dofs are on the facet
      dofmap.tabulate_facet_dofs(&facet_dofs[0], c.index(f));

      for (uint i = 0; i < dofmap.num_facet_dofs(); i++)
      {
        if (shared_dofs.find(old_dofs[facet_dofs[i]]) == shared_dofs.end())
        {
          shared_dofs.insert(old_dofs[facet_dofs[i]]);
          dof_vote[old_dofs[facet_dofs[i]]] = (uint) rand();
          send_buffer.push_back(old_dofs[facet_dofs[i]]);
          send_buffer.push_back(dof_vote[old_dofs[facet_dofs[i]]]);
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
    uint src = (proc_num - k + num_proc) % num_proc;
    uint dest = (proc_num +k) % num_proc;

    uint recv_count = MPI::send_recv(&send_buffer[0], send_buffer.size(), dest,
				                             &recv_buffer[0], max_recv, src);

    for (uint i = 0; i < recv_count; i += 2)
    {
      if (shared_dofs.find(recv_buffer[i]) != shared_dofs.end())
      {
        // Move dofs with higher ownership votes from shared to forbidden
        if (recv_buffer[i+1] < dof_vote[recv_buffer[i]] )
        {
          forbidden_dofs.insert(recv_buffer[i]);
          shared_dofs.erase(recv_buffer[i]);
        }
      }
    }
  }

  // Mark all non forbidden dofs as owned by the processes
  for (CellIterator c(mesh); !c.end(); ++c)
  {
    ufc_cell.update(*c);
    dofmap.tabulate_dofs(&old_dofs[0], ufc_cell, c->index());
    const uint local_dimension = dofmap.local_dimension(ufc_cell);
    for (uint i = 0; i < local_dimension; i++)
    {
      // Mark dof as owned if not forbidden
      if (forbidden_dofs.find(old_dofs[i]) == forbidden_dofs.end())
        owned_dofs.insert(old_dofs[i]);

      // Create map from dof to dofmap offset
      dof2index[old_dofs[i]].push_back(c->index()*local_dimension + i);
    }
  }
  info("Finished determining dof ownership for parallel dof map");
}
//-----------------------------------------------------------------------------
void DofMapBuilder::parallel_renumber(const set& owned_dofs, const set& shared_dofs,
                             const set& forbidden_dofs,
                             const std::map<uint, std::vector<uint> >& dof2index,
                             DofMap& dofmap, const Mesh& mesh)
{
  info("Renumber dofs for parallel dof map");

  dofmap._ufc_to_map.clear();

  // Initialise and get dof map vector
  if (dofmap._map.get())
    dofmap._map->resize(dofmap.max_local_dimension()*mesh.num_cells());
  else
    dofmap._map.reset(new std::vector<uint>(dofmap.max_local_dimension()*mesh.num_cells()));
  std::vector<uint>& _dofmap = *dofmap._map;

  // Compute offset for owned and non-shared dofs
  uint offset = MPI::global_offset(owned_dofs.size(), true);

  // Compute renumbering for local and owned shared dofs
  std::vector<uint> send_buffer;
  for (set_iterator it = owned_dofs.begin(); it != owned_dofs.end(); ++it, offset++)
  {
    dofmap._ufc_to_map[*it] = offset;
    const std::vector<uint>& _dof2index = dof2index.find(*it)->second;
    for (vector_it di = _dof2index.begin(); di != _dof2index.end(); ++di)
      _dofmap[*di] = offset;

    if (shared_dofs.find(*it) != shared_dofs.end())
    {
      send_buffer.push_back(*it);
      send_buffer.push_back(offset);
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

    for (uint i = 0; i < recv_count; i += 2)
    {
      dofmap._ufc_to_map[recv_buffer[i]] = recv_buffer[i+1];

      // Assign new dof number for shared dofs
      if (forbidden_dofs.find(recv_buffer[i]) != forbidden_dofs.end())
      {
        const std::vector<uint>& _dof2index = dof2index.find(recv_buffer[i])->second;
        for (vector_it di = _dof2index.begin(); di != _dof2index.end(); ++di)
          _dofmap[*di] = recv_buffer[i+1];
      }
    }
  }
  info("Finished renumbering dofs for parallel dof map");
}
//-----------------------------------------------------------------------------
