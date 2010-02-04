// Copyright (C) 2008 Anders Logg and Ola Skavhaug.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Niclas Jansson 2009.
//
// First added:  2008-08-12
// Last changed: 2009-11-04

#include <iostream>

#include <algorithm>
#include <cstring>
#include <ctime>
#include <set>
#include <tr1/unordered_set>

#include <dolfin/common/Set.h>
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

// FIXME: Test which 'set' is most efficient

//typedef std::set<dolfin::uint> set;
//typedef std::set<dolfin::uint>::const_iterator set_iterator;

//typedef Set<dolfin::uint> set;
//typedef Set<dolfin::uint>::const_iterator set_iterator;

typedef std::tr1::unordered_set<dolfin::uint> set;
typedef std::tr1::unordered_set<dolfin::uint>::const_iterator set_iterator;

typedef std::vector<dolfin::uint>::const_iterator vector_it;

//-----------------------------------------------------------------------------
void DofMapBuilder::parallel_build(DofMap& dofmap, const Mesh& mesh)
{
  // FIXME: Split this function into two; deciding ownership and then renumbering

  info("Building parallel dof map");

  // Check that dof map has not been built
  if (dofmap._map.get())
    error("Local-to-global mapping has already been computed.");

  dofmap._ufc_to_map.clear();

  const uint max_local_dimension = dofmap.max_local_dimension();

  // Allocate scratch _dofmap
  int* _dofmap = new int[max_local_dimension*mesh.num_cells()];

  // Extract the interior boundary
  BoundaryMesh interior_boundary;
  interior_boundary.init_interior(mesh);
  MeshFunction<uint>* cell_map = interior_boundary.data().mesh_function("cell map");

  set shared_dofs, forbidden_dofs, owned_dofs;
  std::vector<uint> send_buffer;
  std::map<uint, uint> dof_vote;
  std::map<uint, std::vector<uint> > dof2index;

  // Initialize random number generator differently on each process
  srand((uint)time(0) + MPI::process_number());

  // FIXME: Temporary while debugging (to get same results in each run)
  //srand(MPI::process_number());

  UFCCell ufc_cell(mesh);
  uint *old_dofs = new uint[max_local_dimension];
  uint *facet_dofs = new uint[dofmap.num_facet_dofs()];

  // Decide ownership of shared dofs
  for (CellIterator bc(interior_boundary); !bc.end(); ++bc)
  {
    // Get boundary facet
    Facet f(mesh, (*cell_map)[*bc]);

    // Get cell to which facet belongs (pick first)
    Cell c(mesh, f.entities(mesh.topology().dim())[0]);

    // Get local index of facet with respect to the cell
    const uint local_facet = c.index(f);

    ufc_cell.update(c);

    // Tabulate dofs on cell
    dofmap.tabulate_dofs(old_dofs, ufc_cell, c.index());

    // Tabulate which dofs are on the facet
    dofmap.tabulate_facet_dofs(facet_dofs, local_facet);

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

  // Decide ownership of "shared" dofs
  uint src, dest, recv_count;
  const uint num_proc = MPI::num_processes();
  const uint proc_num = MPI::process_number();
  uint max_recv = MPI::global_maximum(send_buffer.size());
  uint *recv_buffer = new uint[max_recv];
  for (uint k = 1; k < MPI::num_processes(); ++k)
  {
    src = (proc_num - k + num_proc) % num_proc;
    dest = (proc_num +k) % num_proc;

    recv_count = MPI::send_recv(&send_buffer[0], send_buffer.size(), dest,
				                        recv_buffer, max_recv, src);

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

  send_buffer.clear();

  // Mark all non forbidden dofs as owned by the processes
  for (CellIterator c(mesh); !c.end(); ++c)
  {
    ufc_cell.update(*c);
    dofmap.tabulate_dofs(old_dofs, ufc_cell, c->index());
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
  delete[] facet_dofs;
  delete[] old_dofs;

  // Compute offset for owned and non-shared dofs
  const uint range = owned_dofs.size();
  uint offset = MPI::global_offset(range, true);

  // Compute renumbering for local and owned shared dofs
  for (set_iterator it = owned_dofs.begin(); it != owned_dofs.end(); ++it, offset++)
  {
    dofmap._ufc_to_map[*it] = offset;
    for (vector_it di = dof2index[*it].begin(); di != dof2index[*it].end(); ++di)
      _dofmap[*di] = offset;

    if (shared_dofs.find(*it) != shared_dofs.end())
    {
      send_buffer.push_back(*it);
      send_buffer.push_back(offset);
    }
  }

  // FIXME: Use MPI::distribute here instead of send_recv

  // Exchange new dof numbers for shared dofs
  delete [] recv_buffer;
  max_recv = MPI::global_maximum(send_buffer.size());
  recv_buffer = new uint[max_recv];
  for (uint k = 1; k < MPI::num_processes(); ++k)
  {
    src  = (proc_num - k + num_proc) % num_proc;
    dest = (proc_num +k) % num_proc;

    recv_count = MPI::send_recv(&send_buffer[0], send_buffer.size(), dest,
                                recv_buffer, max_recv, src);

    for (uint i = 0; i < recv_count; i += 2)
    {
      dofmap._ufc_to_map[recv_buffer[i]] = recv_buffer[i+1];

      // Assign new dof number for shared dofs
      if (forbidden_dofs.find(recv_buffer[i]) != forbidden_dofs.end())
      {
        for (vector_it di = dof2index[recv_buffer[i]].begin();
                       di != dof2index[recv_buffer[i]].end(); ++di)
        {
          _dofmap[*di] = recv_buffer[i+1];
        }
      }
    }
  }
  delete [] recv_buffer;

  // Copy dof map
  if (dofmap._map.get())
    dofmap._map->resize(max_local_dimension*mesh.num_cells());
  else
    dofmap._map.reset(new std::vector<uint>(max_local_dimension*mesh.num_cells()));

  // FIXME: Can this step be avoided?
  std::copy(_dofmap, _dofmap + max_local_dimension*mesh.num_cells(), dofmap._map->begin());

  delete [] _dofmap;

  info("Finished building parallel dof map");
}
//-----------------------------------------------------------------------------
