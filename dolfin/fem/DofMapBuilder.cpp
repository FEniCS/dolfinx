// Copyright (C) 2008 Anders Logg and Ola Skavhaug.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Niclas Jansson 2009.
//
// First added:  2008-08-12
// Last changed: 2009-08-06

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

typedef std::vector<dolfin::uint>::const_iterator vector_iterator;

//-----------------------------------------------------------------------------
void DofMapBuilder::build(DofMap& dof_map, const Mesh& mesh)
{
  info("Building parallel dof map");

  // Check that dof map has not been built
  if (dof_map.dof_map)
    error("Local-to-global mapping has already been computed.");
  
  const uint n = dof_map.max_local_dimension();
  
  // Allocate scratch _dof_map
  int* _dof_map = new int[n*mesh.num_cells()];   

  // Extract the interior boundary
  BoundaryMesh interior_boundary;
  interior_boundary.init_interior(mesh);
  MeshFunction<uint>* cell_map = interior_boundary.data().mesh_function("cell map");
  
  set shared_dofs, forbidden_dofs, owned_dofs;
  std::vector<uint> send_buffer;
  std::map<uint, uint> dof_vote;
  std::map<uint, std::vector<uint> > dof2index;
  
  UFCCell ufc_cell(mesh);
  uint *dofs = new uint[n];

  // Initialize random number generator differently on each process
  srand((uint)time(0) + MPI::process_number());
  
  // Decide ownership of shared dofs
  for (CellIterator bc(interior_boundary); !bc.end(); ++bc) 
  {
    Facet f(mesh, cell_map->get(*bc));
    for (CellIterator c(f); !c.end(); ++c)
    {      
      ufc_cell.update(*c);
      dof_map.tabulate_dofs(dofs, ufc_cell, c->index());        
      for (uint i = 0; i < n; i++)
      {
        // Assign an ownership vote for each "shared" dof
        if (shared_dofs.find(dofs[i]) == shared_dofs.end()) 
        {
          shared_dofs.insert(dofs[i]);
          dof_vote[dofs[i]] = (uint) rand();     
          send_buffer.push_back(dofs[i]);
          send_buffer.push_back(dof_vote[dofs[i]]);
        }
      }
    }
  }

  // Decide ownership of "shared" dofs
  uint src, dest, recv_count;
  const uint num_proc = MPI::num_processes();
  const uint proc_num = MPI::process_number();
  uint max_recv = MPI::global_maximum(send_buffer.size());
  uint *recv_buffer = new uint[max_recv];
  for(uint k = 1; k < MPI::num_processes(); ++k)
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
    dof_map.tabulate_dofs(dofs, ufc_cell, c->index());  
    for (uint i = 0; i < n; i++)
    {
      if (forbidden_dofs.find(dofs[i]) == forbidden_dofs.end())
      {
        // Mark dof as owned
        owned_dofs.insert(dofs[i]);
      }
      
      // Create mapping from dof to dof_map offset
      dof2index[dofs[i]].push_back(c->index() * n + i);
    }    
  }
  
  // Compute offset for owned and non shared dofs
  const uint range = owned_dofs.size();
  uint offset = MPI::global_offset(range, true);   

  // Compute renumbering for local and owned shared dofs
  for (set_iterator it = owned_dofs.begin(); it != owned_dofs.end(); ++it, offset++)
  {
    for(vector_iterator di = dof2index[*it].begin(); di != dof2index[*it].end(); ++di)
    _dof_map[*di] = offset;
    
    if (shared_dofs.find(*it) != shared_dofs.end())
    {
      send_buffer.push_back(*it);
      send_buffer.push_back(offset);
    }
  }

  // Exchange new dof numbers for shared dofs
  delete[] recv_buffer;
  max_recv = MPI::global_maximum(send_buffer.size());
  recv_buffer = new uint[max_recv];
  for(uint k = 1; k < MPI::num_processes(); ++k)
  {
    src = (proc_num - k + num_proc) % num_proc;
    dest = (proc_num +k) % num_proc;
    
    recv_count = MPI::send_recv(&send_buffer[0], send_buffer.size(), dest,
				recv_buffer, max_recv, src);

    for (uint i = 0; i < recv_count; i += 2)
    {
      // Assign new dof number for shared dofs
      if (forbidden_dofs.find(recv_buffer[i]) != forbidden_dofs.end())
      {
        for(vector_iterator di = dof2index[recv_buffer[i]].begin();
                  di != dof2index[recv_buffer[i]].end(); ++di)
          _dof_map[*di] = recv_buffer[i+1];
      }
    }
  }
  delete[] recv_buffer;
  delete[] dofs;

  
  // Allocate dof map
  delete [] dof_map.dof_map;
  dof_map.dof_map_size = dof_map.global_dimension();
  dof_map.dof_map = new int[n*mesh.num_cells()];    
  memcpy(dof_map.dof_map, _dof_map, n*mesh.num_cells() * sizeof(int));  

  delete [] _dof_map;

  info("Finished building parallel dof map");
}
//-----------------------------------------------------------------------------
