// Copyright (C) 2011 Garth N. Wells
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
// First added:  2011-09-17
// Last changed:

#include "dolfin/common/MPI.h"
#include "dolfin/log/log.h"
#include "dolfin/mesh/Mesh.h"
#include "MeshFunction.h"
#include "ParallelData.h"

#include "MeshDistributed.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
std::map<dolfin::uint, std::set<std::pair<dolfin::uint, dolfin::uint> > >
MeshDistributed::off_process_indices(const std::vector<uint>& entity_indices,
                                     uint dim, const Mesh& mesh)
{
  warning("MeshDistributed::host_processes has has limited testing.");

  const uint D = mesh.topology().dim();

  // Check that entity is a vertex or a cell
  if (dim != 0 && dim != D)
    error("This version of MeshDistributed::host_processes is only for vertices or cells.");

  // Check that global numbers have been computed.
  if (!mesh.parallel_data().have_global_entity_indices(dim))
    error("Global mesh entity numbers have not been computed.");

  // Check that global numbers have been computed.
  if (!mesh.parallel_data().have_global_entity_indices(D))
    error("Global mesh entity numbers have not been computed.");

  // Get global cell entity indices on this process
  const MeshFunction<uint>& _global_entity_indices = mesh.parallel_data().global_entity_indices(D);
  const std::vector<uint> global_entity_indices(_global_entity_indices.values(),
                _global_entity_indices.values() + _global_entity_indices.size());

  assert(global_entity_indices.size() == mesh.num_cells());

  // Prepare map to hold process numbers
  std::map<uint, std::set<std::pair<uint, uint> > > processes;

  // List of indices to send
  std::vector<uint> my_entities = entity_indices;

  assert(my_entities.size() > 0);

  // Remove local cells from my_entities to reduce communication
  /*
  if (dim == D)
  {
    std::vector<uint>::iterator it;
    for (uint i = 0; i < global_entity_indices.size(); ++i)
    {
      const uint global_index = global_entity_indices[i];
      it = std::find(my_entities.begin(), my_entities.end(), global_index);
      if (it != my_entities.end())
        my_entities.erase(it);
    }
  }
  */

  // Prepare data structures for send/receive
  const uint num_proc = MPI::num_processes();
  const uint proc_num = MPI::process_number();
  const uint max_recv = MPI::max(my_entities.size());
  std::vector<uint> off_process_entities(max_recv);

  // Send and receive data
  for (uint k = 1; k < MPI::num_processes(); ++k)
  {
    const uint src  = (proc_num - k + num_proc) % num_proc;
    const uint dest = (proc_num + k) % num_proc;
    cout << "Src, dest: " << src << ", " << dest << endl;


    // Send/receive list of entities (global indices)
    const uint recv_enitity_count = MPI::send_recv(&my_entities[0],
                                        my_entities.size(), dest,
                                        &off_process_entities[0],
                                        off_process_entities.size(), src);

    // Check if this process 'hosts' received entities, and if so
    // store process number and local index
    std::vector<uint> my_hosted_entities;
    for (uint j = 0; j < recv_enitity_count; ++j)
    {
      const uint received_entity = off_process_entities[j];

      // Check if this process hosts 'received_entity'
      std::vector<uint>::const_iterator it;
      it = std::find(global_entity_indices.begin(), global_entity_indices.end(), received_entity);
      if (it != global_entity_indices.end())
      {
        const uint local_index  = std::distance(global_entity_indices.begin(), it);
        assert(global_entity_indices[local_index] == *it);
        my_hosted_entities.push_back(*it);
        my_hosted_entities.push_back(local_index);
      }
    }

    // Send/receive back 'hosting' processes
    const uint max_recv_host_proc = MPI::max(my_hosted_entities.size());
    std::vector<uint> host_processes(max_recv_host_proc);
    const uint recv_hostproc_count = MPI::send_recv(&my_hosted_entities[0],
                                            my_hosted_entities.size(), dest,
                                            &host_processes[0],
                                            host_processes.size(), src);

    //cout << "Size of received . . . " << recv_hostproc_count << endl;
    //cout << "Sum:                   " << MPI::sum(recv_hostproc_count) << endl;
    for (uint j = 0; j < recv_hostproc_count; j += 2)
      processes[ host_processes[j] ].insert(std::make_pair(src, host_processes[j + 1]));

    // FIXME: Do later for efficiency
    // Remove entries from entities (my_entities) to be sent that cannot
    // reside on more processes (i.e., cells)
  }

  return processes;
}
//-----------------------------------------------------------------------------
/*
std::map<dolfin::uint, std::set<std::pair<dolfin::uint, dolfin::uint> > >
MeshDistributed::off_process_indices(const std::vector<std::pair<uint, uint> >& entity_indices,
                                     uint dim, const Mesh& mesh)
{
  // Function is for facets and edges
  std::vector<uint> cell_indices(entity_indices.size());
  for (uint i = 0; i < entity_indices.size(); ++i)
    cell_indices[i] = entity_indices[i].first;

  const uint d = mesh.topology().dim();
  return host_processes(cell_indices, d, mesh);
}
*/
//-----------------------------------------------------------------------------
