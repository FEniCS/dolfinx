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
// Modified by Anders Logg 2011
//
// First added:  2011-09-17
// Last changed: 2011-11-14

#include "dolfin/common/MPI.h"
#include "dolfin/log/log.h"
#include "dolfin/mesh/Mesh.h"
#include "MeshFunction.h"

#include "MeshDistributed.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
std::map<std::size_t, std::set<std::pair<std::size_t, std::size_t> > >
MeshDistributed::off_process_indices(const std::vector<std::size_t>& entity_indices,
                                     std::size_t dim, const Mesh& mesh)
{
  if (dim == 0)
    warning("MeshDistributed::host_processes has not been tested for vertices.");

  // Mesh topology dim
  const std::size_t D = mesh.topology().dim();

  // Check that entity is a vertex or a cell
  if (dim != 0 && dim != D)
  {
    dolfin_error("MeshDistributed.cpp",
                 "compute off-process indices",
                 "This version of MeshDistributed::host_processes is only for vertices or cells");
  }

  // Check that global numbers have been computed.
  if (!mesh.topology().have_global_indices(dim))
  {
    dolfin_error("MeshDistributed.cpp",
                 "compute off-process indices",
                 "Global mesh entity numbers have not been computed");
  }

  // Check that global numbers have been computed.
  if (!mesh.topology().have_global_indices(D))
  {
    dolfin_error("MeshDistributed.cpp",
                 "compute off-process indices",
                 "Global mesh entity numbers have not been computed");
  }

  // Get global cell entity indices on this process
  const std::vector<std::size_t> global_entity_indices
      = mesh.topology().global_indices(dim);

  dolfin_assert(global_entity_indices.size() == mesh.num_cells());

  // Prepare map to hold process numbers
  std::map<std::size_t, std::set<std::pair<std::size_t, std::size_t> > > processes;

  // FIXME: work on optimising below code

  // List of indices to send
  std::vector<std::size_t> my_entities;

  // Remove local cells from my_entities to reduce communication
  if (dim == D)
  {
    // In order to fill vector my_entities...
    // build and populate a local set for non-local cells
    std::set<std::size_t> set_of_my_entities(entity_indices.begin(), entity_indices.end());

    // FIXME: This can be made more efficient by exploiting fact that
    //        set is sorted
    // Remove local cells from set_of_my_entities to reduce communication
    for (std::size_t j = 0; j < global_entity_indices.size(); ++j)
      set_of_my_entities.erase(global_entity_indices[j]);

    // Copy entries from set_of_my_entities to my_entities
    my_entities = std::vector<std::size_t>(set_of_my_entities.begin(), set_of_my_entities.end());
  }
  else
    my_entities = entity_indices;

  // FIXME: handle case when my_entities.empty()
  //dolfin_assert(!my_entities.empty());

  // Prepare data structures for send/receive
  const std::size_t num_proc = MPI::num_processes();
  const std::size_t proc_num = MPI::process_number();
  const std::size_t max_recv = MPI::max(my_entities.size());
  std::vector<std::size_t> off_process_entities(max_recv);

  // Send and receive data
  for (std::size_t k = 1; k < MPI::num_processes(); ++k)
  {
    const std::size_t src  = (proc_num - k + num_proc) % num_proc;
    const std::size_t dest = (proc_num + k) % num_proc;

    MPI::send_recv(my_entities, dest, off_process_entities, src);

    const std::size_t recv_entity_count = off_process_entities.size();

    // Check if this process owns received entities, and if so
    // store local index
    std::vector<std::size_t> my_hosted_entities;
    {
      // Build a temporary map hosting global_entity_indices
      std::map<std::size_t, std::size_t> map_of_global_entity_indices;
      for (std::size_t j = 0; j < global_entity_indices.size(); j++)
        map_of_global_entity_indices[global_entity_indices[j]] = j;

      for (std::size_t j = 0; j < recv_entity_count; j++)
      {
        // Check if this process hosts 'received_entity'
        const std::size_t received_entity = off_process_entities[j];
        std::map<std::size_t, std::size_t>::const_iterator it;
        it = map_of_global_entity_indices.find(received_entity);
        if (it != map_of_global_entity_indices.end())
        {
          const std::size_t local_index = it->second;
          my_hosted_entities.push_back(received_entity);
          my_hosted_entities.push_back(local_index);
        }
      }
    }

    // Send/receive hosted cells
    const std::size_t max_recv_host_proc = MPI::max(my_hosted_entities.size());
    std::vector<std::size_t> host_processes(max_recv_host_proc);
    MPI::send_recv(my_hosted_entities, src, host_processes, dest);

    const std::size_t recv_hostproc_count = host_processes.size();
    for (std::size_t j = 0; j < recv_hostproc_count; j += 2)
    {
      const std::size_t global_index = host_processes[j];
      const std::size_t local_index  = host_processes[j + 1];
      processes[global_index].insert(std::make_pair(dest, local_index));
    }

    // FIXME: Do later for efficiency
    // Remove entries from entities (from my_entities) that cannot
    // reside on more processes (i.e., cells)
  }

  // Sanity check
  const std::set<std::size_t> test_set(my_entities.begin(), my_entities.end());
  const std::size_t number_expected = test_set.size();
  if (number_expected != processes.size())
  {
    dolfin_error("MeshDistributed.cpp",
                 "compute off-process indices",
                 "Sanity check failed");
  }

  return processes;
}
//-----------------------------------------------------------------------------
