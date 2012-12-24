// Copyright (C) 2008-2012 Niclas Jansson, Ola Skavhaug, Anders Logg
// and Garth N. Wells
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
// Modified by Kent-Andre Mardal 2011
// Modified by Anders Logg 2011
// Modified by Garth N. Wells 2011-2012
//
// First added:  2008-12-01
// Last changed: 2012-12-24

#include <algorithm>
#include <iterator>
#include <map>
#include <numeric>
#include <set>
#include <boost/multi_array.hpp>

#include <dolfin/log/log.h>
#include <dolfin/common/MPI.h>
#include <dolfin/common/Timer.h>
#include <dolfin/graph/ParMETIS.h>
#include <dolfin/graph/SCOTCH.h>
#include <dolfin/parameter/GlobalParameters.h>
#include "BoundaryMesh.h"
#include "Facet.h"
#include "LocalMeshData.h"
#include "Mesh.h"
#include "MeshDistributed.h"
#include "MeshEditor.h"
#include "MeshEntityIterator.h"
#include "MeshFunction.h"
#include "MeshTopology.h"
#include "MeshValueCollection.h"
#include "Point.h"
#include "Vertex.h"
#include "MeshPartitioning.h"

using namespace dolfin;

// Explicitly instantiate some templated functions to help the Python
// wrappers
template void MeshPartitioning::build_mesh_value_collection(const Mesh& mesh,
   const std::vector<std::pair<std::pair<std::size_t, std::size_t>, std::size_t> >& local_value_data,
   MeshValueCollection<std::size_t>& mesh_values);
template void MeshPartitioning::build_mesh_value_collection(const Mesh& mesh,
   const std::vector<std::pair<std::pair<std::size_t, std::size_t>, int> >& local_value_data,
   MeshValueCollection<int>& mesh_values);
template void MeshPartitioning::build_mesh_value_collection(const Mesh& mesh,
   const std::vector<std::pair<std::pair<std::size_t, std::size_t>, double> >& local_value_data,
   MeshValueCollection<double>& mesh_values);
template void MeshPartitioning::build_mesh_value_collection(const Mesh& mesh,
   const std::vector<std::pair<std::pair<std::size_t, std::size_t>, bool> >& local_value_data,
   MeshValueCollection<bool>& mesh_values);

//-----------------------------------------------------------------------------
void MeshPartitioning::build_distributed_mesh(Mesh& mesh)
{
  if (MPI::num_processes() > 1)
  {
    // Create and distribute local mesh data
    LocalMeshData local_mesh_data(mesh);

    // Build distributed mesh
    build_distributed_mesh(mesh, local_mesh_data);
  }
}
//-----------------------------------------------------------------------------
void MeshPartitioning::build_distributed_mesh(Mesh& mesh,
                                              const LocalMeshData& local_data)
{
  // Partition mesh
  partition(mesh, local_data);

  // Create MeshDomains from local_data
  build_mesh_domains(mesh, local_data);

  // Initialise number of globally connected cells to each facet. This is
  // necessary to distinguish between facets on a exterior boundary and
  // facets on a partition boudnary (see
  // https://bugs.launchpad.net/dolfin/+bug/733834).
  MeshDistributed::init_facet_cell_connections(mesh);
}
//-----------------------------------------------------------------------------
/*
std::map<std::size_t, std::vector<std::pair<std::size_t, std::size_t> > >
  MeshPartitioning::compute_shared_entities(const Mesh& mesh, std::size_t d)
{
  // Compute ownership of entities ([entity vertices], data):
  //  [0]: owned exclusively (will be numbered by this process)
  //  [1]: owned and shared (will be numbered by this process, and number
  //       commuicated to other processes)
  //  [2]: not owned but shared (will be numbered by another process, and number
  //       commuicated to this processes)
  const boost::array<std::map<Entity, EntityData>, 3> entity_ownership
    = compute_entity_ownership(mesh, d);
  const std::map<Entity, EntityData>& shared_entities0 = entity_ownership[1];
  const std::map<Entity, EntityData>& shared_entities1 = entity_ownership[2];

  // Initialize entities of dimension d
  mesh.init(d);

  // Send my local index to sharing processes, and receive local index
  // from sharing processes
  std::map<Entity, EntityData>::const_iterator e;
  std::map<std::size_t, std::vector<std::size_t> > send_local_indices;
  for (e = shared_entities0.begin(); e != shared_entities0.end(); ++e)
  {
    std::vector<std::size_t>::const_iterator dest;
    for (dest = e->second.processes.begin(); dest != e->second.processes.end(); ++dest)
    {
      send_local_indices[*dest].push_back(e->second.local_index);
      send_local_indices[*dest].insert(send_local_indices[*dest].end(), e->first.begin(), e->first.end());
    }
  }
  for (e = shared_entities1.begin(); e != shared_entities1.end(); ++e)
  {
    std::vector<std::size_t>::const_iterator dest;
    for (dest = e->second.processes.begin(); dest != e->second.processes.end(); ++dest)
    {
      send_local_indices[*dest].push_back(e->second.local_index);
      send_local_indices[*dest].insert(send_local_indices[*dest].end(), e->first.begin(), e->first.end());
    }
  }

  // Send/receive data
  MPICommunicator mpi_comm;
  boost::mpi::communicator comm(*mpi_comm, boost::mpi::comm_attach);
  std::vector<boost::mpi::request> reqs;
  std::map<std::size_t, std::vector<std::size_t> >::const_iterator data;
  std::map<std::size_t, std::vector<std::size_t> > recv;
  for (data = send_local_indices.begin(); data != send_local_indices.end(); ++data)
  {
    reqs.push_back(comm.isend(data->first, MPI::process_number(), data->second));
    reqs.push_back(comm.irecv(data->first, data->first, recv[data->first]));
  }
  boost::mpi::wait_all(reqs.begin(), reqs.end());

  // Debug printing
  {
    const std::size_t local_proc = 3;
    const std::size_t remote_proc = 1;
    if (MPI::process_number() == local_proc && send_local_indices.find(remote_proc) != send_local_indices.end())
    {
      const std::vector<std::size_t>& data = send_local_indices.find(remote_proc)->second;
      cout << "Start IO on proc " << local_proc << endl;
      for (std::size_t i = 0; i < data.size(); ++i)
        cout << data[i] << endl;
    }
    else if (MPI::process_number() == local_proc)
      cout << "Do not share data with " << remote_proc << endl;

    MPI::barrier();
    cout << "-------------------------" << endl;
    MPI::barrier();

    if (MPI::process_number() == remote_proc)
    {
      const std::vector<std::size_t>& data = recv[local_proc];
      cout << "Data received on "<< remote_proc << " from " << local_proc << endl;
      for (std::size_t i = 0; i < data.size(); ++i)
        cout << data[i] << endl;
    }
  }

  std::map<std::size_t, std::vector<std::pair<std::size_t, std::size_t> > > sharing;

  return sharing;
}
*/
//-----------------------------------------------------------------------------
void MeshPartitioning::partition(Mesh& mesh, const LocalMeshData& mesh_data)
{
  // Compute cell partition
  std::vector<std::size_t> cell_partition;
  const std::string partitioner = parameters["mesh_partitioner"];
  if (partitioner == "SCOTCH")
    SCOTCH::compute_partition(cell_partition, mesh_data);
  else if (partitioner == "ParMETIS")
    ParMETIS::compute_partition(cell_partition, mesh_data);
  else
  {
    dolfin_error("MeshPartitioning.cpp",
                 "partition mesh",
                 "Mesh partitioner '%s' is not known. Known partitioners are 'SCOTCH' or 'ParMETIS'", partitioner.c_str());
  }

  // Distribute cells
  Timer timer("PARALLEL 2: Distribute mesh (cells and vertices)");
  std::vector<std::size_t> global_cell_indices;
  boost::multi_array<std::size_t, 2> cell_vertices;
  distribute_cells(mesh_data, cell_partition, global_cell_indices, cell_vertices);

  // Distribute vertices
  std::vector<std::size_t> vertex_indices;
  boost::multi_array<double, 2> vertex_coordinates;
  std::map<std::size_t, std::size_t> vertex_global_to_local;
  distribute_vertices(mesh_data, cell_vertices, vertex_indices,
                      vertex_global_to_local, vertex_coordinates);
  timer.stop();

  // Build mesh
  build_mesh(mesh, global_cell_indices, cell_vertices, vertex_indices,
             vertex_coordinates, vertex_global_to_local,
             mesh_data.tdim, mesh_data.gdim, mesh_data.num_global_cells,
             mesh_data.num_global_vertices);
}
//-----------------------------------------------------------------------------
void MeshPartitioning::distribute_cells(const LocalMeshData& mesh_data,
                                   const std::vector<std::size_t>& cell_partition,
                                   std::vector<std::size_t>& global_cell_indices,
                                   boost::multi_array<std::size_t, 2>& cell_vertices)
{
  // This function takes the partition computed by the partitioner
  // (which tells us to which process each of the local cells stored in
  // LocalMeshData on this process belongs. We use MPI::distribute to
  // redistribute all cells (the global vertex indices of all cells).

  // Get dimensions of local mesh_data
  const std::size_t num_local_cells = mesh_data.cell_vertices.size();
  dolfin_assert(mesh_data.global_cell_indices.size() == num_local_cells);
  const std::size_t num_cell_vertices = mesh_data.num_vertices_per_cell;
  if (!mesh_data.cell_vertices.empty())
  {
    if (mesh_data.cell_vertices[0].size() != num_cell_vertices)
    {
      dolfin_error("MeshPartitioning.cpp",
                   "distribute cells",
                   "Mismatch in number of cell vertices (%d != %d) on process %d",
                   mesh_data.cell_vertices[0].size(), num_cell_vertices, MPI::process_number());
    }
  }

  // Build array of cell-vertex connectivity and partition vector
  // Distribute the global cell number as well
  std::vector<std::size_t> send_cell_vertices;
  std::vector<std::size_t> destinations_cell_vertices;
  send_cell_vertices.reserve(num_local_cells*(num_cell_vertices + 1));
  destinations_cell_vertices.reserve(num_local_cells*(num_cell_vertices + 1));
  for (std::size_t i = 0; i < num_local_cells; i++)
  {
    send_cell_vertices.push_back(mesh_data.global_cell_indices[i]);
    destinations_cell_vertices.push_back(cell_partition[i]);
    for (std::size_t j = 0; j < num_cell_vertices; j++)
    {
      send_cell_vertices.push_back(mesh_data.cell_vertices[i][j]);
      destinations_cell_vertices.push_back(cell_partition[i]);
    }
  }

  // Distribute cell-vertex connectivity
  std::vector<std::size_t> received_cell_vertices;
  MPI::distribute(send_cell_vertices, destinations_cell_vertices,
                  received_cell_vertices);
  dolfin_assert(received_cell_vertices.size() % (num_cell_vertices + 1) == 0);
  destinations_cell_vertices.clear();

  // Put mesh_data back into mesh_data.cell_vertices
  const std::size_t num_new_local_cells = received_cell_vertices.size()/(num_cell_vertices + 1);
  cell_vertices.resize(boost::extents[num_new_local_cells][num_cell_vertices]);
  global_cell_indices.resize(num_new_local_cells);

  // Loop over new cells
  for (std::size_t i = 0; i < num_new_local_cells; ++i)
  {
    global_cell_indices[i] = received_cell_vertices[i*(num_cell_vertices + 1)];
    for (std::size_t j = 0; j < num_cell_vertices; ++j)
      cell_vertices[i][j] = received_cell_vertices[i*(num_cell_vertices + 1) + j + 1];
  }
}
//-----------------------------------------------------------------------------
void MeshPartitioning::distribute_vertices(const LocalMeshData& mesh_data,
                    const boost::multi_array<std::size_t, 2>& cell_vertices,
                    std::vector<std::size_t>& vertex_indices,
                    std::map<std::size_t, std::size_t>& glob2loc,
                    boost::multi_array<double, 2>& vertex_coordinates)
{
  // This function distributes all vertices (coordinates and local-to-global
  // mapping) according to the cells that are stored on each process. This
  // happens in several stages: First each process figures out which vertices
  // it needs (by looking at its cells) and where those vertices are located.
  // That information is then distributed so that each process learns where
  // it needs to send its vertices.

  // Get number of processes
  const std::size_t num_processes = MPI::num_processes();

  // Get geometric dimension
  const std::size_t gdim = mesh_data.gdim;

  // Compute which vertices we need
  std::set<std::size_t> needed_vertex_indices;
  boost::multi_array<std::size_t, 2>::const_iterator vertices;
  for (vertices = cell_vertices.begin(); vertices != cell_vertices.end(); ++vertices)
    needed_vertex_indices.insert(vertices->begin(), vertices->end());

  // Compute where (process number) the vertices we need are located
  std::vector<std::size_t> send_vertex_indices;
  std::vector<std::size_t> destinations_vertex;
  std::vector<std::vector<std::size_t> > vertex_location(num_processes);
  std::set<std::size_t>::const_iterator required_vertex;
  for (required_vertex = needed_vertex_indices.begin();
        required_vertex != needed_vertex_indices.end(); ++required_vertex)
  {
    // Get process that has required vertex
    const std::size_t location = MPI::index_owner(*required_vertex, mesh_data.num_global_vertices);
    destinations_vertex.push_back(location);
    send_vertex_indices.push_back(*required_vertex);
    vertex_location[location].push_back(*required_vertex);
  }

  // Send required vertices to other proceses, and receive back vertices
  // required by othe processes.
  std::vector<std::size_t> received_vertex_indices;
  std::vector<std::size_t> sources_vertex;
  MPI::distribute(send_vertex_indices, destinations_vertex,
                  received_vertex_indices, sources_vertex);
  dolfin_assert(received_vertex_indices.size() == sources_vertex.size());

  // Distribute vertex coordinates
  std::vector<double> send_vertex_coordinates;
  std::vector<std::size_t> destinations_vertex_coordinates;
  const std::pair<std::size_t, std::size_t> local_vertex_range = MPI::local_range(mesh_data.num_global_vertices);
  for (std::size_t i = 0; i < sources_vertex.size(); ++i)
  {
    dolfin_assert(received_vertex_indices[i] >= local_vertex_range.first
                      && received_vertex_indices[i] < local_vertex_range.second);
    const std::size_t location = received_vertex_indices[i] - local_vertex_range.first;
    for (std::size_t j = 0; j < gdim; ++j)
    {
      send_vertex_coordinates.push_back(mesh_data.vertex_coordinates[location][j]);
      destinations_vertex_coordinates.push_back(sources_vertex[i]);
    }
  }
  std::vector<double> received_vertex_coordinates;
  std::vector<std::size_t> sources_vertex_coordinates;
  MPI::distribute(send_vertex_coordinates, destinations_vertex_coordinates,
                  received_vertex_coordinates, sources_vertex_coordinates);

  // Set index counters to first position in recieve buffers
  std::vector<std::size_t> index_counters(num_processes, 0);

  // Clear data
  vertex_indices.clear();
  glob2loc.clear();

  // Store coordinates and construct global to local mapping
  const std::size_t num_local_vertices = received_vertex_coordinates.size()/gdim;
  vertex_coordinates.resize(boost::extents[num_local_vertices][gdim]);
  vertex_indices.resize(num_local_vertices);
  for (std::size_t i = 0; i < num_local_vertices; ++i)
  {
    for (std::size_t j = 0; j < gdim; ++j)
      vertex_coordinates[i][j] = received_vertex_coordinates[i*gdim + j];

    const std::size_t sender_process = sources_vertex_coordinates[i*gdim];
    const std::size_t global_vertex_index
      = vertex_location[sender_process][index_counters[sender_process]++];
    glob2loc[global_vertex_index] = i;
    vertex_indices[i] = global_vertex_index;
  }
}
//-----------------------------------------------------------------------------
void MeshPartitioning::build_mesh(Mesh& mesh,
              const std::vector<std::size_t>& global_cell_indices,
              const boost::multi_array<std::size_t, 2>& cell_global_vertices,
              const std::vector<std::size_t>& vertex_indices,
              const boost::multi_array<double, 2>& vertex_coordinates,
              const std::map<std::size_t, std::size_t>& vertex_global_to_local,
              std::size_t tdim, std::size_t gdim, std::size_t num_global_cells,
              std::size_t num_global_vertices)
{
  Timer timer("PARALLEL 3: Build mesh (from local mesh data)");

  // Get number of processes and process number
  const std::size_t num_processes = MPI::num_processes();
  const std::size_t process_number = MPI::process_number();

  // Open mesh for editing
  mesh.clear();
  MeshEditor editor;
  editor.open(mesh, tdim, gdim);

  // Add vertices
  editor.init_vertices(vertex_coordinates.size());
  Point p(gdim);
  dolfin_assert(vertex_indices.size() == vertex_coordinates.size());
  for (std::size_t i = 0; i < vertex_coordinates.size(); ++i)
  {
    for (std::size_t j = 0; j < gdim; ++j)
      p[j] = vertex_coordinates[i][j];
    editor.add_vertex_global(i, vertex_indices[i], p);
  }

  // Add cells
  editor.init_cells(cell_global_vertices.size());
  const std::size_t num_cell_vertices = tdim + 1;
  std::vector<std::size_t> cell(num_cell_vertices);
  for (std::size_t i = 0; i < cell_global_vertices.size(); ++i)
  {
    for (std::size_t j = 0; j < num_cell_vertices; ++j)
    {
      // Get local cell vertex
      std::map<std::size_t, std::size_t>::const_iterator iter
          = vertex_global_to_local.find(cell_global_vertices[i][j]);
      dolfin_assert(iter != vertex_global_to_local.end());
      cell[j] = iter->second;
    }
    editor.add_cell(i, global_cell_indices[i], cell);
  }

  // Close mesh: Note that this must be done after creating the global
  // vertex map or otherwise the ordering in mesh.close() will be wrong
  // (based on local numbers).
  editor.close();

  // Set global number of cells and vertices
  mesh.topology().init_global(0, num_global_vertices);
  mesh.topology().init_global(tdim,  num_global_cells);

  // Construct boundary mesh
  BoundaryMesh bmesh(mesh);

  const MeshFunction<std::size_t>& boundary_vertex_map = bmesh.vertex_map();
  const std::size_t boundary_size = boundary_vertex_map.size();

  // Build sorted array of global boundary vertex indices (global
  // numbering)
  std::vector<std::size_t> global_vertex_send(boundary_size);
  for (std::size_t i = 0; i < boundary_size; ++i)
    global_vertex_send[i] = vertex_indices[boundary_vertex_map[i]];
  std::sort(global_vertex_send.begin(), global_vertex_send.end());

  // Receive buffer
  std::vector<std::size_t> global_vertex_recv;

  // Create shared_vertices data structure: mapping from shared vertices
  // to list of neighboring processes
  std::map<std::size_t, std::set<std::size_t> >& shared_vertices
        = mesh.topology().shared_entities(0);
  shared_vertices.clear();

  // Distribute boundaries and build mappings
  for (std::size_t i = 1; i < num_processes; ++i)
  {
    // We send data to process p - i (i steps to the left)
    const int p = (process_number - i + num_processes) % num_processes;

    // We receive data from process p + i (i steps to the right)
    const int q = (process_number + i) % num_processes;

    // Send and receive
    MPI::send_recv(global_vertex_send, p, global_vertex_recv, q);

    // Compute intersection of global indices
    std::vector<std::size_t> intersection(std::min(global_vertex_send.size(),
                                                   global_vertex_recv.size()));
    std::vector<std::size_t>::iterator intersection_end
      = std::set_intersection(global_vertex_send.begin(), global_vertex_send.end(),
                              global_vertex_recv.begin(), global_vertex_recv.end(),
                              intersection.begin());

    // Fill shared vertices information
    std::vector<std::size_t>::const_iterator global_index;
    for (global_index = intersection.begin(); global_index != intersection_end; ++global_index)
    {
      // Global indices
      shared_vertices[*global_index].insert(q);

      /*
      // Local indices
      std::map<std::size_t, std::size_t>::const_iterator local_index;
      local_index = vertex_global_to_local.find(*global_index);
      dolfin_assert(local_index != vertex_global_to_local.end());
      shared_vertices[local_index->second].insert(q);
      */
    }
  }
}
//-----------------------------------------------------------------------------
void MeshPartitioning::build_mesh_domains(Mesh& mesh,
                                          const LocalMeshData& local_data)
{
  // Local domain data
  const std::map<std::size_t, std::vector< std::pair<std::pair<std::size_t, std::size_t>, std::size_t> > >
    domain_data = local_data.domain_data;
  if (domain_data.empty())
    return;

  // Initialse mesh domains
  const std::size_t D = mesh.topology().dim();
  mesh.domains().init(D);

  std::map<std::size_t, std::vector< std::pair<std::pair<std::size_t, std::size_t>, std::size_t> > >::const_iterator dim_data;
  for (dim_data = domain_data.begin(); dim_data != domain_data.end(); ++dim_data)
  {
    // Get mesh value collection used for marking
    const std::size_t dim = dim_data->first;
    dolfin_assert(mesh.domains().markers(dim));
    MeshValueCollection<std::size_t>& markers = *(mesh.domains().markers(dim));

    const std::vector< std::pair<std::pair<std::size_t, std::size_t>, std::size_t> >&
        local_value_data = dim_data->second;
    build_mesh_value_collection(mesh, local_value_data, markers);
  }
}
//-----------------------------------------------------------------------------
