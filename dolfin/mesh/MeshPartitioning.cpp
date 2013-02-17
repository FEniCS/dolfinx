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
// Last changed: 2013-02-16

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
#include <dolfin/graph/ZoltanPartition.h>
#include <dolfin/parameter/GlobalParameters.h>
#include "BoundaryMesh.h"
#include "DistributedMeshTools.h"
#include "Facet.h"
#include "LocalMeshData.h"
#include "Mesh.h"
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
  // necessary to distinguish between facets on an exterior boundary and
  // facets on a partition boundary (see
  // https://bugs.launchpad.net/dolfin/+bug/733834).
  DistributedMeshTools::init_facet_cell_connections(mesh);
}
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
  else if (partitioner == "ParMETIS_repart")
    ParMETIS::recompute_partition(cell_partition, mesh_data);
  else if (partitioner == "Zoltan")
    ZoltanPartition::compute_partition(cell_partition, mesh_data);
  else
  {
    dolfin_error("MeshPartitioning.cpp",
                 "partition mesh",
                 "Mesh partitioner '%s' is not known. Known partitioners are 'SCOTCH', 'ParMETIS'", partitioner.c_str());
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
  // LocalMeshData on this process belongs. We use MPI::all_to_all to
  // redistribute all cells (the global vertex indices of all cells).

  // Number of MPI processes
  const std::size_t num_processes = MPI::num_processes();

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
  std::vector<std::vector<std::size_t> > send_cell_vertices(num_processes);
  for (std::size_t i = 0; i < num_local_cells; i++)
  {
    const std::size_t dest = cell_partition[i];
    send_cell_vertices[dest].push_back(mesh_data.global_cell_indices[i]);
    for (std::size_t j = 0; j < num_cell_vertices; j++)
      send_cell_vertices[dest].push_back(mesh_data.cell_vertices[i][j]);
  }

  // Distribute cell-vertex connectivity
  std::vector<std::vector<std::size_t> > received_cell_vertices(num_processes);
  MPI::all_to_all(send_cell_vertices, received_cell_vertices);

  // Count number of received cells
  std::size_t num_new_local_cells = 0;
  for (std::size_t p = 0; p < received_cell_vertices.size(); ++p)
    num_new_local_cells += received_cell_vertices[p].size()/(num_cell_vertices + 1);

  // Put mesh_data back into mesh_data.cell_vertices
  cell_vertices.resize(boost::extents[num_new_local_cells][num_cell_vertices]);
  global_cell_indices.resize(num_new_local_cells);

  // Loop over new cells
  std::size_t c = 0;
  for (std::size_t p = 0; p < num_processes; ++p)
  {
    for (std::size_t i = 0; i < received_cell_vertices[p].size(); i += (num_cell_vertices + 1))
    {
      global_cell_indices[c] = received_cell_vertices[p][i];
      for (std::size_t j = 0; j < num_cell_vertices; ++j)
        cell_vertices[c][j] = received_cell_vertices[p][i + 1 + j];

      ++c;
    }
  }
}
//-----------------------------------------------------------------------------
void MeshPartitioning::distribute_vertices(const LocalMeshData& mesh_data,
                    const boost::multi_array<std::size_t, 2>& cell_vertices,
                    std::vector<std::size_t>& vertex_indices,
                    std::map<std::size_t, std::size_t>& vertex_global_to_local,
                    boost::multi_array<double, 2>& vertex_coordinates)
{
  // This function distributes all vertices (coordinates and
  // local-to-global mapping) according to the cells that are stored on
  // each process. This happens in several stages: First each process
  // figures out which vertices it needs (by looking at its cells)
  // and where those vertices are located. That information is then
  // distributed so that each process learns where it needs to send
  // its vertices.

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
  std::vector<std::vector<std::size_t> > send_vertex_indices(num_processes);
  std::vector<std::vector<std::size_t> > vertex_location(num_processes);
  std::set<std::size_t>::const_iterator required_vertex;
  for (required_vertex = needed_vertex_indices.begin();
        required_vertex != needed_vertex_indices.end(); ++required_vertex)
  {
    // Get process that has required vertex
    const std::size_t location
      = MPI::index_owner(*required_vertex, mesh_data.num_global_vertices);
    send_vertex_indices[location].push_back(*required_vertex);
    vertex_location[location].push_back(*required_vertex);
  }

  // Send required vertices to other processes, and receive back vertices
  // required by other processes.
  std::vector<std::vector<std::size_t> > received_vertex_indices;
  MPI::all_to_all(send_vertex_indices, received_vertex_indices);

  // Distribute vertex coordinates
  std::vector<std::vector<double> > send_vertex_coordinates(num_processes);
  const std::pair<std::size_t, std::size_t> local_vertex_range
      = MPI::local_range(mesh_data.num_global_vertices);
  for (std::size_t p = 0; p < num_processes; ++p)
  {
    send_vertex_coordinates[p].reserve(received_vertex_indices[p].size()*gdim);
    for (std::size_t i = 0; i < received_vertex_indices[p].size(); ++i)
    {
      dolfin_assert(received_vertex_indices[p][i] >= local_vertex_range.first
                        && received_vertex_indices[p][i] < local_vertex_range.second);
      const std::size_t location = received_vertex_indices[p][i] - local_vertex_range.first;
      for (std::size_t j = 0; j < gdim; ++j)
        send_vertex_coordinates[p].push_back(mesh_data.vertex_coordinates[location][j]);
    }
  }
  std::vector<std::vector<double> > received_vertex_coordinates;
  MPI::all_to_all(send_vertex_coordinates, received_vertex_coordinates);

  // Set index counters to first position in receive buffers
  std::vector<std::size_t> index_counters(num_processes, 0);

  // Clear data
  vertex_indices.clear();
  vertex_global_to_local.clear();

  // Count number of local vertices
  std::size_t num_local_vertices = 0;
  for (std::size_t p = 0; p < num_processes; ++p)
    num_local_vertices += received_vertex_coordinates[p].size()/gdim;

  // Store coordinates and construct global to local mapping
  vertex_coordinates.resize(boost::extents[num_local_vertices][gdim]);
  vertex_indices.resize(num_local_vertices);
  std::size_t v = 0;
  for (std::size_t p = 0; p < num_processes; ++p)
  {
    for (std::size_t i = 0; i < received_vertex_coordinates[p].size(); i += gdim)
    {
      for (std::size_t j = 0; j < gdim; ++j)
        vertex_coordinates[v][j] = received_vertex_coordinates[p][i + j];

      const std::size_t global_vertex_index = vertex_location[p][index_counters[p]++];
      vertex_global_to_local[global_vertex_index] = v;
      vertex_indices[v] = global_vertex_index;

      ++v;
    }
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
  BoundaryMesh bmesh(mesh, "exterior");

  const MeshFunction<std::size_t>& boundary_vertex_map = bmesh.entity_map(0);
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

  // FIXME: Remove computation from inside communication loop

  // Build shared vertex to sharing processes map
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
      // Get local index
      std::map<std::size_t, std::size_t>::const_iterator local_index;
      local_index = vertex_global_to_local.find(*global_index);
      dolfin_assert(local_index != vertex_global_to_local.end());

      // Insert (local index, [proc])
      shared_vertices[local_index->second].insert(q);
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
