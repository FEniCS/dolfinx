
// Copyright (C) 2012 Chris N Richardson
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
// Modified by Garth N. Wells, 2012
//
// First added:  2012-06-01
// Last changed: 2013-04-10

#ifdef HAS_HDF5

#include <cstdio>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/assign.hpp>
#include <boost/multi_array.hpp>
#include <boost/unordered_map.hpp>

#include <dolfin/common/constants.h>
#include <dolfin/common/MPI.h>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/common/Timer.h>
#include <dolfin/function/Function.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/LocalMeshData.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshEditor.h>
#include <dolfin/mesh/MeshPartitioning.h>
#include <dolfin/mesh/MeshEntityIterator.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/Vertex.h>

#include "HDF5File.h"
#include "HDF5Interface.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
HDF5File::HDF5File(const std::string filename, const std::string file_mode, bool use_mpiio)
  : hdf5_file_open(false), hdf5_file_id(0),
    mpi_io(MPI::num_processes() > 1 && use_mpiio ? true : false)
{
  // HDF5 chunking
  parameters.add("chunking", false);

  // Open HDF5 file
  hdf5_file_id = HDF5Interface::open_file(filename, file_mode, mpi_io);
  hdf5_file_open = true;
}
//-----------------------------------------------------------------------------
HDF5File::~HDF5File()
{
  // Close HDF5 file
  if (hdf5_file_open)
    HDF5Interface::close_file(hdf5_file_id);
}
//-----------------------------------------------------------------------------
void HDF5File::flush()
{
  dolfin_assert(hdf5_file_open);
  HDF5Interface::flush_file(hdf5_file_id);
}
//-----------------------------------------------------------------------------
void HDF5File::write(const GenericVector& x, const std::string dataset_name)
{
  dolfin_assert(x.size() > 0);
  dolfin_assert(hdf5_file_open);

  // Get all local data
  std::vector<double> local_data;
  x.get_local(local_data);

  // Write data to file
  std::pair<std::size_t, std::size_t> local_range = x.local_range();
  const bool chunking = parameters["chunking"];
  const std::vector<std::size_t> global_size(1, x.size());
  HDF5Interface::write_dataset(hdf5_file_id, dataset_name, local_data,
                               local_range, global_size, mpi_io, chunking);

  // Add partitioning attribute to dataset
  std::vector<std::size_t> partitions;
  MPI::gather(local_range.first, partitions);
  MPI::broadcast(partitions);

  HDF5Interface::add_attribute(hdf5_file_id, dataset_name, "partition",
                               partitions);
}
//-----------------------------------------------------------------------------
void HDF5File::write(const Mesh& mesh, const std::string name)
{
  write(mesh, mesh.topology().dim(), name);
}
//-----------------------------------------------------------------------------
void HDF5File::write(const Mesh& mesh, std::size_t cell_dim,
                     const std::string name)
{
  Timer t0("HDF5: write mesh to file");

  dolfin_assert(hdf5_file_open);

  // ---------- Vertices (coordinates)
  {
    // Write vertex data to HDF5 file
    const std::string coord_dataset =  name + "/coordinates";

    // Copy coordinates and indices and remove off-process values
    const std::size_t gdim = mesh.geometry().dim();
    const std::vector<double> vertex_coords
      = reorder_vertices_by_global_indices(mesh);

    // Write coordinates out from each process
    std::vector<std::size_t> global_size(2);
    global_size[0] = MPI::sum(vertex_coords.size()/gdim);
    global_size[1] = gdim;
    dolfin_assert(global_size[0] == mesh.size_global(0));
    write_data(coord_dataset, vertex_coords, global_size);
  }

  // ---------- Topology
  {
    std::vector<std::size_t> topological_data;
    topological_data.reserve(mesh.num_entities(cell_dim)*(cell_dim + 1));

    if (cell_dim == mesh.topology().dim() || MPI::num_processes() == 1)
    {
      // Usual case, with cell output, and/or none shared with another
      // process.
      // Get/build topology data
      for (MeshEntityIterator c(mesh, cell_dim); !c.end(); ++c)
        for (VertexIterator v(*c); !v.end(); ++v)
          topological_data.push_back(v->global_index());
    }
    else
    {
      // Drop duplicate topology for shared entities of less than mesh dimension

      // If not already numbered, number entities of order cell_dim
      // so we can get shared_entities
      DistributedMeshTools::number_entities(mesh, cell_dim);

      const std::size_t my_rank = MPI::process_number();
      const std::map<unsigned int, std::set<unsigned int> >& shared_entities
        = mesh.topology().shared_entities(cell_dim);

      for (MeshEntityIterator c(mesh, cell_dim); !c.end(); ++c)
      {
        std::map<unsigned int, std::set<unsigned int> >::const_iterator
          sh = shared_entities.find(c->index());

        // If unshared, or owned locally, append to topology
        if (sh == shared_entities.end())
        {
          for (VertexIterator v(*c); !v.end(); ++v)
            topological_data.push_back(v->global_index());
        }
        else
        {
          std::set<unsigned int>::const_iterator lowest_proc = sh->second.begin();
          if(*lowest_proc > my_rank)
          {
            for (VertexIterator v(*c); !v.end(); ++v)
              topological_data.push_back(v->global_index());
          }
        }
      }
    }

    // Write topology data
    const std::string topology_dataset =  name + "/topology";
    std::vector<std::size_t> global_size(2);
    global_size[0] = MPI::sum(topological_data.size()/(cell_dim + 1));
    global_size[1] = cell_dim + 1;
    dolfin_assert(global_size[0] == mesh.size_global(cell_dim));
    write_data(topology_dataset, topological_data, global_size);

    // Add cell type attribute
    HDF5Interface::add_attribute(hdf5_file_id, topology_dataset,
                                 "celltype", cell_type(cell_dim, mesh));
  }
}
//-----------------------------------------------------------------------------
void HDF5File::write(const MeshFunction<std::size_t>& meshfunction, const std::string name)
{
  write_mesh_function(meshfunction, name);
}
//-----------------------------------------------------------------------------
void HDF5File::read(MeshFunction<std::size_t>& meshfunction, const std::string name)
{
  read_mesh_function(meshfunction, name);
}
//-----------------------------------------------------------------------------
void HDF5File::write(const MeshFunction<int>& meshfunction, const std::string name)
{
  write_mesh_function(meshfunction, name);
}
//-----------------------------------------------------------------------------
void HDF5File::read(MeshFunction<int>& meshfunction, const std::string name)
{
  read_mesh_function(meshfunction, name);
}
//-----------------------------------------------------------------------------
void HDF5File::write(const MeshFunction<double>& meshfunction, const std::string name)
{
  write_mesh_function(meshfunction, name);
}
//-----------------------------------------------------------------------------
void HDF5File::read(MeshFunction<double>& meshfunction, const std::string name)
{
  read_mesh_function(meshfunction, name);
}
//-----------------------------------------------------------------------------
template <typename T>
void HDF5File::read_mesh_function(MeshFunction<T>& meshfunction, const std::string mesh_name)
{

  const Mesh& mesh = meshfunction.mesh();

  dolfin_assert(hdf5_file_open);
  
  const std::vector<std::string> _dataset_list =
    HDF5Interface::dataset_list(hdf5_file_id, mesh_name);

  std::string topology_name = search_list(_dataset_list,"topology");
  if (topology_name.size() == 0)
  {
    dolfin_error("HDF5File.cpp",
                 "read topology dataset",
                 "Dataset not found");
  }
  topology_name = mesh_name + "/" + topology_name;

  // Look for Coordinates dataset - but not used
  std::string coordinates_name=search_list(_dataset_list,"coordinates");
  if(coordinates_name.size()==0)
  {
    dolfin_error("HDF5File.cpp",
                 "read coordinates dataset",
                 "Dataset not found");
  }
  coordinates_name = mesh_name + "/" + coordinates_name;

  // Look for Values dataset
  std::string values_name=search_list(_dataset_list,"values");
  if(coordinates_name.size()==0)
  {
    dolfin_error("HDF5File.cpp",
                 "read values dataset",
                 "Dataset not found");
  }
  values_name = mesh_name + "/" + values_name;

  // --- Topology ---
  // Discover size of topology dataset
  const std::vector<std::size_t> topology_dim
      = HDF5Interface::get_dataset_size(hdf5_file_id, topology_name);

  // Some consistency checks

  const std::size_t num_global_cells = topology_dim[0];
  const std::size_t vert_per_cell = topology_dim[1];
  const std::size_t cell_dim = vert_per_cell - 1;

  if(cell_dim != meshfunction.dim())
  {
    dolfin_error("HDF5File.cpp",
                 "read meshfunction topology",
                 "Cell dimension mismatch");
  }

  // Ensure size_global(cell_dim) is set
  DistributedMeshTools::number_entities(mesh, cell_dim);

  if(num_global_cells != mesh.size_global(cell_dim))
  {
    dolfin_error("HDF5File.cpp",
                 "read meshfunction topology",
                 "Mesh dimension mismatch");
  }

  // Divide up cells ~equally between processes
  const std::pair<std::size_t,std::size_t> cell_range = MPI::local_range(num_global_cells);
  const std::size_t num_read_cells = cell_range.second - cell_range.first;

  // Read a block of cells
  std::vector<std::size_t> topology_data;
  topology_data.reserve(num_read_cells*vert_per_cell);
  HDF5Interface::read_dataset(hdf5_file_id, topology_name, cell_range, topology_data);

  boost::multi_array_ref<std::size_t, 2> topology_array(topology_data.data(), 
                                                        boost::extents[num_read_cells][vert_per_cell]);

  std::vector<T> value_data;
  value_data.reserve(num_read_cells);
  HDF5Interface::read_dataset(hdf5_file_id, values_name, cell_range, value_data);

  // Now send the read data to each process on the basis of the first vertex of the entity,
  // since we do not know the global_index 
  const std::size_t num_processes = MPI::num_processes();
  const std::size_t max_vertex = mesh.size_global(0);

  std::vector<std::vector<std::size_t> > send_topology(num_processes);
  std::vector<std::vector<std::size_t> > receive_topology(num_processes);
  std::vector<std::vector<T> > send_values(num_processes);
  std::vector<std::vector<T> > receive_values(num_processes);

  for(std::size_t i = 0; i < num_read_cells ; ++i)
  {
    std::vector<std::size_t> cell_topology(vert_per_cell);
    for(std::size_t j = 0; j < vert_per_cell; ++j)
      cell_topology[j] = topology_array[i][j];
    
    std::sort(cell_topology.begin(), cell_topology.end());

    // Use first vertex to decide where to send this data
    const std::size_t send_to_process = MPI::index_owner(cell_topology.front(), max_vertex);

    send_topology[send_to_process].insert(send_topology[send_to_process].end(),
                                          cell_topology.begin(), cell_topology.end());
    send_values[send_to_process].push_back(value_data[i]);
  }
  
  MPI::all_to_all(send_topology, receive_topology);
  MPI::all_to_all(send_values, receive_values);

  // Generate requests for data from remote processes,
  // based on the first vertex of the MeshEntities which belong on this process
  // Send our process number, and our local index, so it can come back directly
  // to the right place
  std::vector<std::vector<std::size_t> > send_requests(num_processes);
  std::vector<std::vector<std::size_t> > receive_requests(num_processes);

  const std::size_t process_number = MPI::process_number();

  for(MeshEntityIterator cell(mesh, cell_dim); !cell.end(); ++cell)
  {
    std::vector<std::size_t> cell_topology;
    for(VertexIterator v(*cell); !v.end(); ++v)
    {
      cell_topology.push_back(v->global_index());
    }
    std::sort(cell_topology.begin(), cell_topology.end());

    // Use first vertex to decide where to send this request
    std::size_t send_to_process = MPI::index_owner(cell_topology.front(), max_vertex);
    // Map to this process and local index by appending to send data
    cell_topology.push_back(cell->index());
    cell_topology.push_back(process_number);
    send_requests[send_to_process].insert(send_requests[send_to_process].end(),
                                          cell_topology.begin(), cell_topology.end());
  }

  MPI::all_to_all(send_requests, receive_requests);

  // At this point, the data with its associated vertices 
  // is in receive_values and receive_topology
  // and the final destinations are stored in receive_requests
  // as [vertices][index][process][vertices][index][process]...
  // Some data will have more than one destination

  // Create a mapping from the topology vector to the desired data
  typedef boost::unordered_map<std::vector<std::size_t>, T> VectorKeyMap;
  VectorKeyMap cell_to_data;
  
  for(std::size_t i = 0; i < receive_values.size(); ++i)
  {
    dolfin_assert(receive_values[i].size()*vert_per_cell == receive_topology[i].size());
    std::vector<std::size_t>::iterator p = receive_topology[i].begin();
    for(std::size_t j = 0; j < receive_values[i].size(); ++j)
    {
      const std::vector<std::size_t> cell(p, p + vert_per_cell);
      cell_to_data[cell] = receive_values[i][j];
      p += vert_per_cell;
    }
  }

  // Clear vectors for reuse - now to send values and indices to final destination
  send_topology = std::vector<std::vector<std::size_t> >(num_processes);
  send_values = std::vector<std::vector<T> >(num_processes);

  // Go through requests, which are stacked as [vertex, vertex, ...] [index] [proc] etc.
  // Use the vertices as the key for the map (above) to retrieve the data to send to proc
  for(std::size_t i = 0; i < receive_requests.size(); ++i)
  {
    for(std::vector<std::size_t>::iterator p = receive_requests[i].begin();
        p != receive_requests[i].end(); p += (vert_per_cell + 2))
    {
      const std::vector<std::size_t> cell(p, p + vert_per_cell);
      const std::size_t remote_index = *(p + vert_per_cell);
      const std::size_t send_to_proc = *(p + vert_per_cell + 1);

      const typename VectorKeyMap::iterator find_cell = cell_to_data.find(cell);
      dolfin_assert(find_cell != cell_to_data.end());
      send_values[send_to_proc].push_back(find_cell->second);
      send_topology[send_to_proc].push_back(remote_index);
    }
  }
  
  MPI::all_to_all(send_topology, receive_topology);
  MPI::all_to_all(send_values, receive_values);

  // At this point, receive_topology should only list the local indices
  // and received values should have the appropriate values for each

  for(std::size_t i = 0; i < receive_values.size(); ++i)
  {
    dolfin_assert(receive_values[i].size() == receive_topology[i].size());
    for(std::size_t j = 0; j < receive_values[i].size(); ++j)
    {
      meshfunction[receive_topology[i][j]] = receive_values[i][j];
    }
  }
  
}
//-----------------------------------------------------------------------------
template <typename T>
void HDF5File::write_mesh_function(const MeshFunction<T>& meshfunction, const std::string name)
{

  if (meshfunction.size() == 0)
  {
    dolfin_error("HDF5File.cpp",
                 "save empty MeshFunction",
                 "No values in MeshFunction");
  }

  const Mesh& mesh = meshfunction.mesh();
  const std::size_t cell_dim = meshfunction.dim();

  // Write a mesh for the MeshFunction - this will also globally
  // number the entities if needed
  write(mesh, cell_dim, name);

  // Storage for output values
  std::vector<T> data_values;

  if(cell_dim == mesh.topology().dim() || MPI::num_processes() == 1)
  {
    // No duplicates
    data_values.assign(meshfunction.values(), meshfunction.values() + meshfunction.size());
  }
  else
  {
    data_values.reserve(mesh.size(cell_dim));

    // Drop duplicate data
    const std::size_t my_rank = MPI::process_number();
    const std::map<unsigned int, std::set<unsigned int> >& shared_entities
      = mesh.topology().shared_entities(cell_dim);

    for(std::size_t i = 0; i < meshfunction.size(); ++i)
    {
      std::map<unsigned int, std::set<unsigned int> >::const_iterator sh
        = shared_entities.find(i);

      // If unshared, or shared and locally owned, append to vector
      if(sh == shared_entities.end())
        data_values.push_back(meshfunction[i]);
      else
      {
        std::set<unsigned int>::iterator lowest_proc = sh->second.begin();
        if(*lowest_proc > my_rank)
          data_values.push_back(meshfunction[i]);
      }
    }
  }

  // Write values to HDF5
  std::vector<std::size_t> global_size(1, MPI::sum(data_values.size()));

  write_data(name + "/values", data_values, global_size);

}
//-----------------------------------------------------------------------------
void HDF5File::read(GenericVector& x, const std::string dataset_name,
                    const bool use_partition_from_file)
{
  dolfin_assert(hdf5_file_open);

  // Check for data set exists
  if (!HDF5Interface::has_dataset(hdf5_file_id, dataset_name))
    error("Data set with name \"%s\" does not exist", dataset_name.c_str());

  // Get dataset rank
  const std::size_t rank = HDF5Interface::dataset_rank(hdf5_file_id, dataset_name);
  dolfin_assert(rank == 1);

  // Get global dataset size
  const std::vector<std::size_t> data_size
      = HDF5Interface::get_dataset_size(hdf5_file_id, dataset_name);

  // Check that rank is 1
  dolfin_assert(data_size.size() == 1);

  // Check input vector, and re-size if not already sized
  if (x.size() == 0)
  {
    // Resize vector
    if (use_partition_from_file)
    {
      // Get partition from file
      std::vector<std::size_t> partitions;
      HDF5Interface::get_attribute(hdf5_file_id, dataset_name, "partition", partitions);

      // Check that number of MPI processes matches partitioning
      if(MPI::num_processes() != partitions.size())
      {
        dolfin_error("HDF5File.cpp",
                     "read vector from file",
                     "Different number of processes used when writing. Cannot restore partitioning");
      }

      // Add global size at end of partition vectors
      partitions.push_back(data_size[0]);

      // Initialise vector
      const std::size_t process_num = MPI::process_number();
      const std::pair<std::size_t, std::size_t>
          local_range(partitions[process_num], partitions[process_num + 1]);
      x.resize(local_range);
    }
    else
      x.resize(data_size[0]);
  }
  else if (x.size() != data_size[0])
  {
    dolfin_error("HDF5File.cpp",
                 "read vector from file",
                 "Size mis-match between vector in file and input vector");
  }

  // Get local range
  const std::pair<std::size_t, std::size_t> local_range = x.local_range();

  // Read data from file
  std::vector<double> data;
  HDF5Interface::read_dataset(hdf5_file_id, dataset_name, local_range, data);

  // Set data
  x.set_local(data);
}
//-----------------------------------------------------------------------------
void HDF5File::read(Mesh& input_mesh, const std::string mesh_name)
{
  dolfin_assert(hdf5_file_open);

  std::vector<std::string> _dataset_list =
    HDF5Interface::dataset_list(hdf5_file_id, mesh_name);

  std::string topology_name = search_list(_dataset_list,"topology");
  if (topology_name.size() == 0)
  {
    dolfin_error("HDF5File.cpp",
                 "read topology dataset",
                 "Dataset not found");
  }
  topology_name = mesh_name + "/" + topology_name;

  // Look for Coordinates dataset
  std::string coordinates_name=search_list(_dataset_list,"coordinates");
  if(coordinates_name.size()==0)
  {
    dolfin_error("HDF5File.cpp",
                 "read coordinates dataset",
                 "Dataset not found");
  }
  coordinates_name = mesh_name + "/" + coordinates_name;

  read_mesh_repartition(input_mesh, coordinates_name,
                                   topology_name);
}
//-----------------------------------------------------------------------------
void HDF5File::read_mesh_repartition(Mesh& input_mesh,
                                     const std::string coordinates_name,
                                     const std::string topology_name)
{
  Timer t("HDF5: read mesh");

  // Structure to store local mesh
  LocalMeshData mesh_data;
  mesh_data.clear();

  // --- Topology ---
  // Discover size of topology dataset
  std::vector<std::size_t> topology_dim
      = HDF5Interface::get_dataset_size(hdf5_file_id, topology_name);

  // Get total number of cells, as number of rows in topology dataset
  const std::size_t num_global_cells = topology_dim[0];
  mesh_data.num_global_cells = num_global_cells;

  // Set vertices-per-cell from number of columns
  const std::size_t num_vertices_per_cell = topology_dim[1];
  mesh_data.num_vertices_per_cell = num_vertices_per_cell;
  mesh_data.tdim = topology_dim[1] - 1;

  // Divide up cells ~equally between processes
  const std::pair<std::size_t,std::size_t> cell_range = MPI::local_range(num_global_cells);
  const std::size_t num_local_cells = cell_range.second - cell_range.first;

  // Read a block of cells
  std::vector<std::size_t> topology_data;
  topology_data.reserve(num_local_cells*num_vertices_per_cell);
  HDF5Interface::read_dataset(hdf5_file_id, topology_name, cell_range, topology_data);

  mesh_data.global_cell_indices.reserve(num_local_cells);
  for(std::size_t i = 0; i < num_local_cells; i++)
    mesh_data.global_cell_indices.push_back(cell_range.first + i);

  // Copy to boost::multi_array
  mesh_data.cell_vertices.resize(boost::extents[num_local_cells][num_vertices_per_cell]);
  std::copy(topology_data.begin(), topology_data.end(),
            mesh_data.cell_vertices.data());

  // --- Coordinates ---
  // Get dimensions of coordinate dataset
  std::vector<std::size_t> coords_dim
    = HDF5Interface::get_dataset_size(hdf5_file_id, coordinates_name);
  mesh_data.num_global_vertices = coords_dim[0];
  mesh_data.gdim = coords_dim[1];

  // Divide range into equal blocks for each process
  const std::pair<std::size_t, std::size_t> vertex_range =
                          MPI::local_range(mesh_data.num_global_vertices);
  const std::size_t num_local_vertices = vertex_range.second - vertex_range.first;

  // Read vertex data to temporary vector
  std::vector<double> coordinates_data;
  coordinates_data.reserve(num_local_vertices*mesh_data.gdim);
  HDF5Interface::read_dataset(hdf5_file_id, coordinates_name, vertex_range,
                              coordinates_data);

  // Copy to boost::multi_array
  mesh_data.vertex_coordinates.resize(boost::extents[num_local_vertices][mesh_data.gdim]);
  std::copy(coordinates_data.begin(), coordinates_data.end(),
            mesh_data.vertex_coordinates.data());

  // Fill vertex indices with values - not used in build_distributed_mesh
  mesh_data.vertex_indices.resize(num_local_vertices);
  for(std::size_t i = 0; i < mesh_data.vertex_coordinates.size(); ++i)
    mesh_data.vertex_indices[i] = vertex_range.first + i;

  // Build distributed mesh

  t.stop();

  if(MPI::num_processes() == 1)
    build_local_mesh(input_mesh, mesh_data);
  else
    MeshPartitioning::build_distributed_mesh(input_mesh, mesh_data);
}
//-----------------------------------------------------------------------------
void HDF5File::build_local_mesh(Mesh &mesh, const LocalMeshData& mesh_data) const
{
  // Create mesh for editing
  MeshEditor editor;
  dolfin_assert(mesh_data.tdim != 0);
  std::string cell_type_str = CellType::type2string((CellType::Type)mesh_data.tdim);

  editor.open(mesh, cell_type_str, mesh_data.tdim, mesh_data.gdim);
  editor.init_vertices(mesh_data.num_global_vertices);

  // Iterate over vertices and add to mesh
  for (std::size_t i = 0; i < mesh_data.num_global_vertices; ++i)
  {
    const std::size_t index = mesh_data.vertex_indices[i];
    const std::vector<double> coords(mesh_data.vertex_coordinates[i].begin(),
                                     mesh_data.vertex_coordinates[i].end());
    Point p(mesh_data.gdim, coords.data());
    editor.add_vertex(index, p);
  }

  editor.init_cells(mesh_data.num_global_cells);

  // Iterate over cells and add to mesh
  for (std::size_t i = 0; i < mesh_data.num_global_cells; ++i)
  {
    const std::size_t index = mesh_data.global_cell_indices[i];
    const std::vector<std::size_t> v(mesh_data.cell_vertices[i].begin(), mesh_data.cell_vertices[i].end());
    editor.add_cell(index, v);
  }

  // Close mesh editor
  editor.close();
}
//-----------------------------------------------------------------------------
std::string HDF5File::search_list(const std::vector<std::string>& list,
                                  const std::string& search_term)
{
  std::vector<std::string>::const_iterator it;
  for (it = list.begin(); it != list.end(); ++it)
  {
    if (it->find(search_term) != std::string::npos)
      return *it;
  }
  return std::string("");
}
//-----------------------------------------------------------------------------
bool HDF5File::has_dataset(const std::string dataset_name) const
{
  dolfin_assert(hdf5_file_open);
  return HDF5Interface::has_dataset(hdf5_file_id, dataset_name);
}
//-----------------------------------------------------------------------------
std::vector<double> HDF5File::reorder_vertices_by_global_indices(const Mesh& mesh) const
{
  std::vector<std::size_t> global_size(2);
  global_size[0] = MPI::sum(mesh.num_vertices()); //including duplicates
  global_size[1] = mesh.geometry().dim();

  std::vector<double> ordered_coordinates(mesh.coordinates());
  reorder_values_by_global_indices(mesh, ordered_coordinates, global_size);
  return ordered_coordinates;
}
//---------------------------------------------------------------------------
void HDF5File::reorder_values_by_global_indices(const Mesh& mesh, std::vector<double>& data, 
                                                std::vector<std::size_t>& global_size) const
  {
    Timer t("HDF5: reorder vertex values");
    
    dolfin_assert(global_size.size() == 2);
    dolfin_assert(mesh.num_vertices()*global_size[1] == data.size());
    dolfin_assert(MPI::sum(mesh.num_vertices()) == global_size[0]);

    const std::size_t width = global_size[1];

    // Get shared vertices
    const std::map<unsigned int, std::set<unsigned int> >& shared_vertices
      = mesh.topology().shared_entities(0);

    // My process rank
    const unsigned int my_rank = MPI::process_number();

    // Number of processes
    const unsigned int num_processes = MPI::num_processes();

    // Build list of vertex data to send. Only send shared vertex if I'm the
    // lowest rank process
    std::vector<bool> vertex_sender(mesh.num_vertices(), true);
    std::map<unsigned int, std::set<unsigned int> >::const_iterator it;
    for (it = shared_vertices.begin(); it != shared_vertices.end(); ++it)
    {
      // Check if vertex is shared
      if (!it->second.empty())
      {
        // Check if I am the lowest rank owner
        const std::size_t sharing_min_rank
          = *std::min_element(it->second.begin(), it->second.end());
        if (my_rank > sharing_min_rank)
          vertex_sender[it->first] = false;
      }
    }

    // Global size
    const std::size_t N = mesh.size_global(0);

    // Process offset
    const std::pair<std::size_t, std::size_t> local_range
      = MPI::local_range(N);
    const std::size_t offset = local_range.first;

    // Build buffer of indices and coords to send
    std::vector<std::vector<std::size_t> > send_buffer_index(num_processes);
    std::vector<std::vector<double> > send_buffer_values(num_processes);
    // Reference to data to send, reorganised as a 2D boost::multi_array
    boost::multi_array_ref<double, 2> data_array(data.data(), boost::extents[mesh.num_vertices()][width]);

    for (VertexIterator v(mesh); !v.end(); ++v)
    {
      const std::size_t vidx = v->index();
      if (vertex_sender[vidx])
      {
        std::size_t owner = MPI::index_owner(v->global_index(), N);
        send_buffer_index[owner].push_back(v->global_index());
        send_buffer_values[owner].insert(send_buffer_values[owner].end(),
                                         data_array[vidx].begin(), data_array[vidx].end());
      }
    }

    // Send/receive indices
    std::vector<std::vector<std::size_t> > receive_buffer_index;
    MPI::all_to_all(send_buffer_index, receive_buffer_index);

    // Send/receive coords
    std::vector<std::vector<double> > receive_buffer_values;
    MPI::all_to_all(send_buffer_values, receive_buffer_values);

    // Build vectors of ordered values
    std::vector<double> ordered_values(width*(local_range.second - local_range.first));
    for (std::size_t p = 0; p < receive_buffer_index.size(); ++p)
    {
      for (std::size_t i = 0; i < receive_buffer_index[p].size(); ++i)
      {
        const std::size_t local_index = receive_buffer_index[p][i] - offset;
        for (std::size_t j = 0; j < width; ++j)
        {
          ordered_values[local_index*width + j] = receive_buffer_values[p][i*width + j];
        }
      }
    }

    data.assign(ordered_values.begin(), ordered_values.end());
    global_size[0] = N;
  }
//-----------------------------------------------------------------------------
const std::string HDF5File::cell_type(const std::size_t cell_dim, const Mesh& mesh)
{
  // Get cell type
  CellType::Type _cell_type = mesh.type().cell_type();
  dolfin_assert(cell_dim <= mesh.topology().dim());
  if (cell_dim == mesh.topology().dim())
    _cell_type = mesh.type().cell_type();
  else if (cell_dim == mesh.topology().dim() - 1)
    _cell_type = mesh.type().facet_type();
  else if (cell_dim == 1)
    _cell_type = CellType::interval;
  else if (cell_dim == 0)
    _cell_type = CellType::point;

  // Get cell type string
  return CellType::type2string(_cell_type);
}
//-----------------------------------------------------------------------------

#endif
