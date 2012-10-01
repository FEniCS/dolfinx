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
// Last changed: 2012-10-01

#ifdef HAS_HDF5

#include <cstdio>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>

#include <dolfin/common/types.h>
#include <dolfin/common/constants.h>
#include <dolfin/common/MPI.h>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/function/Function.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/LocalMeshData.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshPartitioning.h>
#include <dolfin/mesh/MeshEntityIterator.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/Vertex.h>

#include "HDF5File.h"
#include "HDF5Interface.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
HDF5File::HDF5File(const std::string filename, const bool use_mpiio)
  : GenericFile(filename, "H5"),
    hdf5_file_open(false), hdf5_file_id(0),
    mpi_io(MPI::num_processes() > 1 && use_mpiio ? true : false)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
HDF5File::~HDF5File()
{
  // Close HDF5 file
  if (hdf5_file_open)
  {
    herr_t status = H5Fclose(hdf5_file_id);
    dolfin_assert(status != HDF5_FAIL);
  }
}
//-----------------------------------------------------------------------------
void HDF5File::operator<< (const GenericVector& x)
{
  dolfin_assert(x.size() > 0);

  // Open file on first write and add Vector group (overwrite any existing file)
  if (counter == 0)
  {
    // Open file
    dolfin_assert(!hdf5_file_open);
    hdf5_file_id = HDF5Interface::open_file(filename, true, mpi_io);
    hdf5_file_open = true;

    // Add group
    HDF5Interface::add_group(hdf5_file_id, "/Vector");
  }
  dolfin_assert(HDF5Interface::has_group(hdf5_file_id, "/Vector"));

  // Get all local data
  std::vector<double> local_data;
  x.get_local(local_data);

  // Form HDF5 dataset tag
  const std::string dataset_name
      = "/Vector/" + boost::lexical_cast<std::string>(counter);

  // Write data to file
  std::pair<uint,uint> local_range = x.local_range();
  bool chunking = true;
  HDF5Interface::write_dataset(hdf5_file_id, dataset_name, local_data,
                               local_range, 1, mpi_io, chunking);

  // Add partitioning attribute to dataset
  std::vector<uint> partitions;
  MPI::gather(local_range.first, partitions);
  MPI::broadcast(partitions);
        
  HDF5Interface::add_attribute(hdf5_file_id, dataset_name, "partition",
                               partitions);
  
  // Increment counter
  counter++;
}
//-----------------------------------------------------------------------------
void HDF5File::operator>> (GenericVector& x)
{
  // Open file
  if (!hdf5_file_open)
  {
    dolfin_assert(!hdf5_file_open);
    hdf5_file_id = HDF5Interface::open_file(filename, false, mpi_io);
    hdf5_file_open = true;
  }

  // Check that 'Vector' group exists
  dolfin_assert(HDF5Interface::has_group(hdf5_file_id, "/Vector") == 1);

  /*
  // Check that there is only one dataset in group 'Vector'
  dolfin_assert(HDF5Interface::num_datasets_in_group(hdf5_file_id, "/Vector") == 1);
  */

  // Get list all datasets in group
  const std::vector<std::string> datasets
      = HDF5Interface::dataset_list(hdf5_file_id, "/Vector");

  // Make sure there is only one dataset
  dolfin_assert(datasets.size() == 1);

  // Read data set
  read("/Vector/" + datasets[0], x);
}
//-----------------------------------------------------------------------------
void HDF5File::read(const std::string dataset_name, GenericVector& x)
{
  // Open HDF5 file
  if (!hdf5_file_open)
  {
    // Open file
    dolfin_assert(!hdf5_file_open);
    hdf5_file_id = HDF5Interface::open_file(filename, false, mpi_io);
    hdf5_file_open = true;
  }
  dolfin_assert(HDF5Interface::has_group(hdf5_file_id, dataset_name));

  // Get dataset rank
  const uint rank = HDF5Interface::dataset_rank(hdf5_file_id, dataset_name);
  dolfin_assert(rank == 1);

  // Get global dataset size
  const std::vector<uint> data_size
      = HDF5Interface::get_dataset_size(hdf5_file_id, dataset_name);
  dolfin_assert(data_size.size() == 1);

  // Check input vector, and re-size if not already sized
  if (x.size() == 0)
    x.resize(data_size[0]);
  else if (x.size() != data_size[0])
  {
    dolfin_error("HDF5File.cpp",
                 "read vector from file",
                 "Size mis-match between vector in file and input vector");
  }

  // Get local range
  const std::pair<uint, uint> local_range = x.local_range();

  // Read data
  std::vector<double> data;
  HDF5Interface::read_dataset(hdf5_file_id, dataset_name, local_range, data);

  // Set data
  x.set_local(data);
}
//-----------------------------------------------------------------------------
std::string HDF5File::search_list(const std::vector<std::string>& list,
                                  const std::string& search_term) const
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
void HDF5File::operator>> (Mesh& input_mesh)
{
  dolfin_error("HDF5File.cpp",
               "read mesh from file",
               "Mesh input is not supported yet");
}
//-----------------------------------------------------------------------------
void HDF5File::operator<< (const Mesh& mesh)
{
  // Mesh output with true global indices - not currently useable for
  // visualisation
  write_mesh(mesh, true);
}
//-----------------------------------------------------------------------------
void HDF5File::create()
{
  // Create new new HDF5 file (used by XDMFFile)
  HDF5Interface::create(filename, mpi_io);
}
//-----------------------------------------------------------------------------
void HDF5File::write_mesh(const Mesh& mesh, bool true_topology_indices)
{
  // Clear file when writing to file for the first time
  cout << "Create file in HDF5File::write_mesh" << endl;
  if(counter == 0)
    HDF5Interface::create(filename, mpi_io);
  counter++;

  cout << "End create file in HDF5File::write_mesh" << endl;

  // Get local mesh data
  const uint cell_dim = mesh.topology().dim();
  const uint num_local_cells = mesh.num_cells();
  const uint num_local_vertices = mesh.num_vertices();
  const CellType::Type _cell_type = mesh.type().cell_type();
  const std::string cell_type = CellType::type2string(_cell_type);

  // Get cell offset and local cell range
  const uint cell_offset = MPI::global_offset(num_local_cells, true);
  const std::pair<uint, uint> cell_range(cell_offset, cell_offset + num_local_cells);

  // Get vertex offset and local vertex range
  const uint vertex_offset = MPI::global_offset(num_local_vertices, true);
  const std::pair<uint, uint> vertex_range(vertex_offset, vertex_offset + num_local_vertices);

  // Get vertex indices
  std::vector<uint> vertex_indices;
  std::vector<double> vertex_coords;
  vertex_indices.reserve(2*num_local_vertices);
  vertex_coords.reserve(3*num_local_vertices);
  for (VertexIterator v(mesh); !v.end(); ++v)
  {
    // Vertex global index and process number
    vertex_indices.push_back(v->global_index());

    // Vertex coordinates
    const Point p = v->point();
    vertex_coords.push_back(p.x());
    vertex_coords.push_back(p.y());
    vertex_coords.push_back(p.z());
  }

  // Write vertex data to HDF5 file if not already there
  const std::string coord_dataset = mesh_coords_dataset_name(mesh);
  if (!HDF5Interface::dataset_exists(*this, coord_dataset, mpi_io))
  {
    if(true_topology_indices)
    {
      // Reorder local coordinates into global order and distribute
      // FIXME: optimise this section

      std::vector<std::vector<double> > global_vertex_coords;
      std::vector<std::vector<double> > local_vertex_coords;
      for(std::vector<double>::iterator i = vertex_coords.begin(); i != vertex_coords.end(); i += 3)
        local_vertex_coords.push_back(std::vector<double>(i, i + 3));

      redistribute_by_global_index(vertex_indices, local_vertex_coords,
                                   global_vertex_coords);

      vertex_coords.clear();
      vertex_coords.reserve(global_vertex_coords.size()*3);
      for(std::vector<std::vector<double> >::iterator i = global_vertex_coords.begin();
            i != global_vertex_coords.end(); i++)
      {
        const std::vector<double>&v = *i;
        vertex_coords.push_back(v[0]);
        vertex_coords.push_back(v[1]);
        vertex_coords.push_back(v[2]);
      }

      // Write out coordinates - no need for GlobalIndex map
      write_data(coord_dataset, vertex_coords, 3);

      // Write partitions as an attribute
      const uint new_vertex_offset = MPI::global_offset(global_vertex_coords.size(),
                                                        true);
      std::vector<uint> partitions;
      MPI::gather(new_vertex_offset, partitions);
      MPI::broadcast(partitions);
      HDF5Interface::add_attribute(filename, coord_dataset, "partition",
                                   partitions, mpi_io);
    }
    else
    {
      // Write coordinates contiguously from each process
      write_data(coord_dataset, vertex_coords, 3);

      // Write GlobalIndex mapping of coordinates to global vector position
      write_data(mesh_index_dataset_name(mesh), vertex_indices, 1);

      // Write partitions as an attribute
      std::vector<uint> partitions;
      MPI::gather(vertex_offset, partitions);
      MPI::broadcast(partitions);
      HDF5Interface::add_attribute(filename, coord_dataset, "partition",
                                   partitions, mpi_io);
    }

    const uint indexing_indicator = (true_topology_indices ? 1 : 0);
    HDF5Interface::add_attribute(filename, coord_dataset, "true_indexing",
                                 indexing_indicator, mpi_io);
  }

  // Get cell connectivity
  // NOTE: For visualisation via XDMF, the vertex indices correspond
  //       to the local vertex position, and not the true vertex indices.
  std::vector<uint> topological_data;
  if (true_topology_indices)
  {
    // Build connectivity using true vertex indices
    for (CellIterator cell(mesh); !cell.end(); ++cell)
      for (VertexIterator v(*cell); !v.end(); ++v)
        topological_data.push_back(v->global_index());
  }
  else
  {
    // FIXME: No guarantee that local numbering will be contiguous
    // Build connectivity using contiguous local vertex indices
    for (CellIterator cell(mesh); !cell.end(); ++cell)
      for (VertexIterator v(*cell); !v.end(); ++v)
        topological_data.push_back(v->index() + vertex_range.first);
  }

  // Write connectivity to HDF5 file if not already there
  const std::string topology_dataset = mesh_topology_dataset_name(mesh);
  if (!HDF5Interface::dataset_exists(*this, topology_dataset, mpi_io))
  {
    write_data(topology_dataset, topological_data, cell_dim + 1);
    const uint indexing_indicator = (true_topology_indices ? 1 : 0);
    HDF5Interface::add_attribute(filename, topology_dataset, "true_indexing",
                                 indexing_indicator, mpi_io);
    HDF5Interface::add_attribute(filename, topology_dataset, "celltype",
                                 cell_type, mpi_io);

    // Write partitions as an attribute
    std::vector<uint> partitions;
    MPI::gather(cell_offset, partitions);
    MPI::broadcast(partitions);
    HDF5Interface::add_attribute(filename, topology_dataset, "partition",
                                 partitions, mpi_io);
  }
}
//-----------------------------------------------------------------------------
bool HDF5File::dataset_exists(const std::string dataset_name) const
{
  return HDF5Interface::dataset_exists(*this, dataset_name, mpi_io);
}
//-----------------------------------------------------------------------------
std::string HDF5File::mesh_coords_dataset_name(const Mesh& mesh) const
{
  std::stringstream dataset_name;
  dataset_name << "/Mesh/Coordinates_" << std::setfill('0')
          << std::hex << std::setw(8) << mesh.coordinates_hash();
  return dataset_name.str();
}
//-----------------------------------------------------------------------------
std::string HDF5File::mesh_index_dataset_name(const Mesh& mesh) const
{
  std::stringstream dataset_name;
  dataset_name << "/Mesh/GlobalIndex_" << std::setfill('0')
          << std::hex << std::setw(8) << mesh.coordinates_hash();
  return dataset_name.str();
}
//-----------------------------------------------------------------------------
std::string HDF5File::mesh_topology_dataset_name(const Mesh& mesh) const
{
  std::stringstream dataset_name;
  dataset_name << "/Mesh/Topology_" << std::setfill('0')
          << std::hex << std::setw(8) << mesh.topology_hash();
  return dataset_name.str();
}
//-----------------------------------------------------------------------------
template <typename T>
void HDF5File::redistribute_by_global_index(const std::vector<uint>& global_index,
                                            const std::vector<T>& local_vector,
                                            std::vector<T>& global_vector)
{
  dolfin_assert(local_vector.size() == global_index.size());

  // Get number of processes
  const uint num_processes = MPI::num_processes();

  // Calculate size of overall global vector by finding max index value
  // anywhere
  const uint global_vector_size
    = MPI::max(*std::max_element(global_index.begin(), global_index.end())) + 1;

  // Divide up the global vector into local chunks and distribute the
  // partitioning information
  std::pair<uint, uint> range = MPI::local_range(global_vector_size);
  std::vector<uint> partitions;
  MPI::gather(range.first, partitions);
  MPI::broadcast(partitions);
  partitions.push_back(global_vector_size); // add end of last partition

  // Go through each remote process number, finding local values with
  // a global index in the remote partition range, and add to a list.
  std::vector<std::vector<std::pair<uint,T> > > values_to_send(num_processes);
  std::vector<uint> destinations;
  destinations.reserve(num_processes);

  // Set up destination vector for communication with remote processes
  for(uint process_j = 0; process_j < num_processes ; ++process_j)
    destinations.push_back(process_j);

  // Go through local vector and append value to the appropriate list
  // to send to correct process
  for(uint i = 0; i < local_vector.size() ; ++i)
  {
    const uint global_i = global_index[i];

    // Identify process which needs this value, by searching through
    // partitioning
    const uint process_i
       = (uint)(std::upper_bound(partitions.begin(), partitions.end(), global_i) - partitions.begin()) - 1;

    if(global_i >= partitions[process_i] && global_i < partitions[process_i + 1])
    {
      // Send the global index along with the value
      values_to_send[process_i].push_back(make_pair(global_i,local_vector[i]));
    }
    else
    {
      dolfin_error("HDF5File.cpp",
                   "work out which process to send data to",
                   "This should not happen");
    }
  }

  // Redistribute the values to the appropriate process
  std::vector<std::vector<std::pair<uint,T> > > received_values;
  MPI::distribute(values_to_send, destinations, received_values);

  // When receiving, just go through all received values
  // and place them in global_vector, which is the local
  // partition of the global vector.
  global_vector.resize(range.second - range.first);
  for(uint i = 0; i < received_values.size(); ++i)
  {
    const std::vector<std::pair<uint, T> >& received_global_data = received_values[i];
    for(uint j = 0; j < received_global_data.size(); ++j)
    {
      const uint global_i = received_global_data[j].first;
      if(global_i >= range.first && global_i < range.second)
        global_vector[global_i - range.first] = received_global_data[j].second;
      else
      {
        dolfin_error("HDF5File.cpp",
                     "unpack values in vector redistribution",
                     "This should not happen");
      }
    }
  }
}
//-----------------------------------------------------------------------------

#endif
