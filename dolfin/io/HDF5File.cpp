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
// Last changed: 2012-09-28

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
HDF5File::HDF5File(const std::string filename) : GenericFile(filename, "H5")
{
  // Do nothing

  // FIXME: Create file here in constructor?
  // Not all instatiations of HDF5File create a new file.
  // Could possibly open file descriptor here.
}
//-----------------------------------------------------------------------------
HDF5File::~HDF5File()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void HDF5File::create()
{
  // Create a new file - used by XDMFFile
  HDF5Interface::create(filename);
}
//-----------------------------------------------------------------------------
std::string HDF5File::search_list(std::vector<std::string> &list_of_strings, 
                                  const std::string &search_term) const
{
  // Search through a list of names for a name beginning with search_term
  for(std::vector<std::string>::iterator list_iterator = list_of_strings.begin();
      list_iterator != list_of_strings.end();
      ++list_iterator)
  {
    if(list_iterator->find(search_term) != std::string::npos)
      return *list_iterator;
  }
  return std::string("");
}
//-----------------------------------------------------------------------------
// Mesh input not yet supported
void HDF5File::operator>> (Mesh& input_mesh)
{

  dolfin_error("HDF5File.cpp",
               "read mesh from file",
               "Mesh input is not supported yet");

}

//-----------------------------------------------------------------------------
void HDF5File::operator<< (const Mesh& mesh)
{
  // Mesh output with true global indices - not currently useable for visualisation
  write_mesh(mesh, true);
}
//-----------------------------------------------------------------------------
void HDF5File::write_mesh(const Mesh& mesh, bool true_topology_indices)
{
  // Clear file when writing to file for the first time
  if(counter == 0)
    HDF5Interface::create(filename);
  counter++;

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

  // FIXME: This is a bit clumsy because of lack of good support in DOLFIN
  //        for local/global indices. Replace when support in DOLFIN is
  //        improved
  // Get global vertex indices
  MeshFunction<uint> v_indices(mesh, 0);
  if (MPI::num_processes() == 1)
  {
    for (VertexIterator v(mesh); !v.end(); ++v)
      v_indices[*v] = v->index();
  }
  else
    v_indices = mesh.parallel_data().global_entity_indices(0);

  // Get vertex indices
  std::vector<uint> vertex_indices;
  std::vector<double> vertex_coords;
  vertex_indices.reserve(2*num_local_vertices);
  vertex_coords.reserve(3*num_local_vertices);
  const uint process_number = MPI::process_number();
  for (VertexIterator v(mesh); !v.end(); ++v)
  {
    // Vertex global index and process number
    vertex_indices.push_back(v_indices[*v]);

    // Vertex coordinates
    const Point p = v->point();
    vertex_coords.push_back(p.x());
    vertex_coords.push_back(p.y());
    vertex_coords.push_back(p.z());
  }

  // Write vertex data to HDF5 file if not already there
  const std::string coord_dataset = mesh_coords_dataset_name(mesh);
  if (!dataset_exists(coord_dataset))
  {
    if(true_topology_indices)
    {
      // Reorder local coordinates into global order and distribute
      // FIXME: optimise this section

      std::vector<std::vector<double> > global_vertex_coords;
      std::vector<std::vector<double> > local_vertex_coords;
      for(std::vector<double>::iterator i = vertex_coords.begin(); i != vertex_coords.end(); i += 3)
        local_vertex_coords.push_back(std::vector<double>(i,i+3));

      redistribute_by_global_index(vertex_indices,
                                   local_vertex_coords,
                                   global_vertex_coords);

      vertex_coords.clear();
      vertex_coords.reserve(global_vertex_coords.size()*3);
      for(std::vector<std::vector<double> >::iterator i = global_vertex_coords.begin(); i != global_vertex_coords.end(); i++)
      {
        const std::vector<double>&v = *i;
        vertex_coords.push_back(v[0]);
        vertex_coords.push_back(v[1]);
        vertex_coords.push_back(v[2]);
      }

      // Write out coordinates - no need for GlobalIndex map
      write(coord_dataset, vertex_coords, 3);

      // Write partitions as an attribute
      uint new_vertex_offset = MPI::global_offset(global_vertex_coords.size(),true);
      std::vector<uint> partitions;
      MPI::gather(new_vertex_offset, partitions);
      MPI::broadcast(partitions);
      HDF5Interface::add_attribute(filename, coord_dataset, "partition", partitions);

    }
    else
    {
      // Write coordinates contiguously from each process
      write(coord_dataset, vertex_coords, 3);
      // Write GlobalIndex mapping of coordinates to global vector position
      write(mesh_index_dataset_name(mesh), vertex_indices, 1);
      
      // Write partitions as an attribute
      std::vector<uint> partitions;
      MPI::gather(vertex_offset, partitions);
      MPI::broadcast(partitions);
      HDF5Interface::add_attribute(filename, coord_dataset, "partition", partitions);
    }
    
    uint indexing_indicator = (true_topology_indices ? 1 : 0);
    HDF5Interface::add_attribute(filename, coord_dataset, "true_indexing", indexing_indicator);
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
        topological_data.push_back(v_indices[*v]);
  }
  else
  {
    // Build connectivity using contiguous vertex indices
    for (CellIterator cell(mesh); !cell.end(); ++cell)
      for (VertexIterator v(*cell); !v.end(); ++v)
        topological_data.push_back(v->index() + vertex_range.first);
  }

  // Write connectivity to HDF5 file if not already there
  const std::string topology_dataset = mesh_topology_dataset_name(mesh);
  if (!dataset_exists(topology_dataset))
  {
    write(topology_dataset, topological_data, cell_dim + 1);
    uint indexing_indicator = (true_topology_indices ? 1 : 0);
    HDF5Interface::add_attribute(filename, topology_dataset, "true_indexing", indexing_indicator);
    HDF5Interface::add_attribute(filename, topology_dataset, "celltype", cell_type);

    // Write partitions as an attribute
    std::vector<uint> partitions;
    MPI::gather(cell_offset, partitions);
    MPI::broadcast(partitions);
    HDF5Interface::add_attribute(filename, topology_dataset, "partition", partitions);
  }
}
//-----------------------------------------------------------------------------
void HDF5File::operator<< (const GenericVector& x)
{
  // Get local range;
  std::pair<uint, uint> range = x.local_range(0);

  // Get all local data
  std::vector<double> data;
  x.get_local(data);

  // Overwrite any existing file
  if (counter == 0)
    HDF5Interface::create(filename);

  // Write to HDF5 file
  const std::string name = "/Vector/" + boost::lexical_cast<std::string>(counter);
  write(name.c_str(),data, 1);

  // Increment counter
  counter++;
}
//-----------------------------------------------------------------------------
void HDF5File::operator>> (GenericVector& input)
{
  // Read vector from file, assuming partitioning is already known.
  // FIXME: should abort if not input is not allocated
  const std::pair<uint, uint> range = input.local_range(0);
  std::vector<double> data(range.second - range.first);
  HDF5Interface::read(filename, "/Vector/0", data, range, 1);
  input.set_local(data);
}
//-----------------------------------------------------------------------------
// Write data contiguously from each process in parallel into a 2D array
// data contains local portion of data vector
// width is the second dimension of the array (e.g. 3 for xyz data)
// data in XYZXYZXYZ order
template void HDF5File::write(const std::string dataset_name,
                              const std::vector<int>& data,
                              const uint width);

template void HDF5File::write(const std::string dataset_name,
                              const std::vector<uint>& data,
                              const uint width);

template void HDF5File::write(const std::string dataset_name,
                              const std::vector<double>& data,
                              const uint width);

template <typename T>
void HDF5File::write(const std::string dataset_name,
                     const std::vector<T>& data,
                     const uint width)
{
  // Checks on width and size of data
  dolfin_assert(width != 0);
  uint num_items = data.size()/width;
  dolfin_assert( data.size() == num_items*width);
  
  uint offset = MPI::global_offset(num_items,true);
  std::pair<uint,uint> range(offset, offset + num_items);
  HDF5Interface::write(filename, dataset_name, data, range, width);
}
//-----------------------------------------------------------------------------
bool HDF5File::dataset_exists(const std::string &dataset_name)
{
  // Check for existence of dataset - used by XDMFFile
  return HDF5Interface::dataset_exists(filename, dataset_name);
}

//-----------------------------------------------------------------------------
// Work out the names to use for topology and coordinate datasets
// These routines need MPI to work out the hash
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

// Redistribute a vector according to the global index
// global_index contains the global index values
// local_vector contains the items to be redistributed
// global_vector is the result: the local part of the new global vector created.
template <typename T>
void HDF5File::redistribute_by_global_index(const std::vector<uint>& global_index,
                                            const std::vector<T>& local_vector,
                                            std::vector<T>& global_vector)
{

  dolfin_assert(local_vector.size() == global_index.size());

  uint num_processes = MPI::num_processes();

  // Calculate size of overall global vector by finding max index value anywhere
  uint global_vector_size  = MPI::max(*std::max_element(global_index.begin(),global_index.end())) + 1;

  // Divide up the global vector into local chunks and distribute the partitioning information
  std::pair<uint, uint> range = MPI::local_range(global_vector_size);
  std::vector<uint> partitions;
  MPI::gather(range.first, partitions);
  MPI::broadcast(partitions);
  partitions.push_back(global_vector_size); // add end of last partition 
  
  // Go through each remote process number, finding local values with a global index
  // in the remote partition range, and add to a list.
  std::vector<std::vector<std::pair<uint,T> > > values_to_send(num_processes);
  //  values_to_send.reserve(num_processes);
  std::vector<uint> destinations;
  destinations.reserve(num_processes);

  // Set up destination vector for communication with remote processes
  for(uint process_j = 0; process_j < num_processes ; ++process_j)
  {
    destinations.push_back(process_j);  
    //    std::vector<std::pair<uint,T> > send_to_process_j;
    //    values_to_send.push_back(send_to_process_j);
  }
  
  // Go through local vector and append value to the appropriate list to send to correct process
  for(uint i = 0; i < local_vector.size() ; ++i)
  {
    uint global_i = global_index[i];
    // Identify process which needs this value, by searching through partitioning
    uint process_i = (uint)(std::upper_bound(partitions.begin(),partitions.end(),global_i)-partitions.begin()) - 1;

    if(global_i >= partitions[process_i] && global_i < partitions[process_i+1])
    {
      // send the global index along with the value
      values_to_send[process_i].push_back(make_pair(global_i,local_vector[i]));
    }
    else
    {
      dolfin_error("HDF5File.cpp",
                   "work out which process to send data to",
                   "This should not happen");
    }

  }
  
  // redistribute the values to the appropriate process
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
      uint global_i = received_global_data[j].first;

      if(global_i >= range.first && global_i < range.second)
      {
        global_vector[global_i - range.first] = received_global_data[j].second;
      }
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
