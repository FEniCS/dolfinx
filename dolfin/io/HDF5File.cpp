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
// Last changed: 2012-10-20

#ifdef HAS_HDF5

#include <cstdio>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/assign.hpp>

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
  : GenericFile(filename, "H5"), hdf5_file_open(false), hdf5_file_id(0),
    mpi_io(MPI::num_processes() > 1 && use_mpiio ? true : false)
{

  // Add parameter to save GlobalIndex (not required for visualisation meshes
  // but needed to make the mesh intelligible for re-reading into dolfin)
  std::set<std::string> index_modes =  boost::assign::list_of("true")("false")("auto");
  parameters.add("global_topology_indexing", "auto");

  // Chunking seems to improve performance generally, option to turn it off.
  parameters.add("chunking", true);

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
  const bool chunking = parameters["hdf5_chunking"];
  const std::vector<uint> global_size(1, x.size());
  HDF5Interface::write_dataset(hdf5_file_id, dataset_name, local_data,
                               local_range, global_size, mpi_io, chunking);

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

  // Get list all datasets in group
  const std::vector<std::string> datasets
      = HDF5Interface::dataset_list(hdf5_file_id, "/Vector");

  // Make sure there is only one dataset
  dolfin_assert(datasets.size() == 1);

  // Read data set
  read("/Vector/" + datasets[0], x);
}
//-----------------------------------------------------------------------------
void HDF5File::read(const std::string dataset_name, GenericVector& x,
                    const bool use_partition_from_file)
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

  // Check that rank is 1
  dolfin_assert(data_size.size() == 1);

  // Check input vector, and re-size if not already sized
  if (x.size() == 0)
  {
    // Resize vector
    if (use_partition_from_file)
    {
      // Get partition from file
      std::vector<uint> partitions;
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
      const uint process_num = MPI::process_number();
      const std::pair<uint, uint> local_range(partitions[process_num], partitions[process_num + 1]);
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
  const std::pair<uint, uint> local_range = x.local_range();

  // Read data from file
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
  //  remove_duplicate_vertices(input_mesh); 
  read_mesh(Mesh &input_mesh);
  
}

void HDF5File::read_mesh(Mesh &input_mesh)
{
    
  warning("HDF5 Mesh input is still experimental");
  warning("HDF5 Mesh input will always repartition the mesh");

  // FIXME: this works, but is not thoroughly checked
  // or optimised in any way

  // Find all datasets needed, and check parameters, then
  // call the appropriate read function.

  // Open file
  if (!hdf5_file_open)
  {
    dolfin_assert(!hdf5_file_open);
    hdf5_file_id = HDF5Interface::open_file(filename, false, mpi_io);
    hdf5_file_open = true;
  }

  // Get list of all datasets in the /Mesh group
  std::vector<std::string> _dataset_list = HDF5Interface::dataset_list(hdf5_file_id, "/Mesh");

  // Look for Topology dataset
  std::string topology_name=search_list(_dataset_list,"Topology");
  if(topology_name.size()==0)
  {
    dolfin_error("HDF5File.cpp",
                 "read topology dataset",
                 "Dataset not found");
  }
  topology_name = "/Mesh/" + topology_name;

  // Look for Coordinates dataset
  std::string coordinates_name=search_list(_dataset_list,"Coordinates");
  if(coordinates_name.size()==0)
  {
    dolfin_error("HDF5File.cpp",
                 "read coordinates dataset",
                 "Dataset not found");
  }
  coordinates_name = "/Mesh/" + coordinates_name;

  // Look for GlobalIndex dataset - mapping local vertices to global order
  std::string global_index_name = search_list(_dataset_list, "GlobalIndex");
  if(global_index_name.size() == 0)
  {
    dolfin_error("HDF5File.cpp",
                 "read mesh from file",
                 "Cannot find GlobalIndex dataset");
  }
  global_index_name = "/Mesh/" + global_index_name;

  uint global_topology_indexing;
  HDF5Interface::get_attribute(hdf5_file_id, topology_name, "global_indexing", 
                               global_topology_indexing);

  std::vector<uint> partition_data;
  HDF5Interface::get_attribute(hdf5_file_id, topology_name, "partition", partition_data);
  const uint num_partitions = partition_data.size();

  // If written in one partition, it must have global indexing anyway.
  if(num_partitions==1 && global_topology_indexing==0)
  {
    warning("Inconsistent topology index attribute. Ignoring.");
  }
  
  if(global_topology_indexing == 1 || num_partitions == 1 )
  {
    read_mesh_with_global_topology_repartition(input_mesh, coordinates_name, 
                                   topology_name, global_index_name);
  }
  else
  {
    read_mesh_with_local_topology_repartition(input_mesh, coordinates_name, 
                                  topology_name, global_index_name);
  }

}

//-----------------------------------------------------------------------------
void HDF5File::read_mesh_with_local_topology_repartition(Mesh &input_mesh, 
                                    const std::string coordinates_name,
                                    const std::string topology_name,
                                    const std::string global_index_name)
{

  // FIXME:
  // This function is experimental, and not checked or optimised

  warning("HDF5 Mesh read is still experimental");
  warning("HDF5 Mesh read will repartition this mesh");
  warning("HDF5 Mesh read may crash");

  const uint num_processes = MPI::num_processes();
  const uint process_number = MPI::process_number();

  // Divide up the data into blocks aligned with the original
  // partitioning.

  std::vector<uint> topology_partition_data;
  HDF5Interface::get_attribute(hdf5_file_id, topology_name, "partition", 
                               topology_partition_data);
  const uint num_partitions = topology_partition_data.size();
  
  if(num_processes != num_partitions)
  {
    warning("Different number of processes was used to write dataset");
    warning("Efficiency of read will be compromised");
    if(num_processes > num_partitions)
    {
      double pc = 100.0*(double)num_partitions/(double)num_processes;
      warning("Efficiency is: %3.1f%%", pc);
      warning("Try rewriting the mesh with global indexing");
    }
  }

  // Divide up partitions between processes. 
  const uint range_division = num_partitions/num_processes;
  const uint range_remainder = num_partitions%num_processes;
  const uint partition_size = range_division 
    + (process_number < range_remainder ? 1 : 0);
  const uint partition_offset = process_number*range_division 
    + (process_number < range_remainder ? process_number : range_remainder);

  // Get dimensions of topology dataset
  const std::vector<uint> topology_dim = HDF5Interface::get_dataset_size(hdf5_file_id, topology_name);
  // Add extra entry to partitions (end of last partition)
  topology_partition_data.push_back(topology_dim[0]);

  const std::pair<uint,uint>cell_range(topology_partition_data[partition_offset],
                      topology_partition_data[partition_offset + partition_size]);

  std::cout <<  "[" << process_number << "] Partition range:" << partition_offset << "-" << partition_offset+partition_size << " -> [" << cell_range.first << "-" << cell_range.second << "]" << std::endl;

  LocalMeshData mesh_data;
  mesh_data.clear();

  // Set vertices-per-cell from width of array
  const uint num_local_cells = topology_dim[0];
  const uint num_vertices_per_cell = topology_dim[1];
  mesh_data.num_vertices_per_cell = num_vertices_per_cell;
  mesh_data.global_cell_indices.reserve(num_local_cells);
  mesh_data.cell_vertices.reserve(num_local_cells);

  // Read a block of cells
  std::vector<uint> topology_data;
  topology_data.reserve(num_local_cells*num_vertices_per_cell);
  HDF5Interface::read_dataset(hdf5_file_id, topology_name, cell_range, topology_data);

  // --- Coordinates ---

  std::vector<uint> coordinates_partition_data;
  HDF5Interface::get_attribute(hdf5_file_id, coordinates_name, "partition", 
                               coordinates_partition_data);
  const std::vector<uint> coordinates_dim = HDF5Interface::get_dataset_size(hdf5_file_id, coordinates_name);
  // Add extra entry to partitions (end of last partition)
  coordinates_partition_data.push_back(coordinates_dim[0]);

  const std::pair<uint,uint>vertex_range(coordinates_partition_data[partition_offset],
                      coordinates_partition_data[partition_offset + partition_size]);
  const uint num_input_vertices = vertex_range.second - vertex_range.first;
  const uint vertex_dim = coordinates_dim[1];

  // Read vertex data to temporary vector
  // FIXME: should HDF5Interface::read_dataset return vector<vector> for 2D datasets?
  std::vector<double> tmp_vertex_data;
  tmp_vertex_data.reserve(num_input_vertices*vertex_dim);
  HDF5Interface::read_dataset(hdf5_file_id, coordinates_name, vertex_range, tmp_vertex_data);
  // Copy to vector<vector>
  std::vector<std::vector<double> > local_vertex_coordinates;
  local_vertex_coordinates.reserve(num_input_vertices);
  for(uint i = 0; i < num_input_vertices ; ++i)
    local_vertex_coordinates.push_back(std::vector<double>(&tmp_vertex_data[i*vertex_dim],&tmp_vertex_data[(i+1)*vertex_dim]));


  // Read global index from file
  std::vector<uint> global_index_data;
  global_index_data.reserve(num_input_vertices);
  HDF5Interface::read_dataset(hdf5_file_id, global_index_name, vertex_range, global_index_data);

  // --- Remap locally indexed topology to global ---

  // Work through cells, reindexing vertices with global index
  uint cell_index = cell_range.first;
  for(std::vector<uint>::iterator cell_i = topology_data.begin();
      cell_i != topology_data.end(); cell_i += num_vertices_per_cell)
  {
    std::vector<uint> cell;
    mesh_data.global_cell_indices.push_back(cell_index);
    cell_index++;

    for(uint j = 0; j < num_vertices_per_cell; j++)
    {
      uint local_index = *(cell_i + j) - vertex_range.first;
      cell.push_back(global_index_data[local_index]);
    }    
    mesh_data.cell_vertices.push_back(cell);
  }

  // Reorganise the vertices into global order and copy into mesh_data
  redistribute_by_global_index(global_index_data,
                               local_vertex_coordinates, 
                               mesh_data.vertex_coordinates);

  // Duplicates have been elimiated, so num_local_vertices will be different
  // from num_input_vertices
  const uint num_local_vertices = mesh_data.vertex_coordinates.size();  
  mesh_data.num_global_vertices = MPI::sum(num_local_vertices);

  // Fill vertex indices with values 
  mesh_data.vertex_indices.resize(num_local_vertices);
  const uint local_vertex_offset = MPI::global_offset(num_local_vertices, true);
  for(uint i = 0; i < mesh_data.vertex_coordinates.size(); ++i)
    mesh_data.vertex_indices[i] = local_vertex_offset + i;
  
  // Build distributed mesh
  MeshPartitioning::build_distributed_mesh(input_mesh, mesh_data);

}

//-----------------------------------------------------------------------------

void HDF5File::read_mesh_with_global_topology_repartition(Mesh &input_mesh, 
                                    const std::string coordinates_name,
                                    const std::string topology_name,
                                    const std::string global_index_name)
{                                        
  // FIXME:
  // This function is experimental, and not checked or optimised

  warning("HDF5 Mesh read is still experimental");
  warning("HDF5 Mesh read will repartition this mesh");

  // Structure to store local mesh
  LocalMeshData mesh_data;
  mesh_data.clear();

  // --- Topology ---
  // Discover size of topology dataset
  std::vector<uint> topology_dim = HDF5Interface::get_dataset_size(hdf5_file_id, topology_name);

  // Get total number of cells, as number of rows in topology dataset
  const uint num_global_cells = topology_dim[0];
  mesh_data.num_global_cells = num_global_cells;

  // Get dimensionality from number of columns in topology
  // FIXME: not very satisfactory
  mesh_data.gdim = topology_dim[1] - 1; 
  mesh_data.tdim = topology_dim[1] - 1;
  
  // Divide up cells equally between processes
  const std::pair<uint,uint> cell_range = MPI::local_range(num_global_cells);
  const uint num_local_cells = cell_range.second - cell_range.first;
  mesh_data.global_cell_indices.reserve(num_local_cells);
  mesh_data.cell_vertices.reserve(num_local_cells);

  // Set vertices-per-cell from width of array
  const uint num_vertices_per_cell = topology_dim[1];
  mesh_data.num_vertices_per_cell = num_vertices_per_cell;

  // Read a block of cells
  std::vector<uint> topology_data;
  topology_data.reserve(num_local_cells*num_vertices_per_cell);
  HDF5Interface::read_dataset(hdf5_file_id, topology_name, cell_range, topology_data);
  
  // Work through cells
  uint cell_index = cell_range.first;
  for(std::vector<uint>::iterator cell_i = topology_data.begin();
      cell_i != topology_data.end(); cell_i += num_vertices_per_cell)
  {
    std::vector<uint> cell;
    mesh_data.global_cell_indices.push_back(cell_index);
    cell_index++;

    // FIXME: inefficient
    for(uint j = 0; j < num_vertices_per_cell; j++)
      cell.push_back(*(cell_i + j));

    mesh_data.cell_vertices.push_back(cell);
  }

  // --- Coordinates ---
  // Get dimensions of coordinate dataset
  std::vector<uint> coords_dim = HDF5Interface::get_dataset_size(hdf5_file_id, coordinates_name);

  // Divide range into equal blocks for each process
  const std::pair<uint, uint> vertex_range = MPI::local_range(coords_dim[0]);
  const uint num_input_vertices = vertex_range.second - vertex_range.first;
  const uint vertex_dim = coords_dim[1];

  // Read vertex data to temporary vector
  std::vector<double> tmp_vertex_data;
  tmp_vertex_data.reserve(num_input_vertices*vertex_dim);
  HDF5Interface::read_dataset(hdf5_file_id, coordinates_name, vertex_range, tmp_vertex_data);
  // Copy to vector<vector>
  std::vector<std::vector<double> > local_vertex_coordinates;
  for(uint i = 0; i < num_input_vertices ; ++i)
    local_vertex_coordinates.push_back(std::vector<double>(&tmp_vertex_data[i*vertex_dim],&tmp_vertex_data[(i+1)*vertex_dim]));
  
  // Read global index from file
  std::vector<uint> global_index_data;
  global_index_data.reserve(num_input_vertices);
  HDF5Interface::read_dataset(hdf5_file_id, global_index_name, vertex_range, global_index_data);

  // Reorganise the vertices into global order and copy into mesh_data
  // eliminating any duplicate vertices
  redistribute_by_global_index(global_index_data,
                               local_vertex_coordinates, 
                               mesh_data.vertex_coordinates);

  const uint num_local_vertices = mesh_data.vertex_coordinates.size();  
  mesh_data.num_global_vertices = MPI::sum(num_local_vertices);

  // Fill vertex indices with values 
  mesh_data.vertex_indices.resize(num_local_vertices);
  const uint local_vertex_offset = MPI::global_offset(num_local_vertices, true);
  for(uint i = 0; i < mesh_data.vertex_coordinates.size(); ++i)
    mesh_data.vertex_indices[i] = local_vertex_offset + i;
    
  // Build distributed mesh
  MeshPartitioning::build_distributed_mesh(input_mesh, mesh_data);
}

//-----------------------------------------------------------------------------
void HDF5File::operator<< (const Mesh& mesh)
{
  // Parameter determines indexing method used.
  // Global topology indexing cannot be used for visualisation.
  // If parameter is "auto" or "true", then use global_indexing for raw h5 files.
  bool global_topology_indexing = 
    (std::string(parameters["global_topology_indexing"])!="false");

  write_mesh(mesh, mesh.topology().dim(), global_topology_indexing);
}
//-----------------------------------------------------------------------------
void HDF5File::write_mesh(const Mesh& mesh)
{
  // Parameter determines indexing method used.
  // Global topology indexing cannot be used for visualisation.
  // If parameter is "auto" or "false", then do not use global indexing here.  
  bool global_topology_indexing = 
    (std::string(parameters["global_topology_indexing"])=="true");
  
  write_mesh(mesh, mesh.topology().dim(), global_topology_indexing);
}
//-----------------------------------------------------------------------------
void HDF5File::write_mesh(const Mesh& mesh, const uint cell_dim, 
                          const bool global_topology_indexing)
{
  // Clear file when writing to file for the first time
  if(!hdf5_file_open)
  {
    hdf5_file_id = HDF5Interface::open_file(filename, false, mpi_io);
    hdf5_file_open = true;
  }
  counter++;

  // Always save the GlobalIndex dataset
  const bool global_indexing_dataset = true;

  // Create Mesh group in HDF5 file
  if (!HDF5Interface::has_group(hdf5_file_id, "/Mesh"))
    HDF5Interface::add_group(hdf5_file_id, "/Mesh");

  // Get local mesh data
  const uint num_local_cells = mesh.num_cells();
  const uint num_local_vertices = mesh.num_vertices();
  //const CellType::Type _cell_type = mesh.type().cell_type();
  CellType::Type _cell_type = mesh.type().cell_type();
  if (cell_dim == mesh.topology().dim())
    _cell_type = mesh.type().cell_type();
  else if (cell_dim == mesh.topology().dim() - 1)
    _cell_type = mesh.type().facet_type();
  else
  {
    dolfin_error("HDF5File.cpp",
                 "write mesh to file",
                 "Only Mesh for Mesh facets can be written to file");
  }

  const std::string cell_type = CellType::type2string(_cell_type);

  // Get cell offset and local cell range
  const uint cell_offset = MPI::global_offset(num_local_cells, true);
  const std::pair<uint, uint> cell_range(cell_offset, cell_offset + num_local_cells);

  // Get vertex offset and local vertex range
  const uint vertex_offset = MPI::global_offset(num_local_vertices, true);
  const std::pair<uint, uint> vertex_range(vertex_offset, vertex_offset + num_local_vertices);

  const uint gdim = mesh.geometry().dim();
  std::vector<double> vertex_coords;
  vertex_coords.reserve(gdim*num_local_vertices);

  for (VertexIterator v(mesh); !v.end(); ++v)
    for (uint i = 0; i < gdim; ++i)
        vertex_coords.push_back(v->x(i));

  std::vector<uint> vertex_indices;
  if(global_indexing_dataset)
    {
      vertex_indices.reserve(num_local_vertices);
      for (VertexIterator v(mesh); !v.end(); ++v)
        vertex_indices.push_back(v->global_index());
    }

  // Write vertex data to HDF5 file if not already there
  const std::string coord_dataset = mesh_coords_dataset_name(mesh);
  if (!HDF5Interface::has_dataset(hdf5_file_id, coord_dataset))
  {
    // Write coordinates contiguously from each process
    std::vector<uint> global_size(2);
    global_size[0] = MPI::sum(vertex_coords.size()/gdim);
    global_size[1] = gdim;
    write_data("/Mesh", coord_dataset, vertex_coords, global_size);

    // Write GlobalIndex mapping of coordinates to global vector position
    // Without these, the mesh cannot be read back in... optional output
    if(global_indexing_dataset)
    {
      std::vector<uint> global_size_map(1);
      global_size_map[0] = MPI::sum(num_local_vertices);
      write_data("/Mesh", mesh_index_dataset_name(mesh), vertex_indices,
                 global_size_map);
    }

    // Write partitions as an attribute
    std::vector<uint> partitions;
    MPI::gather(vertex_offset, partitions);
    MPI::broadcast(partitions);
    HDF5Interface::add_attribute(hdf5_file_id, coord_dataset, "partition",
                                 partitions);

    const uint indexing_indicator = (global_indexing_dataset ? 1 : 0);
    HDF5Interface::add_attribute(hdf5_file_id, coord_dataset, "global_indexing_dataset",
                                 indexing_indicator);
  }

  // FIXME: No guarantee that local numbering will be contiguous. Is
  //        this a problem?

  // Get cell connectivity (local or global indices)
  std::vector<uint> topological_data;
  topological_data.reserve(num_local_cells*(cell_dim - 1));
  if(global_topology_indexing)
  {
    for (MeshEntityIterator c(mesh, cell_dim); !c.end(); ++c)
      for (VertexIterator v(*c); !v.end(); ++v)
        topological_data.push_back(v->global_index());      
  }
  else
  {
    for (MeshEntityIterator c(mesh, cell_dim); !c.end(); ++c)
      for (VertexIterator v(*c); !v.end(); ++v)
       topological_data.push_back(v->index() + vertex_range.first);
  }

  // Write connectivity to HDF5 file if not already there
  const std::string topology_dataset = mesh_topology_dataset_name(mesh);
  if (!HDF5Interface::has_dataset(hdf5_file_id, topology_dataset))
  {
    std::vector<uint> global_size(2);
    global_size[0] = MPI::sum(topological_data.size()/(cell_dim + 1));
    global_size[1] = cell_dim + 1;
    write_data("/Mesh", topology_dataset, topological_data, global_size);

    const uint indexing_indicator = (global_topology_indexing ? 1 : 0);
    HDF5Interface::add_attribute(hdf5_file_id, topology_dataset,
                                 "global_indexing", indexing_indicator);
    HDF5Interface::add_attribute(hdf5_file_id, topology_dataset, "celltype",
                                 cell_type);

    // Write partitions as an attribute
    std::vector<uint> partitions;
    MPI::gather(cell_offset, partitions);
    MPI::broadcast(partitions);
    HDF5Interface::add_attribute(hdf5_file_id, topology_dataset, "partition",
                                 partitions);
  }
}
//-----------------------------------------------------------------------------
bool HDF5File::dataset_exists(const std::string dataset_name) const
{
  dolfin_assert(hdf5_file_open);
  return HDF5Interface::has_dataset(hdf5_file_id, dataset_name);
}
//-----------------------------------------------------------------------------
void HDF5File::open_hdf5_file(bool truncate)
{
  dolfin_assert(!hdf5_file_open);
  hdf5_file_id = HDF5Interface::open_file(filename, truncate, mpi_io);
  hdf5_file_open = true;
}
//-----------------------------------------------------------------------------
std::string HDF5File::mesh_coords_dataset_name(const Mesh& mesh) const
{
  std::stringstream dataset_name;
  dataset_name << "/Mesh/Coordinates_" << std::setfill('0')
          << std::hex << std::setw(8) << mesh.geometry().hash();
  return dataset_name.str();
}
//-----------------------------------------------------------------------------
std::string HDF5File::mesh_index_dataset_name(const Mesh& mesh) const
{
  std::stringstream dataset_name;
  dataset_name << "/Mesh/GlobalIndex_" << std::setfill('0')
          << std::hex << std::setw(8) << mesh.geometry().hash();
  return dataset_name.str();
}
//-----------------------------------------------------------------------------
std::string HDF5File::mesh_topology_dataset_name(const Mesh& mesh) const
{
  const uint D = mesh.topology().dim();
  std::stringstream dataset_name;
  dataset_name << "/Mesh/Topology_" << std::setfill('0')
          << std::hex << std::setw(8) << mesh.topology()(D, D).hash();
  return dataset_name.str();
}
//-----------------------------------------------------------------------------

void HDF5File::remove_duplicate_vertices(Mesh &mesh)
{

  const uint num_processes = MPI::num_processes();
  const uint process_number = MPI::process_number();
  const uint num_local_vertices = mesh.num_vertices();
  
  const std::map<uint, std::set<uint> >& shared_vertices
    = mesh.topology().shared_entities(0);

  // Create global => local map for all local vertices
  std::map<uint, uint> local;
  for (VertexIterator v(mesh); !v.end(); ++v)
    local[v->global_index()] = v->index();

  // New local indexing after removing duplicates
  // Initialise to '1' and mark removed vertices with '0'
  std::vector<uint> remap(num_local_vertices, 1);

  // Structures for MPI::distribute
  std::vector<uint> destinations(num_processes);
  std::vector<std::vector<std::pair<uint,uint> > > values_to_send(num_processes);

  // Go through shared vertices looking for vertices which are
  // on a lower numbered process. Mark these as being off-process.
  // Meanwhile, create a list of processes to tell about locally owned
  // shared vertices.
  uint count = num_local_vertices;
  for(std::map<uint, std::set<uint> >::const_iterator 
      shared_v_it = shared_vertices.begin();
      shared_v_it != shared_vertices.end();
      shared_v_it++)
  {
    const uint global_index = shared_v_it->first;
    const uint local_index = local[global_index];
    const std::set<uint>& procs = shared_v_it->second;
    // Determine whether this vertex is also on a lower numbered process
    if(*(procs.begin()) < process_number) 
    {
      remap[local_index] = 0;
      count--;
    } 
    else
    {
      for(std::set<uint>::iterator proc = procs.begin(); 
          proc != procs.end(); ++proc)
      {
        std::pair<uint, uint> global_to_local(global_index, local_index);
        values_to_send[*proc].push_back(global_to_local);
      }
    }
  }

  const uint vertex_offset = MPI::global_offset(count, true);
  const std::pair<uint, uint> vertex_range(vertex_offset, vertex_offset + count);

  std::stringstream s;      
  s << "nv=" << MPI::sum(count) << std::endl;  

  // Remap local indices to account for missing vertices
  count = vertex_range.first - 1;
  for(uint i = 0; i < num_local_vertices; i++)
  {
    count += remap[i];
    remap[i] = count;

    s << i << "=" <<  remap[i] << std::endl;
  }

  // Now need to remap shared vertices in values_to_send

  // Redistribute the values to the appropriate process
  std::vector<std::vector<std::pair<uint,uint> > > received_values;
  MPI::distribute(values_to_send, destinations, received_values);

  // flatten and insert global remapping into remap
  for(std::vector<std::vector<std::pair<uint, uint> > >::iterator rv=received_values.begin(); rv != received_values.end(); ++rv)
    for(std::vector<std::pair<uint, uint> >::iterator f=rv->begin(); f != rv->end(); ++f)
      remap[local[f->first]] = f->second;
  
  std::cout << s.str() ;

}

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
