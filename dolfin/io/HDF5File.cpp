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
// Last changed: 2012-11-12

#ifdef HAS_HDF5

#include <cstdio>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/assign.hpp>
#include <boost/multi_array.hpp>
#include <boost/bind.hpp>

#include <dolfin/common/types.h>
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
  // HDF5 chunking
  parameters.add("chunking", false);
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
void HDF5File::flush()
{
  dolfin_assert(hdf5_file_open);
  HDF5Interface::flush_file(hdf5_file_id);
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
  const bool chunking = parameters["chunking"];
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
      const std::pair<uint, uint>
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
  const std::pair<uint, uint> local_range = x.local_range();

  // Read data from file
  std::vector<double> data;
  HDF5Interface::read_dataset(hdf5_file_id, dataset_name, local_range, data);

  // Set data
  x.set_local(data);
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
void HDF5File::operator>> (Mesh& input_mesh)
{
  read_mesh(input_mesh);
}
//-----------------------------------------------------------------------------
void HDF5File::read_mesh(Mesh& input_mesh)
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
  std::vector<std::string> _dataset_list
      = HDF5Interface::dataset_list(hdf5_file_id, "/Mesh");
  
  // Should return a list of groups

  std::cout << _dataset_list.size() << " groups found...\n" ;

  if(_dataset_list.size()==0)
  {
    dolfin_error("HDF5File.cpp",
                 "open Mesh",
                 "Dataset not found");
  }
  
  if(_dataset_list.size()!=1)
  {
    warning("Multiple Mesh datasets found. Using first dataset.");
  }
  std::string mesh_group_name = "/Mesh/" + _dataset_list[0];
  
  _dataset_list = HDF5Interface::dataset_list(hdf5_file_id, mesh_group_name);
      
  // Look for Topology dataset
  std::string topology_name = search_list(_dataset_list,"topology");
  if (topology_name.size() == 0)
  {
    dolfin_error("HDF5File.cpp",
                 "read topology dataset",
                 "Dataset not found");
  }
  topology_name = mesh_group_name + "/" + topology_name;

  // Look for global_index dataset
  std::string global_index_name = search_list(_dataset_list,"global_index");
  if (global_index_name.size() == 0)
  {
    dolfin_error("HDF5File.cpp",
                 "read global index dataset",
                 "Dataset not found");
  }
  global_index_name = mesh_group_name + "/" + global_index_name;

  // Look for Coordinates dataset
  std::string coordinates_name=search_list(_dataset_list,"coordinates");
  if(coordinates_name.size()==0)
  {
    dolfin_error("HDF5File.cpp",
                 "read coordinates dataset",
                 "Dataset not found");
  }
  coordinates_name = mesh_group_name + "/" + coordinates_name;

  read_mesh_repartition(input_mesh, coordinates_name, global_index_name,
                                   topology_name);
}
//-----------------------------------------------------------------------------
void HDF5File::read_mesh_repartition(Mesh& input_mesh,
                                     const std::string coordinates_name,
                                     const std::string global_index_name,
                                     const std::string topology_name)
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
  std::vector<uint> topology_dim
      = HDF5Interface::get_dataset_size(hdf5_file_id, topology_name);

  // Get total number of cells, as number of rows in topology dataset
  const uint num_global_cells = topology_dim[0];
  mesh_data.num_global_cells = num_global_cells;

  // Set vertices-per-cell from width of array
  const uint num_vertices_per_cell = topology_dim[1];
  mesh_data.num_vertices_per_cell = num_vertices_per_cell;

  // Get dimensionality from number of columns in topology
  mesh_data.tdim = topology_dim[1] - 1;

  // Divide up cells ~equally between processes
  const std::pair<uint,uint> cell_range = MPI::local_range(num_global_cells);
  const uint num_local_cells = cell_range.second - cell_range.first;

  // Read a block of cells
  std::vector<uint> topology_data;
  topology_data.reserve(num_local_cells*num_vertices_per_cell);
  mesh_data.cell_vertices.resize(boost::extents[num_local_cells][num_vertices_per_cell]);
  HDF5Interface::read_dataset(hdf5_file_id, topology_name, cell_range, topology_data);

  // Copy to boost::multi_array
  // FIXME: there should be a more efficient way to do this?
  mesh_data.global_cell_indices.reserve(num_local_cells);
  std::vector<uint>::iterator topology_it = topology_data.begin();
  for(uint i = 0; i < num_local_cells; i++)
  {
    mesh_data.global_cell_indices.push_back(cell_range.first + i);
    std::copy(topology_it, topology_it + num_vertices_per_cell,
              mesh_data.cell_vertices[i].begin());
    topology_it += num_vertices_per_cell;
  }

  // --- Coordinates ---
  // Get dimensions of coordinate dataset
  std::vector<uint> coords_dim
    = HDF5Interface::get_dataset_size(hdf5_file_id, coordinates_name);
  mesh_data.num_global_vertices = coords_dim[0];
  const uint vertex_dim = coords_dim[1];
  mesh_data.gdim = vertex_dim;

  // Divide range into equal blocks for each process
  const std::pair<uint, uint> vertex_range = MPI::local_range(coords_dim[0]);
  const uint num_local_vertices = vertex_range.second - vertex_range.first;

  // Read vertex data to temporary vector
  std::vector<double> tmp_vertex_data;
  tmp_vertex_data.reserve(num_local_vertices*vertex_dim);
  HDF5Interface::read_dataset(hdf5_file_id, coordinates_name, vertex_range,
                              tmp_vertex_data);
  // Copy to vector<vector>
  mesh_data.vertex_coordinates.reserve(num_local_vertices);
  for(std::vector<double>::iterator v = tmp_vertex_data.begin();
      v != tmp_vertex_data.end(); v += vertex_dim)
  {
    mesh_data.vertex_coordinates.push_back(std::vector<double>(v, v + vertex_dim));
  }

  // Fill vertex indices with values
  mesh_data.vertex_indices.resize(num_local_vertices);
  HDF5Interface::read_dataset(hdf5_file_id, global_index_name, vertex_range,
                              mesh_data.vertex_indices);

  // Build distributed mesh
  MeshPartitioning::build_distributed_mesh(input_mesh, mesh_data);
}
//-----------------------------------------------------------------------------
void HDF5File::operator<< (const Mesh& mesh)
{
  const std::string name = "/Mesh/" + boost::lexical_cast<std::string>(counter);
  write_mesh_global_index(mesh, mesh.topology().dim(), name);
  counter++;
}
//-----------------------------------------------------------------------------
void HDF5File::write_mesh(const Mesh& mesh, const std::string name)
{
  write_mesh_global_index(mesh, mesh.topology().dim(), name);
}
//-----------------------------------------------------------------------------
void HDF5File::write_mesh_global_index(const Mesh& mesh, uint cell_dim, const std::string name)
{

  warning("Writing mesh with GlobalIndex - not suitable for visualisation");
  
  // Clear file when writing to file for the first time
  if(!hdf5_file_open)
  {
    hdf5_file_id = HDF5Interface::open_file(filename, false, mpi_io);
    hdf5_file_open = true;
  }

  // Create Mesh group in HDF5 file
  if (!HDF5Interface::has_group(hdf5_file_id, "/Mesh"))
    HDF5Interface::add_group(hdf5_file_id, "/Mesh");

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

  // ------ Output vertex coordinates and global index

  // Vertex numbers, ranges and offsets
  const uint num_local_vertices = mesh.num_vertices();
  const uint vertex_offset = MPI::global_offset(num_local_vertices, true);

  // Write vertex data to HDF5 file
  const std::string coord_dataset = name + "/coordinates";
  const std::string index_dataset = name + "/global_index";
  const std::vector<uint>& global_indices = mesh.topology().global_indices(0);

  {
    const uint gdim = mesh.geometry().dim();
    const std::vector<double>& vertex_coords = mesh.coordinates();

    // Write coordinates contiguously from each process
    std::vector<uint> global_size(2);
    global_size[0] = MPI::sum(num_local_vertices);
    global_size[1] = gdim;
    write_data(coord_dataset, vertex_coords, global_size);
    global_size.resize(1); //remove second dimension
    write_data(index_dataset, global_indices, global_size);
  }

  // ------ Topology

  // Get/build topology data
  std::vector<uint> topological_data;
  if (cell_dim == mesh.topology().dim())
  {
    topological_data = mesh.cells();
    std::transform(topological_data.begin(), topological_data.end(),
                   topological_data.begin(),
                   boost::bind<const uint &>(&std::vector<uint>::at, &global_indices, _1));
  }
  else
  {
    topological_data.reserve(mesh.num_cells()*(cell_dim - 1));
    for (MeshEntityIterator c(mesh, cell_dim); !c.end(); ++c)
      for (VertexIterator v(*c); !v.end(); ++v)
        topological_data.push_back(v->global_index());
  }

  // Write topology data
  const std::string topology_dataset = name + "/topology";
  std::vector<uint> global_size(2);
  global_size[0] = MPI::sum(topological_data.size()/(cell_dim + 1));
  global_size[1] = cell_dim + 1;
  write_data(topology_dataset, topological_data, global_size);
  
  HDF5Interface::add_attribute(hdf5_file_id, topology_dataset, "celltype",
                               cell_type);

}
//-----------------------------------------------------------------------------
void HDF5File::write_visualisation_mesh(const Mesh& mesh, const std::string name)
{
  write_visualisation_mesh(mesh, mesh.topology().dim(), name);
}
//-----------------------------------------------------------------------------
void HDF5File::write_visualisation_mesh(const Mesh& mesh, const uint cell_dim,
                          const std::string name)
{
  // Clear file when writing to file for the first time
  if (!hdf5_file_open)
  {
    hdf5_file_id = HDF5Interface::open_file(filename, false, mpi_io);
    hdf5_file_open = true;
  }

  // Create VisualisationMesh group in HDF5 file
  if (!HDF5Interface::has_group(hdf5_file_id, "/VisualisationMesh"))
    HDF5Interface::add_group(hdf5_file_id, "/VisualisationMesh");

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

  // Cell type string
  const std::string cell_type = CellType::type2string(_cell_type);

  // Vertex numbers, ranges and offsets
  const uint num_local_vertices = mesh.num_vertices();
  const uint vertex_offset = MPI::global_offset(num_local_vertices, true);

  // Write vertex data to HDF5 file
  const std::string coord_dataset = name + "/coordinates";
  {
    const uint gdim = mesh.geometry().dim();
    const std::vector<double>& vertex_coords = mesh.coordinates();
    
    // Write coordinates contiguously from each process
    std::vector<uint> global_size(2);
    global_size[0] = MPI::sum(num_local_vertices);
    global_size[1] = gdim;
    write_data(coord_dataset, vertex_coords, global_size);
  }

  // Write connectivity to HDF5 file (using local indices + offset)
  {
    // Get/build topology data
    std::vector<uint> topological_data;
    if (cell_dim == mesh.topology().dim())
    {
      topological_data = mesh.cells();
      std::transform(topological_data.begin(), topological_data.end(),
                     topological_data.begin(),
                     std::bind2nd(std::plus<uint>(), vertex_offset));
    }
    else
    {
      topological_data.reserve(mesh.num_cells()*(cell_dim - 1));
      for (MeshEntityIterator c(mesh, cell_dim); !c.end(); ++c)
        for (VertexIterator v(*c); !v.end(); ++v)
         topological_data.push_back(v->index() + vertex_offset);
    }

    // Write topology data
    const std::string topology_dataset = name + "/topology";
    std::vector<uint> global_size(2);
    global_size[0] = MPI::sum(topological_data.size()/(cell_dim + 1));
    global_size[1] = cell_dim + 1;
    write_data(topology_dataset, topological_data, global_size);

    HDF5Interface::add_attribute(hdf5_file_id, topology_dataset, "celltype",
                                 cell_type);
  }

  counter++;
}
//-----------------------------------------------------------------------------
bool HDF5File::has_dataset(const std::string dataset_name) const
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

#endif
