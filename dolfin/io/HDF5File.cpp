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

#ifdef HAS_HDF5

#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <boost/unordered_map.hpp>
#include <boost/filesystem.hpp>
#include <boost/multi_array.hpp>

#include <dolfin/common/constants.h>
#include <dolfin/common/MPI.h>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/common/Timer.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/LocalMeshData.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshEditor.h>
#include <dolfin/mesh/MeshPartitioning.h>
#include <dolfin/mesh/MeshEntityIterator.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/MeshValueCollection.h>
#include <dolfin/mesh/Vertex.h>
#include "HDF5Attribute.h"
#include "HDF5Interface.h"
#include "HDF5Utility.h"
#include "HDF5File.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
HDF5File::HDF5File(MPI_Comm comm, const std::string filename,
                   const std::string file_mode)
  : _hdf5_file_id(0), _mpi_comm(comm)
{
  // See https://www.hdfgroup.org/hdf5-quest.html#gzero on zero for
  // _hdf5_file_id(0)

  // HDF5 chunking
  parameters.add("chunking", false);

  // Create directory, if required (create on rank 0)
  if (MPI::rank(_mpi_comm) == 0)
  {
    const boost::filesystem::path path(filename);
    if (path.has_parent_path()
        && !boost::filesystem::is_directory(path.parent_path()))
    {
      boost::filesystem::create_directories(path.parent_path());
      if (!boost::filesystem::is_directory(path.parent_path()))
      {
        dolfin_error("HDF5File.cpp",
                     "open file",
                     "Could not create directory \"%s\"",
                     path.parent_path().string().c_str());
      }
    }
  }

  // Wait until directory has been created
  MPI::barrier(_mpi_comm);

  // Open HDF5 file
  const bool mpi_io = MPI::size(_mpi_comm) > 1 ? true : false;
  _hdf5_file_id = HDF5Interface::open_file(_mpi_comm, filename, file_mode,
                                          mpi_io);
  dolfin_assert(_hdf5_file_id > 0);
}
//-----------------------------------------------------------------------------
HDF5File::~HDF5File()
{
  close();
}
//-----------------------------------------------------------------------------
void HDF5File::close()
{
  // Close HDF5 file
  if (_hdf5_file_id > 0)
    HDF5Interface::close_file(_hdf5_file_id);
  _hdf5_file_id = 0;
}
//-----------------------------------------------------------------------------
void HDF5File::flush()
{
  dolfin_assert(_hdf5_file_id > 0);
  HDF5Interface::flush_file(_hdf5_file_id);
}
//-----------------------------------------------------------------------------
void HDF5File::write(const std::vector<Point>& points,
                     const std::string dataset_name)
{
  dolfin_assert(points.size() > 0);
  dolfin_assert(_hdf5_file_id > 0);

  // Get number of points (global)
  std::size_t num_points_global = MPI::sum(_mpi_comm, points.size());

  // Data set name
  const std::string coord_dataset =  dataset_name + "/coordinates";

  // Pack data
  const std::size_t n = points.size();
  std::vector<double> x(3*n);
  for (std::size_t i = 0; i< n; ++i)
    for (std::size_t j = 0; j < 3; ++j)
      x[3*i + j] = points[i][j];

  // Write data to file
  //  const bool chunking = parameters["chunking"];
  std::vector<std::int64_t> global_size(2);
  global_size[0] = num_points_global;
  global_size[1] = 3;

  const bool mpi_io = MPI::size(_mpi_comm) > 1 ? true : false;
  write_data(coord_dataset, x, global_size, mpi_io);
}
//-----------------------------------------------------------------------------
void HDF5File::write(const std::vector<double>& values,
                     const std::string dataset_name)
{
  std::vector<std::int64_t> global_size(1, MPI::sum(_mpi_comm, values.size()));
  const bool mpi_io = MPI::size(_mpi_comm) > 1 ? true : false;
  write_data(dataset_name, values, global_size, mpi_io);
}
//-----------------------------------------------------------------------------
void HDF5File::write(const GenericVector& x, const std::string dataset_name)
{
  dolfin_assert(x.size() > 0);
  dolfin_assert(_hdf5_file_id > 0);

  // Get all local data
  std::vector<double> local_data;
  x.get_local(local_data);

  // Write data to file
  std::pair<std::size_t, std::size_t> local_range = x.local_range();
  const bool chunking = parameters["chunking"];
  const std::vector<std::int64_t> global_size(1, x.size());
  const bool mpi_io = MPI::size(_mpi_comm) > 1 ? true : false;
  HDF5Interface::write_dataset(_hdf5_file_id, dataset_name, local_data,
                               local_range, global_size, mpi_io, chunking);

  // Add partitioning attribute to dataset
  std::vector<std::size_t> partitions;
  std::vector<std::size_t> local_range_first(1, local_range.first);
  MPI::gather(_mpi_comm, local_range_first, partitions);
  MPI::broadcast(_mpi_comm, partitions);

  HDF5Interface::add_attribute(_hdf5_file_id, dataset_name, "partition",
                               partitions);
}
//-----------------------------------------------------------------------------
void HDF5File::read(GenericVector& x, const std::string dataset_name,
                    const bool use_partition_from_file) const
{
  dolfin_assert(_hdf5_file_id > 0);

  // Check for data set exists
  if (!HDF5Interface::has_dataset(_hdf5_file_id, dataset_name))
  {
    dolfin_error("HDF5File.cpp",
                 "read vector from file",
                 "Data set with name \"%s\" does not exist",
                 dataset_name.c_str());
  }

  // Get dataset rank
  const std::size_t rank = HDF5Interface::dataset_rank(_hdf5_file_id,
                                                       dataset_name);

  if (rank != 1)
    warning("Reading non-scalar data in HDF5 Vector");

  // Get global dataset size
  const std::vector<std::int64_t> data_shape
      = HDF5Interface::get_dataset_shape(_hdf5_file_id, dataset_name);

  // Check that rank is 1 or 2
  dolfin_assert(data_shape.size() == 1
                or (data_shape.size() == 2 and data_shape[1] == 1));

  // Check input vector, and re-size if not already sized
  if (x.empty())
  {
    // Initialize vector
    if (use_partition_from_file)
    {
      // Get partition from file
      std::vector<std::size_t> partitions;
      HDF5Interface::get_attribute(_hdf5_file_id, dataset_name, "partition",
                                   partitions);

      // Check that number of MPI processes matches partitioning
      if (MPI::size(_mpi_comm) != partitions.size())
      {
        dolfin_error("HDF5File.cpp",
                     "read vector from file",
                     "Different number of processes used when writing. Cannot restore partitioning");
      }

      // Add global size at end of partition vectors
      partitions.push_back(data_shape[0]);

      // Initialise vector
      const std::size_t process_num = MPI::rank(_mpi_comm);
      const std::pair<std::size_t, std::size_t>
        local_range(partitions[process_num], partitions[process_num + 1]);
      x.init(_mpi_comm, local_range);
    }
    else
      x.init(_mpi_comm, data_shape[0]);
  }
  else if ((std::int64_t) x.size() != data_shape[0])
  {
    dolfin_error("HDF5File.cpp",
                 "read vector from file",
                 "Size mis-match between vector in file and input vector");
  }

  // Get local range
  const std::pair<std::size_t, std::size_t> local_range = x.local_range();

  // Read data from file
  std::vector<double> data;
  HDF5Interface::read_dataset(_hdf5_file_id, dataset_name, local_range, data);

  // Set data
  x.set_local(data);
  x.apply("insert");
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
  // FIXME: break up this function

  Timer t0("HDF5: write mesh to file");

  const std::size_t tdim = mesh.topology().dim();
  const std::size_t gdim = mesh.geometry().dim();

  const bool mpi_io = MPI::size(_mpi_comm) > 1 ? true : false;
  dolfin_assert(_hdf5_file_id > 0);

  CellType::Type cell_type = mesh.type().entity_type(cell_dim);
  std::unique_ptr<CellType> celltype(CellType::create(cell_type));
  std::size_t num_cell_points = 0;
  for (std::size_t i = 0; i <= cell_dim; ++i)
    num_cell_points +=
      mesh.geometry().num_entity_coordinates(i)*celltype->num_entities(i);

  // ---------- Vertices (coordinates)
  {
    // Write vertex data to HDF5 file
    const std::string coord_dataset =  name + "/coordinates";

    // Copy coordinates and indices and remove off-process values
    std::vector<double> vertex_coords;
    if (!mpi_io)
      vertex_coords = mesh.geometry().x();
    else
      vertex_coords
        = DistributedMeshTools::reorder_vertices_by_global_indices(mesh);

    // Write coordinates out from each process
    std::vector<std::int64_t> global_size(2);
    global_size[0] = MPI::sum(_mpi_comm, vertex_coords.size()/gdim);
    global_size[1] = gdim;
    write_data(coord_dataset, vertex_coords, global_size, mpi_io);
  }

  // ---------- Topology
  {
    // Get/build topology data
    std::vector<std::int64_t> topological_data;
    topological_data.reserve(mesh.num_entities(cell_dim)*(num_cell_points));

    const std::vector<std::size_t>& global_vertices
      = mesh.topology().global_indices(0);

    // Permutation to VTK ordering
    const std::vector<unsigned int> perm = celltype->vtk_mapping();

    if (cell_dim == tdim or !mpi_io)
    {
      // Usual case, with cell output, and/or none shared with another
      // process.
      if (mesh.geometry().degree() > 1)
      {
        const MeshGeometry& geom = mesh.geometry();

        // Only cope with quadratic for now
        dolfin_assert(geom.degree() == 2);
        // FIXME: make it work in parallel
        dolfin_assert(!mpi_io);

        std::vector<std::size_t> edge_mapping;
        if (tdim == 1)
          edge_mapping = {0};
        else if (tdim == 2)
          edge_mapping = {2, 0, 1};
        else
          edge_mapping = {5, 2, 4, 3, 1, 0};

        for (CellIterator c(mesh); !c.end(); ++c)
        {
          // Add indices for vertices and edges
          for (unsigned int dim = 0; dim != 2; ++dim)
          {
            for (unsigned int i = 0; i != celltype->num_entities(dim); ++i)
            {
              std::size_t im = (dim == 0) ? i : edge_mapping[i];
              const std::size_t entity_index
                = (dim == tdim) ? c->index() : c->entities(dim)[im];
              const std::size_t local_idx
                = geom.get_entity_index(dim, 0, entity_index);
              topological_data.push_back(local_idx);
            }
          }
        }
      }
      else if (cell_dim == 0)
      {
        for (VertexIterator v(mesh); !v.end(); ++v)
          topological_data.push_back(v->global_index());
      }
      else
      {
        for (MeshEntityIterator c(mesh, cell_dim); !c.end(); ++c)
          for (unsigned int i = 0; i != c->num_entities(0); ++i)
          {
            const unsigned int local_idx = c->entities(0)[perm[i]];
            topological_data.push_back(global_vertices[local_idx]);
          }
      }
    }
    else
    {
      // Drop duplicate topology for shared entities of less than mesh
      // dimension

      // If not already numbered, number entities of order cell_dim so
      // we can get shared_entities
      DistributedMeshTools::number_entities(mesh, cell_dim);

      const std::size_t mpi_rank = MPI::rank(_mpi_comm);
      const std::map<unsigned int, std::set<unsigned int>>& shared_entities
        = mesh.topology().shared_entities(cell_dim);

      std::set<unsigned int> non_local_entities;
      if (mesh.topology().size(tdim) == mesh.topology().ghost_offset(tdim))
      {
        // No ghost cells - exclude shared entities which are on lower
        // rank processes
        for (auto sh = shared_entities.begin(); sh != shared_entities.end(); ++sh)
        {
          const unsigned int lowest_proc = *(sh->second.begin());
          if (lowest_proc < mpi_rank)
            non_local_entities.insert(sh->first);
        }
      }
      else
      {
        // Iterate through ghost cells, adding non-ghost entities
        // which are in lower rank process cells to a set for
        // exclusion from output
        for (MeshEntityIterator c(mesh, tdim, "ghost"); !c.end(); ++c)
        {
          const unsigned int cell_owner = c->owner();
          for (MeshEntityIterator ent(*c, cell_dim); !ent.end(); ++ent)
            if (!ent->is_ghost() && cell_owner < mpi_rank)
                non_local_entities.insert(ent->index());
        }
      }

      if (cell_dim == 0)
      {
        // Special case for mesh of points
        for (VertexIterator v(mesh); !v.end(); ++v)
        {
          if (non_local_entities.find(v->index())
              == non_local_entities.end())
            topological_data.push_back(v->global_index());
        }
      }
      else
      {
        for (MeshEntityIterator ent(mesh, cell_dim); !ent.end(); ++ent)
        {
          // If not excluded, add to topology
          if (non_local_entities.find(ent->index())
              == non_local_entities.end())
          {
            for (unsigned int i = 0; i != ent->num_entities(0); ++i)
            {
              const unsigned int local_idx = ent->entities(0)[perm[i]];
              topological_data.push_back(global_vertices[local_idx]);
            }
          }
        }
      }
    }

    // Write topology data
    const std::string topology_dataset =  name + "/topology";
    std::vector<std::int64_t> global_size(2);
    global_size[0] = MPI::sum(_mpi_comm,
                              topological_data.size()/num_cell_points);
    global_size[1] = num_cell_points;
    dolfin_assert(global_size[0] == (std::int64_t) mesh.size_global(cell_dim));
    const bool mpi_io = MPI::size(_mpi_comm) > 1 ? true : false;
    write_data(topology_dataset, topological_data, global_size, mpi_io);

    // For cells, write the global cell index
    if (cell_dim == mesh.topology().dim())
    {
      const std::string cell_index_dataset = name + "/cell_indices";
      global_size.pop_back();
      const std::vector<std::size_t>& cell_index_ref
        = mesh.topology().global_indices(cell_dim);
      const std::vector<std::size_t> cells(cell_index_ref.begin(),
            cell_index_ref.begin() + mesh.topology().ghost_offset(cell_dim));
      const bool mpi_io = MPI::size(_mpi_comm) > 1 ? true : false;
      write_data(cell_index_dataset, cells, global_size, mpi_io);
    }

    // Add cell type attribute
    HDF5Interface::add_attribute(_hdf5_file_id, topology_dataset, "celltype",
                                 CellType::type2string(cell_type));

    // Add partitioning attribute to dataset
    std::vector<std::size_t> partitions;
    const std::size_t topology_offset
      = MPI::global_offset(_mpi_comm, topological_data.size()/(cell_dim + 1),
                           true);

    std::vector<std::size_t> topology_offset_tmp(1, topology_offset);
    MPI::gather(_mpi_comm, topology_offset_tmp, partitions);
    MPI::broadcast(_mpi_comm, partitions);
    HDF5Interface::add_attribute(_hdf5_file_id, topology_dataset,
                                 "partition", partitions);

    // ---------- Markers
    for (std::size_t d = 0; d <= mesh.domains().max_dim(); d++)
    {
      const std::map<std::size_t, std::size_t>& domain
        = mesh.domains().markers(d);
      auto _mesh = reference_to_no_delete_pointer(mesh);
      MeshValueCollection<std::size_t> collection(_mesh, d);
      std::map<std::size_t, std::size_t>::const_iterator it;
      for (it = domain.begin(); it != domain.end(); ++it)
        collection.set_value(it->first, it->second);
      const std::string marker_dataset
        = name + "/domain_" + std::to_string(d);
      write_mesh_value_collection(collection, marker_dataset);
    }

  }
}
//-----------------------------------------------------------------------------
void HDF5File::write(const MeshFunction<std::size_t>& meshfunction,
                     const std::string name)
{
  write_mesh_function(meshfunction, name);
}
//-----------------------------------------------------------------------------
void HDF5File::read(MeshFunction<std::size_t>& meshfunction,
                    const std::string name) const
{
  read_mesh_function(meshfunction, name);
}
//-----------------------------------------------------------------------------
void HDF5File::write(const MeshFunction<int>& meshfunction,
                     const std::string name)
{
  write_mesh_function(meshfunction, name);
}
//-----------------------------------------------------------------------------
void HDF5File::read(MeshFunction<int>& meshfunction,
                    const std::string name) const
{
  read_mesh_function(meshfunction, name);
}
//-----------------------------------------------------------------------------
void HDF5File::write(const MeshFunction<double>& meshfunction,
                     const std::string name)
{
  write_mesh_function(meshfunction, name);
}
//-----------------------------------------------------------------------------
void HDF5File::read(MeshFunction<double>& meshfunction,
                    const std::string name) const
{
  read_mesh_function(meshfunction, name);
}
//-----------------------------------------------------------------------------
void HDF5File::write(const MeshFunction<bool>& meshfunction,
                     const std::string name)
{
  std::shared_ptr<const Mesh> mesh = meshfunction.mesh();
  dolfin_assert(mesh);
  const std::size_t cell_dim = meshfunction.dim();

  // HDF5 does not support a boolean type,
  // so copy to int with values 1 and 0
  MeshFunction<int> mf(mesh, cell_dim);
  for (MeshEntityIterator cell(*mesh, cell_dim); !cell.end(); ++cell)
    mf[cell->index()] = (meshfunction[cell->index()] ? 1 : 0);

  write_mesh_function(mf, name);
}
//-----------------------------------------------------------------------------
void HDF5File::read(MeshFunction<bool>& meshfunction,
                    const std::string name) const
{
  std::shared_ptr<const Mesh> mesh = meshfunction.mesh();
  dolfin_assert(mesh);

  const std::size_t cell_dim = meshfunction.dim();

  // HDF5 does not support bool, so use int instead
  MeshFunction<int> mf(mesh, cell_dim);
  read_mesh_function(mf, name);

  for (MeshEntityIterator cell(*mesh, cell_dim); !cell.end(); ++cell)
    meshfunction[cell->index()] = (mf[cell->index()] != 0);
}
//-----------------------------------------------------------------------------
template <typename T>
void HDF5File::read_mesh_function(MeshFunction<T>& meshfunction,
                                  const std::string mesh_name) const
{
  std::shared_ptr<const Mesh> mesh = meshfunction.mesh();
  dolfin_assert(mesh);
  dolfin_assert(_hdf5_file_id > 0);

  const std::string topology_name = mesh_name + "/topology";

  if (!HDF5Interface::has_dataset(_hdf5_file_id, topology_name))
  {
    dolfin_error("HDF5File.cpp",
                 "read topology dataset",
                 "Dataset \"%s\" not found", topology_name.c_str());
  }

  // Look for Coordinates dataset - but not used
  const std::string coordinates_name = mesh_name + "/coordinates";
  if (!HDF5Interface::has_dataset(_hdf5_file_id, coordinates_name))
  {
    dolfin_error("HDF5File.cpp",
                 "read coordinates dataset",
                 "Dataset \"%s\" not found", coordinates_name.c_str());
  }

  // Look for Values dataset
  const std::string values_name = mesh_name + "/values";
  if (!HDF5Interface::has_dataset(_hdf5_file_id, values_name))
  {
    dolfin_error("HDF5File.cpp",
                 "read values dataset",
                 "Dataset \"%s\" not found", values_name.c_str());
  }

  // --- Topology ---

  // Discover size of topology dataset
  const std::vector<std::int64_t> topology_shape
      = HDF5Interface::get_dataset_shape(_hdf5_file_id, topology_name);

  // Some consistency checks

  const std::size_t num_global_cells = topology_shape[0];
  const std::size_t vertices_per_cell = topology_shape[1];
  const std::size_t cell_dim = vertices_per_cell - 1;

  // Initialise if called from MeshFunction constructor with filename
  // argument
  if (meshfunction.size() == 0)
    meshfunction.init(cell_dim);

  // Otherwise, pre-existing MeshFunction must have correct dimension
  if (cell_dim != meshfunction.dim())
  {
    dolfin_error("HDF5File.cpp",
                 "read meshfunction topology",
                 "Cell dimension mismatch");
  }

  // Ensure size_global(cell_dim) is set
  DistributedMeshTools::number_entities(*mesh, cell_dim);

  if (num_global_cells != mesh->size_global(cell_dim))
  {
    dolfin_error("HDF5File.cpp",
                 "read meshfunction topology",
                 "Mesh dimension mismatch");
  }

  // Divide up cells ~equally between processes
  const std::pair<std::size_t, std::size_t> cell_range
    = MPI::local_range(_mpi_comm, num_global_cells);
  const std::size_t num_read_cells = cell_range.second - cell_range.first;

  // Read a block of cells
  std::vector<std::size_t> topology_data;
  topology_data.reserve(num_read_cells*vertices_per_cell);
  HDF5Interface::read_dataset(_hdf5_file_id, topology_name, cell_range,
                              topology_data);

  boost::multi_array_ref<std::size_t, 2>
    topology_array(topology_data.data(),
                   boost::extents[num_read_cells][vertices_per_cell]);

  std::vector<T> value_data;
  value_data.reserve(num_read_cells);
  HDF5Interface::read_dataset(_hdf5_file_id, values_name, cell_range,
                              value_data);

  // Now send the read data to each process on the basis of the first
  // vertex of the entity, since we do not know the global_index
  const std::size_t num_processes = MPI::size(_mpi_comm);
  const std::size_t max_vertex = mesh->size_global(0);

  std::vector<std::vector<std::size_t>> send_topology(num_processes);
  std::vector<std::vector<T>> send_values(num_processes);
  for (std::size_t i = 0; i < num_read_cells ; ++i)
  {
    std::vector<std::size_t> cell_topology(topology_array[i].begin(),
                                           topology_array[i].end());
    std::sort(cell_topology.begin(), cell_topology.end());

    // Use first vertex to decide where to send this data
    const std::size_t send_to_process
      = MPI::index_owner(_mpi_comm, cell_topology.front(), max_vertex);

    send_topology[send_to_process].insert(send_topology[send_to_process].end(),
                                          cell_topology.begin(),
                                          cell_topology.end());
    send_values[send_to_process].push_back(value_data[i]);
  }

  std::vector<std::vector<std::size_t>> receive_topology(num_processes);
  std::vector<std::vector<T>> receive_values(num_processes);
  MPI::all_to_all(_mpi_comm, send_topology, receive_topology);
  MPI::all_to_all(_mpi_comm, send_values, receive_values);

  // Generate requests for data from remote processes, based on the
  // first vertex of the MeshEntities which belong on this process
  // Send our process number, and our local index, so it can come back
  // directly to the right place
  std::vector<std::vector<std::size_t>> send_requests(num_processes);
  const std::size_t process_number = MPI::rank(_mpi_comm);
  for (MeshEntityIterator cell(*mesh, cell_dim, "all"); !cell.end(); ++cell)
  {
    std::vector<std::size_t> cell_topology;
    for (VertexIterator v(*cell); !v.end(); ++v)
      cell_topology.push_back(v->global_index());
    std::sort(cell_topology.begin(), cell_topology.end());

    // Use first vertex to decide where to send this request
    std::size_t send_to_process = MPI::index_owner(_mpi_comm,
                                                   cell_topology.front(),
                                                   max_vertex);
    // Map to this process and local index by appending to send data
    cell_topology.push_back(cell->index());
    cell_topology.push_back(process_number);
    send_requests[send_to_process].insert(send_requests[send_to_process].end(),
                                          cell_topology.begin(),
                                          cell_topology.end());
  }

  std::vector<std::vector<std::size_t>> receive_requests(num_processes);
  MPI::all_to_all(_mpi_comm, send_requests, receive_requests);

  // At this point, the data with its associated vertices is in
  // receive_values and receive_topology and the final destinations
  // are stored in receive_requests as
  // [vertices][index][process][vertices][index][process]...  Some
  // data will have more than one destination

  // Create a mapping from the topology vector to the desired data
  typedef boost::unordered_map<std::vector<std::size_t>, T> VectorKeyMap;
  VectorKeyMap cell_to_data;

  for (std::size_t i = 0; i < receive_values.size(); ++i)
  {
    dolfin_assert(receive_values[i].size()*vertices_per_cell
                  == receive_topology[i].size());
    std::vector<std::size_t>::iterator p = receive_topology[i].begin();
    for (std::size_t j = 0; j < receive_values[i].size(); ++j)
    {
      const std::vector<std::size_t> cell(p, p + vertices_per_cell);
      cell_to_data[cell] = receive_values[i][j];
      p += vertices_per_cell;
    }
  }

  // Clear vectors for reuse - now to send values and indices to final
  // destination
  send_topology = std::vector<std::vector<std::size_t>>(num_processes);
  send_values = std::vector<std::vector<T>>(num_processes);

  // Go through requests, which are stacked as [vertex, vertex, ...]
  // [index] [proc] etc.  Use the vertices as the key for the map
  // (above) to retrieve the data to send to proc
  for (std::size_t i = 0; i < receive_requests.size(); ++i)
  {
    for (std::vector<std::size_t>::iterator p = receive_requests[i].begin();
         p != receive_requests[i].end(); p += (vertices_per_cell + 2))
    {
      const std::vector<std::size_t> cell(p, p + vertices_per_cell);
      const std::size_t remote_index = *(p + vertices_per_cell);
      const std::size_t send_to_proc = *(p + vertices_per_cell + 1);

      const typename VectorKeyMap::iterator find_cell = cell_to_data.find(cell);
      dolfin_assert(find_cell != cell_to_data.end());
      send_values[send_to_proc].push_back(find_cell->second);
      send_topology[send_to_proc].push_back(remote_index);
    }
  }

  MPI::all_to_all(_mpi_comm, send_topology, receive_topology);
  MPI::all_to_all(_mpi_comm, send_values, receive_values);

  // At this point, receive_topology should only list the local indices
  // and received values should have the appropriate values for each
  for (std::size_t i = 0; i < receive_values.size(); ++i)
  {
    dolfin_assert(receive_values[i].size() == receive_topology[i].size());
    for (std::size_t j = 0; j < receive_values[i].size(); ++j)
      meshfunction[receive_topology[i][j]] = receive_values[i][j];
  }
}
//-----------------------------------------------------------------------------
template <typename T>
void HDF5File::write_mesh_function(const MeshFunction<T>& meshfunction,
                                   const std::string name)
{
  if (meshfunction.size() == 0)
  {
    dolfin_error("HDF5File.cpp",
                 "save empty MeshFunction",
                 "No values in MeshFunction");
  }

  const Mesh& mesh = *meshfunction.mesh();
  const std::size_t cell_dim = meshfunction.dim();

  // Write a mesh for the MeshFunction - this will also globally
  // number the entities if needed
  write(mesh, cell_dim, name);

  // Storage for output values
  std::vector<T> data_values;

  if (cell_dim == mesh.topology().dim() || MPI::size(_mpi_comm) == 1)
  {
    // No duplicates - ignore ghost cells if present
    data_values.assign(meshfunction.values(),
       meshfunction.values() + mesh.topology().ghost_offset(cell_dim));
  }
  else
  {
    // In parallel and not CellFunction
    data_values.reserve(mesh.size(cell_dim));

    // Drop duplicate data
    const std::size_t tdim = mesh.topology().dim();
    const std::size_t mpi_rank = MPI::rank(_mpi_comm);
    const std::map<unsigned int, std::set<unsigned int>>& shared_entities
      = mesh.topology().shared_entities(cell_dim);

    std::set<unsigned int> non_local_entities;
    if (mesh.topology().size(tdim) == mesh.topology().ghost_offset(tdim))
    {
      // No ghost cells
      // Exclude shared entities which are on lower rank processes
      for (auto sh = shared_entities.begin(); sh != shared_entities.end(); ++sh)
      {
        const unsigned int lowest_proc = *(sh->second.begin());
        if (lowest_proc < mpi_rank)
          non_local_entities.insert(sh->first);
      }
    }
    else
    {
      // Iterate through ghost cells, adding non-ghost entities which are
      // shared from lower rank process cells to a set for exclusion
      // from output
      for (MeshEntityIterator c(mesh, tdim, "ghost"); !c.end(); ++c)
      {
        const unsigned int cell_owner = c->owner();
        for (MeshEntityIterator ent(*c, cell_dim); !ent.end(); ++ent)
        {
          if (!ent->is_ghost() && cell_owner < mpi_rank)
            non_local_entities.insert(ent->index());
        }
      }
    }

    for (MeshEntityIterator ent(mesh, cell_dim); !ent.end(); ++ent)
    {
      if (non_local_entities.find(ent->index()) == non_local_entities.end())
        data_values.push_back(meshfunction[*ent]);
    }
  }

  // Write values to HDF5
  std::vector<std::int64_t> global_size(1, MPI::sum(_mpi_comm,
                                                    data_values.size()));
  const bool mpi_io = MPI::size(_mpi_comm) > 1 ? true : false;
  write_data(name + "/values", data_values, global_size, mpi_io);
}
//-----------------------------------------------------------------------------
void HDF5File::write(const Function& u, const std::string name,
                     double timestamp)
{
  dolfin_assert(_hdf5_file_id > 0);
  if (!HDF5Interface::has_dataset(_hdf5_file_id, name))
  {
    write(u, name);
    const std::size_t vec_count = 1;
    attributes(name).set("count", vec_count);
    const std::string vec_name = name + "/vector_0";
    attributes(vec_name).set("timestamp", timestamp);
  }
  else
  {
    HDF5Attribute attr = attributes(name);
    if (!attr.exists("count"))
    {
      dolfin_error("HDF5File.cpp",
                   "append to series",
                   "Function dataset does not contain a series 'count' attribute");
    }

    // Get count of vectors in dataset, and increment
    std::size_t vec_count;
    attr.get("count", vec_count);
    std::string vec_name = name + "/vector_" + std::to_string(vec_count);
    ++vec_count;
    attr.set("count", vec_count);

    // Write new vector and save timestamp
    write(*u.vector(), vec_name);
    attributes(vec_name).set("timestamp", timestamp);

  }
}
//-----------------------------------------------------------------------------
void HDF5File::write(const Function& u, const std::string name)
{
  Timer t0("HDF5: write Function");
  dolfin_assert(_hdf5_file_id > 0);

  // Get mesh and dofmap
  dolfin_assert(u.function_space()->mesh());
  const Mesh& mesh = *u.function_space()->mesh();

  dolfin_assert(u.function_space()->dofmap());
  const GenericDofMap& dofmap = *u.function_space()->dofmap();

  // FIXME:
  // Possibly sort cell_dofs into global cell order before writing?

  // Save data in compressed format with an index to mark out
  // the start of each row

  const std::size_t tdim = mesh.topology().dim();
  std::vector<dolfin::la_index> cell_dofs;
  std::vector<std::size_t> x_cell_dofs;
  const std::size_t n_cells = mesh.topology().ghost_offset(tdim);
  x_cell_dofs.reserve(n_cells);

  std::vector<std::size_t> local_to_global_map;
  dofmap.tabulate_local_to_global_dofs(local_to_global_map);

  for (std::size_t i = 0; i != n_cells; ++i)
  {
    x_cell_dofs.push_back(cell_dofs.size());
    const ArrayView<const dolfin::la_index> cell_dofs_i = dofmap.cell_dofs(i);
    for (auto p = cell_dofs_i.begin(); p != cell_dofs_i.end(); ++p)
    {
      dolfin_assert(*p < (dolfin::la_index)local_to_global_map.size());
      cell_dofs.push_back(local_to_global_map[*p]);
    }
  }

  // Add offset to CSR index to be seamless in parallel
  std::size_t offset = MPI::global_offset(_mpi_comm, cell_dofs.size(), true);
  std::transform(x_cell_dofs.begin(),
                 x_cell_dofs.end(),
                 x_cell_dofs.begin(),
                 std::bind2nd(std::plus<std::size_t>(), offset));

  const bool mpi_io = MPI::size(_mpi_comm) > 1 ? true : false;

  // Save DOFs on each cell
  std::vector<std::int64_t> global_size(1, MPI::sum(_mpi_comm,
                                                    cell_dofs.size()));
  write_data(name + "/cell_dofs", cell_dofs, global_size, mpi_io);
  if (MPI::rank(_mpi_comm) == MPI::size(_mpi_comm) - 1)
    x_cell_dofs.push_back(global_size[0]);
  global_size[0] = mesh.size_global(tdim) + 1;
  write_data(name + "/x_cell_dofs", x_cell_dofs, global_size, mpi_io);

  // Save cell ordering - copy to local vector and cut off ghosts
  std::vector<std::size_t> cells(mesh.topology().global_indices(tdim).begin(),
                       mesh.topology().global_indices(tdim).begin() + n_cells);

  global_size[0] = mesh.size_global(tdim);
  write_data(name + "/cells", cells, global_size, mpi_io);

  HDF5Interface::add_attribute(_hdf5_file_id, name, "signature",
                               u.function_space()->element()->signature());

  // Save vector
  write(*u.vector(), name + "/vector_0");
}
//-----------------------------------------------------------------------------
void HDF5File::read(Function& u, const std::string name)
{
  Timer t0("HDF5: read Function");
  dolfin_assert(_hdf5_file_id > 0);

  // FIXME: This routine is long and involves a lot of MPI, but it
  // should work for the general case of reading a function that was
  // written from a different number of processes.  Memory efficiency
  // could be improved by limiting the scope of some of the temporary
  // variables

  std::string basename = name;
  std::string vector_dataset_name = name + "/vector_0";

  // Check that the name we have been given corresponds to a "group"
  // If not, then maybe we have been given the vector dataset name
  // directly, so the group name should be one level up.
  if (!HDF5Interface::has_group(_hdf5_file_id, basename))
  {
    basename = name.substr(0, name.rfind("/"));
    vector_dataset_name = name;
  }

  const std::string cells_dataset_name = basename + "/cells";
  const std::string cell_dofs_dataset_name = basename + "/cell_dofs";
  const std::string x_cell_dofs_dataset_name = basename + "/x_cell_dofs";

  // Check datasets exist
  if (!HDF5Interface::has_group(_hdf5_file_id, basename))
  {
    dolfin_error("HDF5File.cpp",
                 "read function from file",
                 "Group with name \"%s\" does not exist", name.c_str());
  }

  if (!HDF5Interface::has_dataset(_hdf5_file_id, cells_dataset_name))
  {
    dolfin_error("HDF5File.cpp",
                 "read function from file",
                 "Dataset with name \"%s\" does not exist",
                 cells_dataset_name.c_str());
  }

  if (!HDF5Interface::has_dataset(_hdf5_file_id, cell_dofs_dataset_name))
  {
    dolfin_error("HDF5File.cpp",
                 "read function from file",
                 "Dataset with name \"%s\" does not exist",
                 cell_dofs_dataset_name.c_str());
  }

  if (!HDF5Interface::has_dataset(_hdf5_file_id, x_cell_dofs_dataset_name))
  {
    dolfin_error("HDF5File.cpp",
                 "read function from file",
                 "Dataset with name \"%s\" does not exist",
                 x_cell_dofs_dataset_name.c_str());
  }

  // Check if it has the vector_0-dataset. If not, it may be stored
  // with an older version, and instead have a vector-dataset.
  if (!HDF5Interface::has_dataset(_hdf5_file_id, vector_dataset_name))
  {
    std::string tmp_name = vector_dataset_name;
    const std::size_t N = vector_dataset_name.rfind("/vector_0");
    if (N != std::string::npos)
      vector_dataset_name = vector_dataset_name.substr(0, N) + "/vector";

    if (!HDF5Interface::has_dataset(_hdf5_file_id, vector_dataset_name))
    {
      dolfin_error("HDF5File.cpp",
                   "read function from file",
                   "Dataset with name \"%s\" does not exist",
                   tmp_name.c_str());
    }
  }

  // Get existing mesh and dofmap - these should be pre-existing
  // and set up by user when defining the Function
  dolfin_assert(u.function_space()->mesh());
  const Mesh& mesh = *u.function_space()->mesh();
  dolfin_assert(u.function_space()->dofmap());
  const GenericDofMap& dofmap = *u.function_space()->dofmap();

  // Get dimension of dataset
  const std::vector<std::int64_t> dataset_shape =
    HDF5Interface::get_dataset_shape(_hdf5_file_id, cells_dataset_name);
  const std::size_t num_global_cells = dataset_shape[0];
  if (mesh.size_global(mesh.topology().dim()) != num_global_cells)
  {
    dolfin_error("HDF5File.cpp",
                 "read Function from file",
                 "Number of global cells does not match");
  }

  // Divide cells equally between processes
  const std::pair<std::size_t, std::size_t> cell_range
    = MPI::local_range(_mpi_comm, num_global_cells);

  // Read cells
  std::vector<std::size_t> input_cells;
  HDF5Interface::read_dataset(_hdf5_file_id, cells_dataset_name,
                              cell_range, input_cells);

  // Overlap reads of DOF indices, to get full range on each process
  std::vector<std::size_t> x_cell_dofs;
  HDF5Interface::read_dataset(_hdf5_file_id, x_cell_dofs_dataset_name,
                              std::make_pair(cell_range.first,
                                             cell_range.second + 1),
                              x_cell_dofs);

  // Read cell-DOF maps
  std::vector<dolfin::la_index> input_cell_dofs;
  HDF5Interface::read_dataset(_hdf5_file_id, cell_dofs_dataset_name,
                              std::make_pair(x_cell_dofs.front(),
                                             x_cell_dofs.back()),
                              input_cell_dofs);

  GenericVector& x = *u.vector();

  const std::vector<std::int64_t> vector_shape =
    HDF5Interface::get_dataset_shape(_hdf5_file_id, vector_dataset_name);
  const std::size_t num_global_dofs = vector_shape[0];
  dolfin_assert(num_global_dofs == x.size(0));
  const std::pair<dolfin::la_index, dolfin::la_index>
    input_vector_range = MPI::local_range(_mpi_comm, vector_shape[0]);

  std::vector<double> input_values;
  HDF5Interface::read_dataset(_hdf5_file_id, vector_dataset_name,
                              input_vector_range, input_values);

  // Calculate one (global cell, local_dof_index) to associate with
  // each item in the vector on this process
  std::vector<std::size_t> global_cells;
  std::vector<std::size_t> remote_local_dofi;
  HDF5Utility::map_gdof_to_cell(_mpi_comm, input_cells, input_cell_dofs,
                                x_cell_dofs, input_vector_range, global_cells,
                                remote_local_dofi);

  // At this point, each process has a set of data, and for each
  // value, a global_cell and local_dof to send it to.  However, it is
  // not known which processes the cells are actually on.

  // Find where the needed cells are held
  std::vector<std::pair<std::size_t, std::size_t>>
    cell_ownership = HDF5Utility::cell_owners(mesh, global_cells);

  // Having found the cell location, the actual global_dof index held
  // by that (cell, local_dof) is needed on the process which holds
  // the data values
  std::vector<dolfin::la_index> global_dof;
  HDF5Utility::get_global_dof(_mpi_comm, cell_ownership, remote_local_dofi,
                              input_vector_range, dofmap, global_dof);


  const std::size_t num_processes = MPI::size(_mpi_comm);

  // Shift to dividing things into the vector range of Function Vector
  const std::pair<dolfin::la_index, dolfin::la_index>
    vector_range = x.local_range();

  std::vector<std::vector<double>> receive_values(num_processes);
  std::vector<std::vector<dolfin::la_index>> receive_indices(num_processes);
  {
    std::vector<std::vector<double>> send_values(num_processes);
    std::vector<std::vector<dolfin::la_index>> send_indices(num_processes);
    const std::size_t
      n_vector_vals = input_vector_range.second - input_vector_range.first;
    std::vector<dolfin::la_index> all_vec_range;

    std::vector<dolfin::la_index> vector_range_second(1, vector_range.second);
    MPI::gather(_mpi_comm, vector_range_second, all_vec_range);
    MPI::broadcast(_mpi_comm, all_vec_range);

    for (std::size_t i = 0; i != n_vector_vals; ++i)
    {
      const std::size_t dest
        = std::upper_bound(all_vec_range.begin(), all_vec_range.end(),
                           global_dof[i]) - all_vec_range.begin();
      dolfin_assert(dest < num_processes);
      dolfin_assert(i < input_values.size());
      send_indices[dest].push_back(global_dof[i]);
      send_values[dest].push_back(input_values[i]);
    }

    MPI::all_to_all(_mpi_comm, send_values, receive_values);
    MPI::all_to_all(_mpi_comm, send_indices, receive_indices);
  }

  std::vector<double> vector_values(vector_range.second - vector_range.first);
  for (std::size_t i = 0; i != num_processes; ++i)
  {
    const std::vector<double>& rval = receive_values[i];
    const std::vector<dolfin::la_index>& rindex = receive_indices[i];
    dolfin_assert(rval.size() == rindex.size());
    for (std::size_t j = 0; j != rindex.size(); ++j)
    {
      dolfin_assert(rindex[j] >= vector_range.first);
      dolfin_assert(rindex[j] < vector_range.second);
      vector_values[rindex[j] - vector_range.first] = rval[j];
    }
  }

  x.set_local(vector_values);
  x.apply("insert");
}
//-----------------------------------------------------------------------------
void HDF5File::write(const MeshValueCollection<std::size_t>& mesh_values,
                     const std::string name)
{
  write_mesh_value_collection(mesh_values, name);
}
//-----------------------------------------------------------------------------
void HDF5File::read(MeshValueCollection<std::size_t>& mesh_values,
                    const std::string name) const
{
  read_mesh_value_collection(mesh_values, name);
}
//-----------------------------------------------------------------------------
void HDF5File::write(const MeshValueCollection<double>& mesh_values,
                     const std::string name)
{
  write_mesh_value_collection(mesh_values, name);
}
//-----------------------------------------------------------------------------
void HDF5File::read(MeshValueCollection<double>& mesh_values,
                    const std::string name) const
{
  read_mesh_value_collection(mesh_values, name);
}
//-----------------------------------------------------------------------------
void HDF5File::write(const MeshValueCollection<bool>& mesh_values,
                     const std::string name)
{
  // HDF5 does not implement bool, use int and copy

  MeshValueCollection<int> mvc_int(mesh_values.mesh(), mesh_values.dim());
  const std::map<std::pair<std::size_t, std::size_t>, bool>& values
    = mesh_values.values();
  for (auto mesh_value_it = values.begin(); mesh_value_it != values.end();
       ++mesh_value_it)
  {
    mvc_int.set_value(mesh_value_it->first.first, mesh_value_it->first.second,
                      mesh_value_it->second ? 1 : 0);
  }

  write_mesh_value_collection(mvc_int, name);
}
//-----------------------------------------------------------------------------
void HDF5File::read(MeshValueCollection<bool>& mesh_values,
                    const std::string name) const
{
  // HDF5 does not implement bool, use int and copy

  MeshValueCollection<int> mvc_int(mesh_values.mesh(), mesh_values.dim());
  read_mesh_value_collection(mvc_int, name);

  const std::map<std::pair<std::size_t, std::size_t>, int>& values
    = mvc_int.values();
  for (auto mesh_value_it = values.begin(); mesh_value_it != values.end();
       ++mesh_value_it)
  {
    mesh_values.set_value(mesh_value_it->first.first,
                          mesh_value_it->first.second,
                          (mesh_value_it->second != 0));
  }

}
//-----------------------------------------------------------------------------
template <typename T>
void HDF5File::write_mesh_value_collection(const MeshValueCollection<T>& mesh_values,
                                           const std::string name)
{
  dolfin_assert(_hdf5_file_id > 0);

  const std::size_t dim = mesh_values.dim();
  std::shared_ptr<const Mesh> mesh = mesh_values.mesh();

  const std::map<std::pair<std::size_t, std::size_t>, T>& values
    = mesh_values.values();

  std::unique_ptr<CellType>
    entity_type(CellType::create(mesh->type().entity_type(dim)));
  const std::size_t num_vertices_per_entity
    = (dim == 0) ? 1 : entity_type->num_vertices();

  std::vector<std::size_t> topology;
  std::vector<T> value_data;
  topology.reserve(values.size()*num_vertices_per_entity);
  value_data.reserve(values.size());

  const std::size_t tdim = mesh->topology().dim();
  mesh->init(tdim, dim);
  for (auto &p : values)
  {
    MeshEntity cell = Cell(*mesh, p.first.first);
    if (dim != tdim)
    {
      const unsigned int entity_local_idx = cell.entities(dim)[p.first.second];
      cell = MeshEntity(*mesh, dim, entity_local_idx);
    }
    for (VertexIterator v(cell); !v.end(); ++v)
      topology.push_back(v->global_index());
    value_data.push_back(p.second);
  }

  const bool mpi_io = MPI::size(_mpi_comm) > 1 ? true : false;
  std::vector<std::int64_t> global_size(2);

  global_size[0] = MPI::sum(_mpi_comm, values.size());
  global_size[1] = num_vertices_per_entity;

  // FIXME: this should throw an error, but is here because
  // "mesh domains" call write_mesh_value_collection with empty
  // datasets sometimes. Remove when mesh domains are removed.
  if (global_size[0] > 0)
  {
    write_data(name + "/topology", topology, global_size, mpi_io);

    global_size[1] = 1;
    write_data(name + "/values", value_data, global_size, mpi_io);
    HDF5Interface::add_attribute(_hdf5_file_id, name, "dimension",
                                 mesh_values.dim());
  }
}
//-----------------------------------------------------------------------------
template <typename T>
void HDF5File::write_mesh_value_collection_old(
                    const MeshValueCollection<T>& mesh_values,
                    const std::string name)
{
  dolfin_assert(_hdf5_file_id > 0);

  const std::map<std::pair<std::size_t, std::size_t>, T>& values
    = mesh_values.values();

  const Mesh& mesh = *mesh_values.mesh();
  const std::vector<std::size_t>& global_cell_index
    = mesh.topology().global_indices(mesh.topology().dim());

  std::vector<T> data_values;
  std::vector<std::size_t> entities;
  std::vector<std::size_t> cells;
  for (auto p = values.begin(); p != values.end(); ++p)
  {
    cells.push_back(global_cell_index[p->first.first]);
    entities.push_back(p->first.second);
    data_values.push_back(p->second);
  }

  std::vector<std::int64_t> global_size(1, MPI::sum(_mpi_comm,
                                                   data_values.size()));

  // Only write if the global size is larger than 0
  if (global_size[0] > 0)
  {
    const bool mpi_io = MPI::size(_mpi_comm) > 1 ? true : false;
    write_data(name + "/values", data_values, global_size, mpi_io);
    write_data(name + "/entities", entities, global_size, mpi_io);
    write_data(name + "/cells", cells, global_size, mpi_io);

    HDF5Interface::add_attribute(_hdf5_file_id, name, "dimension",
                                 mesh_values.dim());
  }
}
//-----------------------------------------------------------------------------
template <typename T>
void HDF5File::read_mesh_value_collection(MeshValueCollection<T>& mesh_vc,
                                          const std::string name) const
{
  Timer t1("HDF5: read mesh value collection");
  dolfin_assert(_hdf5_file_id > 0);

  if (!HDF5Interface::has_group(_hdf5_file_id, name))
  {
    dolfin_error("HDF5File.cpp",
                 "open MeshValueCollection dataset",
                 "Group \"%s\" not found in file", name.c_str());
  }

  if (HDF5Interface::has_dataset(_hdf5_file_id, name + "/cells"))
  {
    warning("Found old MeshValueCollection format");
    read_mesh_value_collection_old(mesh_vc, name);
    return;
  }

  std::size_t dim = 0;
  HDF5Interface::get_attribute(_hdf5_file_id, name, "dimension", dim);
  std::shared_ptr<const Mesh> mesh = mesh_vc.mesh();
  dolfin_assert(mesh);
  std::unique_ptr<CellType>
    entity_type(CellType::create(mesh->type().entity_type(dim)));
  const std::size_t num_verts_per_entity = entity_type->num_entities(0);

  // Reset MeshValueCollection
  mesh_vc.clear();
  mesh_vc.init(mesh, dim);

  const std::string values_name = name + "/values";
  const std::string topology_name = name + "/topology";

  if (!HDF5Interface::has_dataset(_hdf5_file_id, values_name))
  {
    dolfin_error("HDF5File.cpp",
                 "open MeshValueCollection dataset",
                 "Dataset \"%s\" not found in file", values_name.c_str());
  }

  if (!HDF5Interface::has_dataset(_hdf5_file_id, topology_name))
  {
    dolfin_error("HDF5File.cpp",
                 "open MeshValueCollection dataset",
                 "Dataset \"%s\" not found in file", topology_name.c_str());
  }

  // Check both datasets have the same number of entries
  const std::vector<std::int64_t> values_shape
      = HDF5Interface::get_dataset_shape(_hdf5_file_id, values_name);
  const std::vector<std::int64_t> topology_shape
    = HDF5Interface::get_dataset_shape(_hdf5_file_id, topology_name);
  dolfin_assert(values_shape[0] == topology_shape[0]);

  // Divide range between processes
  const std::pair<std::size_t, std::size_t> data_range
    = MPI::local_range(_mpi_comm, values_shape[0]);
  const std::size_t local_size = data_range.second - data_range.first;

  // Read local range of values and entities
  std::vector<T> values_data;
  values_data.reserve(local_size);
  HDF5Interface::read_dataset(_hdf5_file_id, values_name, data_range,
                              values_data);
  std::vector<std::size_t> topology_data;
  topology_data.reserve(local_size*num_verts_per_entity);
  HDF5Interface::read_dataset(_hdf5_file_id, topology_name, data_range,
                              topology_data);

  /// Basically need to tabulate all entities by vertex, and get their
  /// local index, transmit them to a 'sorting' host.  Also send the
  /// read data to the 'sorting' hosts.

  // Ensure the mesh dimension is initialised
  mesh->init(dim);
  std::size_t global_vertex_range = mesh->size_global(0);
  std::vector<std::size_t> v(num_verts_per_entity);
  const std::size_t num_processes = MPI::size(_mpi_comm);

  // Calculate map from entity vertices to {process, local index}
  std::map<std::vector<std::size_t>,
           std::vector<std::size_t>> entity_map;

  std::vector<std::vector<std::size_t>> send_entities(num_processes);
  std::vector<std::vector<std::size_t>> recv_entities(num_processes);

  for (MeshEntityIterator m(*mesh, dim); !m.end(); ++m)
  {
    if (dim == 0)
      v[0] = m->global_index();
    else
    {
      for (VertexIterator vtx(*m); !vtx.end(); ++vtx)
        v[vtx.pos()] = vtx->global_index();
      std::sort(v.begin(), v.end());
    }

    std::size_t dest = MPI::index_owner(_mpi_comm,
                                        v[0], global_vertex_range);
    send_entities[dest].push_back(m->index());
    send_entities[dest].insert(send_entities[dest].end(),
                               v.begin(), v.end());
  }

  MPI::all_to_all(_mpi_comm, send_entities, recv_entities);

  for (std::size_t i = 0; i != num_processes; ++i)
  {
    for (std::vector<std::size_t>::const_iterator it
           = recv_entities[i].begin(); it != recv_entities[i].end();
         it += (num_verts_per_entity + 1))
    {
      std::copy(it + 1, it + num_verts_per_entity + 1, v.begin());
      auto map_it = entity_map.insert({v, {i, *it}});
      if (!map_it.second)
      {
        // Entry already exists, add to it
        map_it.first->second.push_back(i);
        map_it.first->second.push_back(*it);
      }
    }
  }

  // Send data from MeshValueCollection to sorting process

  std::vector<std::vector<T>> send_data(num_processes);
  std::vector<std::vector<T>> recv_data(num_processes);
  // Reset send/recv arrays
  send_entities = std::vector<std::vector<std::size_t>>(num_processes);
  recv_entities = std::vector<std::vector<std::size_t>>(num_processes);

  std::size_t i = 0;
  for (auto it = topology_data.begin(); it != topology_data.end();
       it += num_verts_per_entity)
  {
    std::partial_sort_copy(it, it + num_verts_per_entity,
                           v.begin(), v.end());
    std::size_t dest = MPI::index_owner(_mpi_comm,
                                        v[0], global_vertex_range);
    send_entities[dest].insert(send_entities[dest].end(),
                               v.begin(), v.end());
    send_data[dest].push_back(values_data[i]);
    ++i;
  }

  MPI::all_to_all(_mpi_comm, send_entities, recv_entities);
  MPI::all_to_all(_mpi_comm, send_data, recv_data);

  // Reset send arrays
  send_data = std::vector<std::vector<T>>(num_processes);
  send_entities = std::vector<std::vector<std::size_t>>(num_processes);

  // Locate entity in map, and send back to data to owning processes
  for (std::size_t i = 0; i != num_processes; ++i)
  {
    dolfin_assert(recv_data[i].size()*num_verts_per_entity
                  == recv_entities[i].size());

    for (std::size_t j = 0; j != recv_data[i].size(); ++j)
    {
      auto it = recv_entities[i].begin() + j*num_verts_per_entity;
      std::copy(it, it + num_verts_per_entity, v.begin());
      auto map_it = entity_map.find(v);

      if (map_it == entity_map.end())
      {
        dolfin_error("HDF5File.cpp",
                     "find entity in map",
                     "Error reading MeshValueCollection");
      }
      for (auto p = map_it->second.begin(); p != map_it->second.end(); p += 2)
      {
        const std::size_t dest = *p;
        dolfin_assert(dest < num_processes);
        send_entities[dest].push_back(*(p + 1));
        send_data[dest].push_back(recv_data[i][j]);
      }
    }
  }

  // Send to owning processes and set in MeshValueCollection
  MPI::all_to_all(_mpi_comm, send_entities, recv_entities);
  MPI::all_to_all(_mpi_comm, send_data, recv_data);

  for (std::size_t i = 0; i != num_processes; ++i)
  {
    dolfin_assert(recv_entities[i].size() == recv_data[i].size());
    for (std::size_t j = 0; j != recv_data[i].size(); ++j)
    {
      mesh_vc.set_value(recv_entities[i][j], recv_data[i][j]);
    }
  }

}
//-----------------------------------------------------------------------------
template <typename T>
void HDF5File::read_mesh_value_collection_old(MeshValueCollection<T>& mesh_vc,
                                          const std::string name) const
{
  Timer t1("HDF5: read mesh value collection");
  dolfin_assert(_hdf5_file_id > 0);

  mesh_vc.clear();
  if (!HDF5Interface::has_group(_hdf5_file_id, name))
  {
    dolfin_error("HDF5File.cpp",
                 "open MeshValueCollection dataset",
                 "Group \"%s\" not found in file", name.c_str());
  }

  std::size_t dim = 0;
  HDF5Interface::get_attribute(_hdf5_file_id, name, "dimension", dim);

  const std::string values_name = name + "/values";
  const std::string entities_name = name + "/entities";
  const std::string cells_name = name + "/cells";

  if (!HDF5Interface::has_dataset(_hdf5_file_id, values_name))
  {
    dolfin_error("HDF5File.cpp",
                 "open MeshValueCollection dataset",
                 "Dataset \"%s\" not found in file", values_name.c_str());
  }

  if (!HDF5Interface::has_dataset(_hdf5_file_id, entities_name))
  {
    dolfin_error("HDF5File.cpp",
                 "open MeshValueCollection dataset",
                 "Dataset \"%s\" not found in file", entities_name.c_str());
  }

  if (!HDF5Interface::has_dataset(_hdf5_file_id, cells_name))
  {
    dolfin_error("HDF5File.cpp",
                 "open MeshValueCollection dataset",
                 "Dataset \"%s\" not found in file", cells_name.c_str());
  }

  // Check all datasets have the same size
  const std::vector<std::int64_t> values_shape
      = HDF5Interface::get_dataset_shape(_hdf5_file_id, values_name);
  const std::vector<std::int64_t> entities_shape
      = HDF5Interface::get_dataset_shape(_hdf5_file_id, entities_name);
  const std::vector<std::int64_t> cells_shape
    = HDF5Interface::get_dataset_shape(_hdf5_file_id, cells_name);
  dolfin_assert(values_shape[0] == entities_shape[0]);
  dolfin_assert(values_shape[0] == cells_shape[0]);

  // Check size of dataset. If small enough, just read on all
  // processes...

  // FIXME: optimise value
  const std::int64_t max_data_one = 1048576; // arbitrary 1M

  if (values_shape[0] < max_data_one)
  {
    // read on all processes
    const std::pair<std::size_t, std::size_t> range(0, values_shape[0]);
    const std::size_t local_size = range.second - range.first;

    std::vector<T> values_data;
    values_data.reserve(local_size);
    HDF5Interface::read_dataset(_hdf5_file_id, values_name, range, values_data);
    std::vector<std::size_t> entities_data;
    entities_data.reserve(local_size);
    HDF5Interface::read_dataset(_hdf5_file_id, entities_name, range,
                                entities_data);
    std::vector<std::size_t> cells_data;
    cells_data.reserve(local_size);
    HDF5Interface::read_dataset(_hdf5_file_id, cells_name, range, cells_data);

    // Get global mapping to restore values
    const Mesh& mesh = *mesh_vc.mesh();
    const std::vector<std::size_t>& global_cell_index
      = mesh.topology().global_indices(mesh.topology().dim());

    // Reference to actual map of MeshValueCollection
    std::map<std::pair<std::size_t, std::size_t>, T>& mvc_map
      = mesh_vc.values();

    // Find cells which are on this process,
    // under the assumption that global_cell_index is ordered.
    dolfin_assert(std::is_sorted(global_cell_index.begin(),
                                 global_cell_index.end()));

    // cells_data in general is not ordered, so we sort it
    // keeping track of the indices
    std::vector<std::size_t> cells_data_index(cells_data.size());
    std::iota(cells_data_index.begin(), cells_data_index.end(), 0);
    std::sort(cells_data_index.begin(), cells_data_index.end(),
              [&cells_data](std::size_t i, size_t j)
              { return cells_data[i] < cells_data[j]; });

    // The implementation follows std::set_intersection, which we are
    // not able to use here since we need the indices of the
    // intersection, not just the values.
    std::vector<std::size_t>::const_iterator i = global_cell_index.begin();
    std::vector<std::size_t>::const_iterator j = cells_data_index.begin();
    while (i!=global_cell_index.end() && j!=cells_data_index.end())
    {
      // Global cell index is less than the cell_data index read from
      // file, if global cell index is larger than the cell_data index
      // read from file, else global cell index is the same as the
      // cell_data index read from file
      if (*i < cells_data[*j])
        ++i;
      else if (*i > cells_data[*j])
        ++j;
      else
      {
        // Here we do not increment j because cells_data_index is
        // ordered but not *strictly* ordered.
        std::size_t lidx = i - global_cell_index.begin();
        mvc_map[std::make_pair(lidx, entities_data[*j])] = values_data[*j];
        ++j;
      }
    }
  }
  else
  {
    const Mesh& mesh = *mesh_vc.mesh();

    // Divide range between processes
    const std::pair<std::size_t, std::size_t> data_range
      = MPI::local_range(_mpi_comm, values_shape[0]);
    const std::size_t local_size = data_range.second - data_range.first;

    // Read local range of values, entities and cells
    std::vector<T> values_data;
    values_data.reserve(local_size);
    HDF5Interface::read_dataset(_hdf5_file_id, values_name, data_range,
                                values_data);
    std::vector<std::size_t> entities_data;
    entities_data.reserve(local_size);
    HDF5Interface::read_dataset(_hdf5_file_id, entities_name, data_range,
                                entities_data);
    std::vector<std::size_t> cells_data;
    cells_data.reserve(local_size);
    HDF5Interface::read_dataset(_hdf5_file_id, cells_name, data_range,
                                cells_data);

    std::vector<std::pair<std::size_t, std::size_t>> cell_ownership;
    cell_ownership = HDF5Utility::cell_owners(mesh, cells_data);

    const std::size_t num_processes = MPI::size(_mpi_comm);
    std::vector<std::vector<std::size_t>> send_entities(num_processes);
    std::vector<std::vector<std::size_t>> send_local(num_processes);
    std::vector<std::vector<T>> send_values(num_processes);
    for (std::size_t i = 0; i != cells_data.size(); ++i)
    {
      const std::size_t dest = cell_ownership[i].first;
      send_local[dest].push_back(cell_ownership[i].second);
      send_entities[dest].push_back(entities_data[i]);
      send_values[dest].push_back(values_data[i]);
    }

    std::vector<std::vector<T>> recv_values(num_processes);
    std::vector<std::vector<std::size_t>> recv_entities(num_processes);
    std::vector<std::vector<std::size_t>> recv_local(num_processes);
    MPI::all_to_all(_mpi_comm, send_entities, recv_entities);
    MPI::all_to_all(_mpi_comm, send_local, recv_local);
    MPI::all_to_all(_mpi_comm, send_values, recv_values);

    // Reference to actual map of MeshValueCollection
    std::map<std::pair<std::size_t, std::size_t>, T>& mvc_map
      = mesh_vc.values();

    for (std::size_t i = 0; i < num_processes; ++i)
    {
      const std::vector<std::size_t>& local_index = recv_local[i];
      const std::vector<std::size_t>& local_entities = recv_entities[i];
      const std::vector<T>& local_values = recv_values[i];
      dolfin_assert(local_index.size() == local_entities.size());
      dolfin_assert(local_index.size() == local_values.size());

      for (std::size_t j = 0; j < local_index.size(); ++j)
      {
        mvc_map[std::make_pair(local_index[j], local_entities[j])]
          = local_values[j];
      }
    }
  }
}
//-----------------------------------------------------------------------------
void HDF5File::read(Mesh& input_mesh, const std::string data_path,
                    bool use_partition_from_file) const
{
  dolfin_assert(_hdf5_file_id > 0);

  // Check that topology data set is found in HDF5 file
  const std::string topology_path = data_path + "/topology";
  if (!HDF5Interface::has_dataset(_hdf5_file_id, topology_path))
  {
    dolfin_error("HDF5File.cpp",
                 "read topology dataset",
                 "Dataset \"%s\" not found", topology_path.c_str());
  }

  // Get topology data
  std::string cell_type_str;
  if (HDF5Interface::has_attribute(_hdf5_file_id, topology_path, "celltype"))
  {
    HDF5Interface::get_attribute(_hdf5_file_id, topology_path, "celltype",
                                 cell_type_str);
  }

  // Create CellType from string
  std::unique_ptr<CellType> cell_type(CellType::create(cell_type_str));
  dolfin_assert(cell_type);

  // Check that coordinate data set is found in HDF5 file
  const std::string geometry_path = data_path + "/coordinates";
  if (!HDF5Interface::has_dataset(_hdf5_file_id, geometry_path))
  {
    dolfin_error("HDF5File.cpp",
                 "read coordinates dataset",
                 "Dataset \"%s\" not found", geometry_path.c_str());
  }

  // Get dimensions of coordinate dataset
  std::vector<std::int64_t> coords_shape
    = HDF5Interface::get_dataset_shape(_hdf5_file_id, geometry_path);
  dolfin_assert(coords_shape.size() < 3);
  if (coords_shape.size() == 1)
  {
    dolfin_error("HDF5File.cpp",
                 "get geometric dimension",
                 "Cannot determine geometric dimension from one-dimensional array storage in HDF5 file");
  }
  else if (coords_shape.size() > 2)
  {
    dolfin_error("HDF5File.cpp",
                 "get geometric dimension",
                 "Cannot determine geometric dimension from high-rank array storage in HDF5 file");
  }

  // Extract geometric dimension
  int gdim = coords_shape[1];

  // Build mesh from data in HDF5 file
  read(input_mesh, topology_path, geometry_path, gdim, *cell_type, -1,
       coords_shape[0],
       use_partition_from_file);
}
//-----------------------------------------------------------------------------
void HDF5File::read(Mesh& input_mesh,
                    const std::string topology_path,
                    const std::string geometry_path,
                    const int gdim, const CellType& cell_type,
                    const std::int64_t expected_num_global_cells,
                    const std::int64_t expected_num_global_points,
                    bool use_partition_from_file) const
{
  // FIXME: This function is too big. Split up.

  Timer t("HDF5: read mesh");
  dolfin_assert(_hdf5_file_id > 0);

  // Create structure to store local mesh
  LocalMeshData local_mesh_data(_mpi_comm);
  local_mesh_data.gdim = gdim;

  // --- Topology ---

  // Get number of vertices per cell from CellType
  const int num_vertices_per_cell = cell_type.num_entities(0);

  // Set topology dim and cell type
  local_mesh_data.tdim = cell_type.dim();
  local_mesh_data.cell_type = cell_type.cell_type();
  local_mesh_data.num_vertices_per_cell = num_vertices_per_cell;

  // Discover shape of the topology data set in HDF5 file
  std::vector<std::int64_t> topology_shape
    = HDF5Interface::get_dataset_shape(_hdf5_file_id, topology_path);

  // If cell type is encoded in HDF5, then check for consistency with cell
  // type passed into this function
  if (HDF5Interface::has_attribute(_hdf5_file_id, topology_path, "celltype"))
  {
    std::string cell_type_str;
    HDF5Interface::get_attribute(_hdf5_file_id, topology_path, "celltype",
                                 cell_type_str);
    if (cell_type.cell_type() != CellType:: string2type(cell_type_str))
    {
      dolfin_error("HDF5File.cpp",
                   "read topology data",
                   "Inconsistency between expected cell type and cell type attribie in HDF file");

    }
  }

  // Compute number of global cells (handle case that topology may be
  // arranged a 1D or 2D array)
  std::size_t num_global_cells = 0;
  if (topology_shape.size() == 1)
  {
    dolfin_assert(topology_shape[0] % num_vertices_per_cell == 0);
    num_global_cells = topology_shape[0]/num_vertices_per_cell;
  }
  else if (topology_shape.size() == 2)
  {
    num_global_cells = topology_shape[0];
    if (topology_shape[1] != num_vertices_per_cell)
    {
      dolfin_error("HDF5File.cpp",
                   "read topology data",
                   "Topology in HDF5 file has inconsistent size");
    }
  }
  else
  {
    dolfin_error("HDF5File.cpp",
                 "read coordinate data",
                 "Topology in HDF5 file has wrong shape");
  }

  // Check number of cells (global)
  if (expected_num_global_cells >= 0)
  {
    // Check number of cells for consistency with expected number of cells
    if (num_global_cells != expected_num_global_cells)
    {
      dolfin_error("HDF5File.cpp",
                   "read cell data",
                   "Inconsistentcy between expected number of cells and number of cells in topology in HDF5 file");
    }
  }

  // Set number of cells (global)
  local_mesh_data.num_global_cells = num_global_cells;

  // FIXME: 'partition' is a poor descriptor
  // Get partition from file, if available
  std::vector<std::size_t> cell_partitions;
  if (HDF5Interface::has_attribute(_hdf5_file_id, topology_path, "partition"))
  {
    HDF5Interface::get_attribute(_hdf5_file_id, topology_path, "partition",
                                 cell_partitions);
  }

  // Prepare range of cells to read on this process
  std::pair<std::size_t, std::size_t> cell_range;

  // Check whether number of MPI processes matches partitioning, and
  // restore if possible
  if (dolfin::MPI::size(_mpi_comm) == cell_partitions.size())
  {
    cell_partitions.push_back(num_global_cells);
    const std::size_t proc = MPI::rank(_mpi_comm);
    cell_range = std::make_pair(cell_partitions[proc], cell_partitions[proc + 1]);

    // Restore partitioning if requested
    if (use_partition_from_file)
    {
      local_mesh_data.cell_partition
        = std::vector<std::size_t>(cell_range.second - cell_range.first, proc);
    }
  }
  else
  {
    if (use_partition_from_file)
      warning("Could not use partition from file: wrong size");

    // Divide up cells approximately equally between processes
    cell_range = MPI::local_range(_mpi_comm, num_global_cells);
  }

  // Get number of cells to read on this process
  const int num_local_cells = cell_range.second - cell_range.first;

  // Modify range of array to read for flat HDF5 storage
  std::pair<std::size_t, std::size_t> cell_data_range = cell_range;
  if (topology_shape.size() == 1)
  {
    cell_data_range.first *= num_vertices_per_cell;
    cell_data_range.second *= num_vertices_per_cell;
  }

  // Read a block of cells
  std::vector<std::int64_t> topology_data;
  topology_data.reserve(num_local_cells*num_vertices_per_cell);
  HDF5Interface::read_dataset(_hdf5_file_id, topology_path, cell_data_range,
                              topology_data);

  // FIXME: explain this more clearly.
  // Reconstruct mesh_name from topology_name - needed for
  // cell_indices and domains
  std::string mesh_name = topology_path.substr(0, topology_path.rfind("/"));

  // FIXME: Imrpove comment - it's unclear this is about
  // Look for cell indices in dataset, and use if available
  std::vector<std::size_t>& global_cell_indices = local_mesh_data.global_cell_indices;
  global_cell_indices.clear();
  const std::string cell_indices_name = mesh_name + "/cell_indices";
  if (HDF5Interface::has_dataset(_hdf5_file_id, cell_indices_name))
  {
    global_cell_indices.reserve(num_local_cells);
    HDF5Interface::read_dataset(_hdf5_file_id, cell_indices_name,
                                cell_range, global_cell_indices);
  }
  else
  {
    global_cell_indices.resize(num_local_cells);
    std::iota(global_cell_indices.begin(), global_cell_indices.end(),
              cell_range.first);
  }

  // FIXME: allocate multi_array data and pass to HDFr read function
  // to avoid copy
  // Copy to boost::multi_array
  local_mesh_data.cell_vertices.resize(boost::extents[num_local_cells][num_vertices_per_cell]);
  boost::multi_array_ref<std::int64_t, 2>
    topology_data_array(topology_data.data(),
                        boost::extents[num_local_cells][num_vertices_per_cell]);

  // Remap vertices to DOLFIN ordering from VTK/XDMF ordering
  const std::vector<unsigned int> perm = cell_type.vtk_mapping();
  for (int i = 0; i != num_local_cells; ++i)
  {
    for (int j = 0; j != num_vertices_per_cell; ++j)
      local_mesh_data.cell_vertices[i][j] = topology_data_array[i][perm[j]];
  }

  // --- Coordinates ---

  // Get dimensions of coordinate dataset
  std::vector<std::int64_t> coords_shape
    = HDF5Interface::get_dataset_shape(_hdf5_file_id, geometry_path);

  // Compute number of vertcies
  if (coords_shape.size() == 1)
  {
    dolfin_assert(coords_shape[0] % gdim == 0);
    local_mesh_data.num_global_vertices = coords_shape[0]/gdim;
  }
  else if (coords_shape.size() == 2)
  {
    dolfin_assert((int) coords_shape[1] == gdim);
    local_mesh_data.num_global_vertices = coords_shape[0];
  }
  else
  {
    dolfin_error("HDF5File.cpp",
                 "read coordinate data",
                 "Topology in HDF5 file has wrong shape");
  }

  // Check number of vertices (global) against expected number
  if (expected_num_global_points >= 0)
  {
    // Check number of cells for consistency with expected number of cells
    if (local_mesh_data.num_global_vertices != expected_num_global_points)
    {
      dolfin_error("HDF5File.cpp",
                   "read vertex data",
                   "Inconsistentcy between expected number of vertices and number of vertices in geometry in HDF5 file");
    }
  }

  // Divide point range into equal blocks for each process
  std::pair<std::size_t, std::size_t> vertex_range
    = MPI::local_range(_mpi_comm, local_mesh_data.num_global_vertices);
  const std::size_t num_local_vertices
    = vertex_range.second - vertex_range.first;

  // Modify vertex data range for flat storage
  std::pair<std::size_t, std::size_t> vertex_data_range = vertex_range;
  if (coords_shape.size() == 1)
  {
    vertex_data_range.first *= gdim;
    vertex_data_range.second *= gdim;
  }

  // Read vertex data to temporary vector
  {
    std::vector<double> coordinates_data;
    coordinates_data.reserve(num_local_vertices*gdim);
    HDF5Interface::read_dataset(_hdf5_file_id, geometry_path, vertex_data_range,
                                coordinates_data);

    // Copy to boost::multi_array
    local_mesh_data.vertex_coordinates.resize(boost::extents[num_local_vertices][gdim]);
    std::copy(coordinates_data.begin(), coordinates_data.end(),
              local_mesh_data.vertex_coordinates.data());
  }

  // FIXME: explain this better - comment is cryptic
  // Fill vertex indices with values - not used in build_distributed_mesh
  local_mesh_data.vertex_indices.resize(num_local_vertices);
  for (std::size_t i = 0; i < local_mesh_data.vertex_coordinates.size(); ++i)
    local_mesh_data.vertex_indices[i] = vertex_range.first + i;

  t.stop();

  // Build distributed mesh
  // FIXME: Why is the mesh built int HDF5Utility? This should be in
  // the mesh code.
  if (MPI::size(_mpi_comm) == 1)
    HDF5Utility::build_local_mesh(input_mesh, local_mesh_data);
  else
    MeshPartitioning::build_distributed_mesh(input_mesh, local_mesh_data);

  // ---- Markers ----

  // Check if we have any domains
  for (std::size_t d = 0; d <= input_mesh.topology().dim(); ++d)
  {
    const std::string marker_dataset = mesh_name + "/domain_" + std::to_string(d);
    if (!has_dataset(marker_dataset))
      continue;

    auto _mesh = reference_to_no_delete_pointer(input_mesh);
    MeshValueCollection<std::size_t> mvc(_mesh, d);
    read_mesh_value_collection(mvc, marker_dataset);

    // Get mesh value collection data
    const std::map<std::pair<std::size_t, std::size_t>, std::size_t>&
      values = mvc.values();

    // Get mesh domain data and fill
    std::map<std::size_t, std::size_t>& markers
      = input_mesh.domains().markers(d);
    if (d != input_mesh.topology().dim())
    {
      input_mesh.init(d);
      for (auto entry = values.begin(); entry != values.end(); ++entry)
      {
        const Cell cell(input_mesh, entry->first.first);
        const std::size_t entity_index
          = cell.entities(d)[entry->first.second];
        markers[entity_index] = entry->second;
      }
    }
    else
    {
      // Special case for cells
      for (auto entry = values.begin(); entry != values.end(); ++entry)
        markers[entry->first.first] = entry->second;
    }
  }
}
//-----------------------------------------------------------------------------
bool HDF5File::has_dataset(const std::string dataset_name) const
{
  dolfin_assert(_hdf5_file_id > 0);
  return HDF5Interface::has_dataset(_hdf5_file_id, dataset_name);
}
//-----------------------------------------------------------------------------
HDF5Attribute HDF5File::attributes(const std::string dataset_name)
{
  dolfin_assert(_hdf5_file_id > 0);
  if (!has_dataset(dataset_name))
  {
    dolfin_error("HDF5File.cpp",
                 "accessing attributes",
                 "Dataset \"%s\" not found", dataset_name.c_str());
  }

  return HDF5Attribute(_hdf5_file_id, dataset_name);
}
//-----------------------------------------------------------------------------

#endif
