// Copyright (C) 2012 Chris N Richardson
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "HDF5File.h"
#include "HDF5Interface.h"
#include "HDF5Utility.h"
#include "cells.h"
#include <Eigen/Dense>
#include <boost/filesystem.hpp>
#include <boost/unordered_map.hpp>
#include <cstdio>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/log.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/function/Function.h>
#include <dolfinx/function/FunctionSpace.h>
#include <dolfinx/io/cells.h>
#include <dolfinx/la/PETScVector.h>
#include <dolfinx/la/utils.h>
#include <dolfinx/mesh/CoordinateDofs.h>
#include <dolfinx/mesh/DistributedMeshTools.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/MeshEntity.h>
#include <dolfinx/mesh/MeshFunction.h>
#include <dolfinx/mesh/MeshIterator.h>
#include <dolfinx/mesh/MeshValueCollection.h>
#include <dolfinx/mesh/PartitionData.h>
#include <dolfinx/mesh/Partitioning.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <petscvec.h>
#include <string>

using namespace dolfinx;
using namespace dolfinx::io;

//-----------------------------------------------------------------------------
HDF5File::HDF5File(MPI_Comm comm, const std::string filename,
                   const std::string file_mode)
    : _hdf5_file_id(0), _mpi_comm(comm)
{
  // See https://www.hdfgroup.org/hdf5-quest.html#gzero on zero for
  // _hdf5_file_id(0)

  // Create directory, if required (create on rank 0)
  if (MPI::rank(_mpi_comm.comm()) == 0)
  {
    const boost::filesystem::path path(filename);
    if (path.has_parent_path()
        && !boost::filesystem::is_directory(path.parent_path()))
    {
      boost::filesystem::create_directories(path.parent_path());
      if (!boost::filesystem::is_directory(path.parent_path()))
      {
        throw std::runtime_error("Could not create directory \""
                                 + path.parent_path().string() + "\"");
      }
    }
  }

  // Wait until directory has been created
  MPI::barrier(_mpi_comm.comm());

  // Open HDF5 file
  const bool mpi_io = MPI::size(_mpi_comm.comm()) > 1 ? true : false;
#ifndef H5_HAVE_PARALLEL
  if (mpi_io)
  {
    throw std::runtime_error(
        "Cannot open file. HDF5 has not been compiled with support for MPI");
  }
#endif
  _hdf5_file_id
      = HDF5Interface::open_file(_mpi_comm.comm(), filename, file_mode, mpi_io);
  assert(_hdf5_file_id > 0);
}
//-----------------------------------------------------------------------------
HDF5File::~HDF5File() { close(); }
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
  assert(_hdf5_file_id > 0);
  HDF5Interface::flush_file(_hdf5_file_id);
}
//-----------------------------------------------------------------------------
void HDF5File::write(const std::vector<Eigen::Vector3d>& points,
                     const std::string dataset_name)
{
  assert(points.size() > 0);
  assert(_hdf5_file_id > 0);

  // Pack data
  const std::size_t n = points.size();
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> x(n, 3);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < 3; ++j)
      x(i, j) = points[i][j];

  const bool mpi_io = MPI::size(_mpi_comm.comm()) > 1 ? true : false;
  write_data(dataset_name, x, mpi_io);
}
//-----------------------------------------------------------------------------
void HDF5File::write(const std::vector<double>& values,
                     const std::string dataset_name)
{
  std::vector<std::int64_t> global_size(
      1, MPI::sum(_mpi_comm.comm(), values.size()));
  const bool mpi_io = MPI::size(_mpi_comm.comm()) > 1 ? true : false;
  write_data(dataset_name, values, global_size, mpi_io);
}
//-----------------------------------------------------------------------------
void HDF5File::write(const la::PETScVector& x, const std::string dataset_name)
{
  assert(x.size() > 0);
  assert(_hdf5_file_id > 0);

  // Get all local data
  PetscErrorCode ierr;
  const PetscScalar* x_ptr = nullptr;
  ierr = VecGetArrayRead(x.vec(), &x_ptr);
  if (ierr != 0)
    la::petsc_error(ierr, __FILE__, "VecGetArrayRead");

  // Write data to file
  const auto local_range = x.local_range();
  const std::vector<std::int64_t> global_size(1, x.size());
  const bool mpi_io = MPI::size(_mpi_comm.comm()) > 1 ? true : false;
  HDF5Interface::write_dataset(_hdf5_file_id, dataset_name, x_ptr, local_range,
                               global_size, mpi_io, chunking);

  ierr = VecRestoreArrayRead(x.vec(), &x_ptr);
  if (ierr != 0)
    la::petsc_error(ierr, __FILE__, "VecRestoreArrayRead");

  // Add partitioning attribute to dataset
  std::vector<std::size_t> partitions;
  std::vector<std::size_t> local_range_first(1, local_range[0]);
  MPI::gather(_mpi_comm.comm(), local_range_first, partitions);
  MPI::broadcast(_mpi_comm.comm(), partitions);

  HDF5Interface::add_attribute(_hdf5_file_id, dataset_name, "partition",
                               partitions);
}
//-----------------------------------------------------------------------------
la::PETScVector HDF5File::read_vector(MPI_Comm comm,
                                      const std::string dataset_name,
                                      const bool use_partition_from_file) const
{
  assert(_hdf5_file_id > 0);

  // Check for data set exists
  if (!HDF5Interface::has_dataset(_hdf5_file_id, dataset_name))
  {
    throw std::runtime_error("Cannot read vector from file. "
                             "Data set with name \""
                             + dataset_name + "\" does not exist");
  }

  // Get dataset rank
  const std::size_t rank
      = HDF5Interface::dataset_rank(_hdf5_file_id, dataset_name);
  if (rank != 1)
    LOG(WARNING) << "Reading non-scalar data in HDF5 Vector";

  // Get global dataset size
  const std::vector<std::int64_t> data_shape
      = HDF5Interface::get_dataset_shape(_hdf5_file_id, dataset_name);

  // Check that rank is 1 or 2
  assert(data_shape.size() == 1
         or (data_shape.size() == 2 and data_shape[1] == 1));

  // Initialize vector
  std::array<std::int64_t, 2> range;
  if (use_partition_from_file)
  {
    // Get partition from file
    std::vector<std::size_t> partitions
        = HDF5Interface::get_attribute<std::vector<std::size_t>>(
            _hdf5_file_id, dataset_name, "partition");

    // Check that number of MPI processes matches partitioning
    if (MPI::size(_mpi_comm.comm()) != partitions.size())
    {
      throw std::runtime_error("Different number of processes used when "
                               "writing. Cannot restore partitioning");
    }

    // Add global size at end of partition vectors
    partitions.push_back(data_shape[0]);

    // Initialise vector
    const std::size_t process_num = MPI::rank(_mpi_comm.comm());
    range = {{(std::int64_t)partitions[process_num],
              (std::int64_t)partitions[process_num + 1]}};
  }
  else
  {
    range = MPI::local_range(comm, data_shape[0]);
  }
  Eigen::Array<std::int64_t, Eigen::Dynamic, 1> ghosts;
  la::PETScVector x(comm, range, ghosts, 1);

  // Get local range
  const std::array<std::int64_t, 2> local_range = x.local_range();

  // Read data from file
  std::vector<PetscScalar> data = HDF5Interface::read_dataset<PetscScalar>(
      _hdf5_file_id, dataset_name, local_range);

  // Set data
  PetscErrorCode ierr;
  PetscScalar* x_ptr = nullptr;
  ierr = VecGetArray(x.vec(), &x_ptr);
  if (ierr != 0)
    la::petsc_error(ierr, __FILE__, "VecGetArray");
  std::copy(data.begin(), data.end(), x_ptr);
  ierr = VecRestoreArray(x.vec(), &x_ptr);
  if (ierr != 0)
    la::petsc_error(ierr, __FILE__, "VecRestoreArray");

  return x;
}
//-----------------------------------------------------------------------------
void HDF5File::write(const mesh::Mesh& mesh, const std::string name)
{
  write(mesh, mesh.topology().dim(), name);
}
//-----------------------------------------------------------------------------
void HDF5File::write(const mesh::Mesh& mesh, int cell_dim,
                     const std::string name)
{
  // FIXME: break up this function

  common::Timer t0("HDF5: write mesh to file");

  const int tdim = mesh.topology().dim();
  const int gdim = mesh.geometry().dim();

  const bool mpi_io = MPI::size(_mpi_comm.comm()) > 1 ? true : false;
  assert(_hdf5_file_id > 0);

  mesh::CellType cell_type
      = mesh::cell_entity_type(mesh.topology().cell_type(), cell_dim);
  const graph::AdjacencyList<std::int32_t>& cell_points
      = mesh.coordinate_dofs().entity_points();

  // Allowing for higher order meshes to be written to file
  std::size_t num_cell_points;
  if (cell_dim == tdim)
    num_cell_points = cell_points.num_links(0);
  else
    num_cell_points = mesh::cell_num_entities(cell_type, 0);

  // ---------- Vertices (coordinates)
  {
    // Write vertex data to HDF5 file
    const std::string coord_dataset = name + "/coordinates";

    // Copy coordinates and indices and remove off-process values
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        _vertex_coords;
    _vertex_coords = mesh::DistributedMeshTools::reorder_by_global_indices(
        mesh.mpi_comm(), mesh.geometry().points(),
        mesh.geometry().global_indices());

    Eigen::Map<Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>> varray(
        _vertex_coords.data(), _vertex_coords.size() / 3, 3);

    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        vertex_coords(varray.rows(), gdim);
    vertex_coords = varray.block(0, 0, varray.rows(), gdim);

    // Write coordinates out from each process
    write_data(coord_dataset, vertex_coords, mpi_io);
  }

  // ---------- Topology
  {
    // Get/build topology data
    std::vector<std::int64_t> topological_data;

    const std::size_t degree = mesh.degree();

    if (degree > 1)
    {
      if (cell_dim != tdim)
      {
        throw std::runtime_error("Cannot create topology data for mesh. "
                                 "Can only create mesh of cells");
      }

      const auto& global_points = mesh.geometry().global_indices();

      // Adjust num_nodes_per_cell to appropriate size
      int num_nodes_per_cell = cell_points.num_links(0);
      topological_data.reserve(num_nodes_per_cell * mesh.num_entities(tdim));

      int num_nodes = mesh.coordinate_dofs().entity_points().num_links(0);
      const std::vector<std::uint8_t> perm
          = io::cells::dolfin_to_vtk(mesh.topology().cell_type(), num_nodes);

      for (std::int32_t c = 0; c < mesh.num_entities(tdim); ++c)
      {
        auto points = cell_points.links(c);
        for (std::int32_t i = 0; i < num_nodes_per_cell; ++i)
        {
          topological_data.push_back(global_points[points[perm[i]]]);
        }
      }
    }
    else
    {
      topological_data.reserve(mesh.num_entities(cell_dim) * (num_cell_points));
      auto map = mesh.topology().index_map(0);
      assert(map);
      const std::vector<std::int64_t> global_vertices
          = map->global_indices(false);

      // Permutation to VTK ordering
      int num_nodes = mesh.coordinate_dofs().entity_points().num_links(0);
      const std::vector<std::uint8_t> perm
          = io::cells::dolfin_to_vtk(mesh.topology().cell_type(), num_nodes);

      if (cell_dim == tdim or !mpi_io)
      {
        // Usual case, with cell output, and/or none shared with another
        // process.
        if (cell_dim == 0)
        {
          for (auto& v : mesh::MeshRange(mesh, 0))
            topological_data.push_back(global_vertices[v.index()]);
        }
        else
        {
          const int num_vertices = mesh::cell_num_entities(
              mesh::cell_entity_type(mesh.topology().cell_type(), cell_dim), 0);
          for (auto& c : mesh::MeshRange(mesh, cell_dim))
          {
            for (int i = 0; i < num_vertices; ++i)
            {
              const int local_idx = c.entities(0)[perm[i]];
              topological_data.push_back(global_vertices[local_idx]);
            }
          }
        }
      }
      else
      {
        // Drop duplicate topology for shared entities of less than mesh
        // dimension

        const int mpi_rank = MPI::rank(_mpi_comm.comm());
        const std::map<std::int32_t, std::set<std::int32_t>> shared_entities
            = mesh.topology().index_map(cell_dim)->compute_shared_indices();

        std::set<int> non_local_entities;
        if (mesh.topology().index_map(tdim)->num_ghosts() == 0)
        {
          // No ghost cells - exclude shared entities which are on lower
          // rank processes
          for (auto sh = shared_entities.begin(); sh != shared_entities.end();
               ++sh)
          {
            const int lowest_proc = *(sh->second.begin());
            if (lowest_proc < mpi_rank)
              non_local_entities.insert(sh->first);
          }
        }
        else
        {
          // Iterate through ghost cells, adding non-ghost entities
          // which are in lower rank process cells to a set for
          // exclusion from output
          const Eigen::Array<int, Eigen::Dynamic, 1>& cell_owners
              = mesh.topology().index_map(tdim)->ghost_owners();
          const std::int32_t ghost_offset_c
              = mesh.topology().index_map(tdim)->size_local();
          const std::int32_t ghost_offset_e
              = mesh.topology().index_map(cell_dim)->size_local();
          for (auto& c :
               mesh::MeshRange(mesh, tdim, mesh::MeshRangeType::GHOST))
          {
            assert(c.index() >= ghost_offset_c);
            const int cell_owner = cell_owners[c.index() - ghost_offset_c];
            for (auto& e : mesh::EntityRange(c, cell_dim))
            {
              const bool not_ghost = e.index() < ghost_offset_e;
              if (not_ghost and cell_owner < mpi_rank)
                non_local_entities.insert(e.index());
            }
          }
        }

        if (cell_dim == 0)
        {
          // Special case for mesh of points
          for (auto& v : mesh::MeshRange(mesh, 0))
          {
            if (non_local_entities.find(v.index()) == non_local_entities.end())
              topological_data.push_back(global_vertices[v.index()]);
          }
        }
        else
        {
          const int num_vertices = mesh::cell_num_entities(
              mesh::cell_entity_type(mesh.topology().cell_type(), cell_dim), 0);
          for (auto& ent : mesh::MeshRange(mesh, cell_dim))
          {
            // If not excluded, add to topology
            if (non_local_entities.find(ent.index())
                == non_local_entities.end())
            {
              for (int i = 0; i < num_vertices; ++i)
              {
                const int local_idx = ent.entities(0)[perm[i]];
                topological_data.push_back(global_vertices[local_idx]);
              }
            }
          }
        }
      }
    }

    // Write topology data
    const std::string topology_dataset = name + "/topology";
    std::vector<std::int64_t> global_size(2);
    global_size[0]
        = MPI::sum(_mpi_comm.comm(), topological_data.size() / num_cell_points);
    global_size[1] = num_cell_points;

    const std::int64_t num_cells = mpi_io ? mesh.num_entities_global(cell_dim)
                                          : mesh.num_entities(cell_dim);
    assert(global_size[0] == num_cells);
    write_data(topology_dataset, topological_data, global_size, mpi_io);

    // For cells, write the global cell index
    if (cell_dim == mesh.topology().dim())
    {
      const std::string cell_index_dataset = name + "/cell_indices";
      global_size.pop_back();

      auto map = mesh.topology().index_map(cell_dim);
      assert(map);
      const std::vector<std::int64_t> cell_index_ref
          = map->global_indices(false);

      const std::vector<std::int64_t> cells(
          cell_index_ref.begin(),
          cell_index_ref.begin()
              + mesh.topology().index_map(cell_dim)->size_local());

      write_data(cell_index_dataset, cells, global_size, mpi_io);
    }

    // Add cell type attribute
    HDF5Interface::add_attribute(_hdf5_file_id, topology_dataset, "celltype",
                                 mesh::to_string(cell_type));

    // Add partitioning attribute to dataset
    std::vector<std::size_t> partitions;
    const std::size_t topology_offset = MPI::global_offset(
        _mpi_comm.comm(), topological_data.size() / num_cell_points, true);

    std::vector<std::size_t> topology_offset_tmp(1, topology_offset);
    MPI::gather(_mpi_comm.comm(), topology_offset_tmp, partitions);
    MPI::broadcast(_mpi_comm.comm(), partitions);

    HDF5Interface::add_attribute(_hdf5_file_id, topology_dataset, "partition",
                                 partitions);
  }
}
//-----------------------------------------------------------------------------
void HDF5File::write(const mesh::MeshFunction<std::size_t>& meshfunction,
                     const std::string name)
{
  write_mesh_function(meshfunction, name);
}
//-----------------------------------------------------------------------------
mesh::MeshFunction<std::size_t>
HDF5File::read_mf_size_t(std::shared_ptr<const mesh::Mesh> mesh,
                         const std::string name) const
{
  return read_mesh_function<std::size_t>(mesh, name);
}
//-----------------------------------------------------------------------------
void HDF5File::write(const mesh::MeshFunction<int>& meshfunction,
                     const std::string name)
{
  write_mesh_function(meshfunction, name);
}
//-----------------------------------------------------------------------------
mesh::MeshFunction<int>
HDF5File::read_mf_int(std::shared_ptr<const mesh::Mesh> mesh,
                      const std::string name) const
{
  return read_mesh_function<int>(mesh, name);
}
//-----------------------------------------------------------------------------
void HDF5File::write(const mesh::MeshFunction<double>& meshfunction,
                     const std::string name)
{
  write_mesh_function(meshfunction, name);
}
//-----------------------------------------------------------------------------
mesh::MeshFunction<double>
HDF5File::read_mf_double(std::shared_ptr<const mesh::Mesh> mesh,
                         const std::string name) const
{
  return read_mesh_function<double>(mesh, name);
}
//-----------------------------------------------------------------------------
template <typename T>
mesh::MeshFunction<T>
HDF5File::read_mesh_function(std::shared_ptr<const mesh::Mesh> mesh,
                             const std::string mesh_name) const
{
  assert(mesh);
  assert(_hdf5_file_id > 0);

  const std::string topology_name = mesh_name + "/topology";

  if (!HDF5Interface::has_dataset(_hdf5_file_id, topology_name))
  {
    throw std::runtime_error("Cannot read topology dataset."
                             "Dataset \""
                             + topology_name + "\" not found");
  }

  // Look for Coordinates dataset - but not used
  const std::string coordinates_name = mesh_name + "/coordinates";
  if (!HDF5Interface::has_dataset(_hdf5_file_id, coordinates_name))
  {
    throw std::runtime_error("Cannot read coordinates dataset. "
                             "Dataset \""
                             + coordinates_name + "\" not found");
  }

  // Look for Values dataset
  const std::string values_name = mesh_name + "/values";
  if (!HDF5Interface::has_dataset(_hdf5_file_id, values_name))
  {
    throw std::runtime_error("Cannot read coordinates dataset. "
                             "Dataset \""
                             + values_name + "\" not found");
  }

  // --- Topology ---

  // Discover size of topology dataset
  const std::vector<std::int64_t> topology_shape
      = HDF5Interface::get_dataset_shape(_hdf5_file_id, topology_name);

  // Some consistency checks

  // FIXME: Will break for other non-simplex
  const std::int64_t num_global_cells = topology_shape[0];
  const std::size_t vertices_per_cell = topology_shape[1];
  const std::size_t dim = vertices_per_cell - 1;

  if (num_global_cells != mesh->num_entities_global(dim))
  {
    throw std::runtime_error(
        "Cannot read meshfunction topology. Mesh dimension mismatch");
  }

  // Divide up cells ~equally between processes
  const std::array<std::int64_t, 2> cell_range
      = MPI::local_range(_mpi_comm.comm(), num_global_cells);
  const std::size_t num_read_cells = cell_range[1] - cell_range[0];

  // Read a block of cells
  std::vector<std::size_t> topology_data
      = HDF5Interface::read_dataset<std::size_t>(_hdf5_file_id, topology_name,
                                                 cell_range);

  // Wrap data as 2D array
  Eigen::Map<Eigen::Array<std::size_t, Eigen::Dynamic, Eigen::Dynamic,
                          Eigen::RowMajor>>
      topology_array(topology_data.data(), num_read_cells, vertices_per_cell);

  std::vector<T> value_data
      = HDF5Interface::read_dataset<T>(_hdf5_file_id, values_name, cell_range);

  // Now send the read data to each process on the basis of the first
  // vertex of the entity, since we do not know the global_index
  const std::size_t num_processes = MPI::size(_mpi_comm.comm());
  const std::size_t max_vertex = mesh->num_entities_global(0);

  std::vector<std::vector<std::size_t>> send_topology(num_processes);
  std::vector<std::vector<T>> send_values(num_processes);
  for (Eigen::Index i = 0; i < topology_array.rows(); ++i)
  {
    std::sort(topology_array.row(i).data(),
              topology_array.row(i).data() + topology_array.row(i).cols());

    // Use first vertex to decide where to send this data
    assert(topology_array.row(i).cols() > 0);
    const std::size_t send_to_process
        = MPI::index_owner(_mpi_comm.comm(), topology_array(i, 0), max_vertex);

    send_topology[send_to_process].insert(
        send_topology[send_to_process].end(), topology_array.row(i).data(),
        topology_array.row(i).data() + topology_array.row(i).cols());

    send_values[send_to_process].push_back(value_data[i]);
  }

  std::vector<std::vector<std::size_t>> receive_topology(num_processes);
  std::vector<std::vector<T>> receive_values(num_processes);
  MPI::all_to_all(_mpi_comm.comm(), send_topology, receive_topology);
  MPI::all_to_all(_mpi_comm.comm(), send_values, receive_values);

  // Generate requests for data from remote processes, based on the
  // first vertex of the mesh::MeshEntities which belong on this process
  // Send our process number, and our local index, so it can come back
  // directly to the right place
  std::vector<std::vector<std::size_t>> send_requests(num_processes);
  const std::size_t process_number = MPI::rank(_mpi_comm.comm());

  auto map = mesh->topology().index_map(0);
  assert(map);
  const std::vector<std::int64_t> global_indices = map->global_indices(false);

  for (auto& cell : mesh::MeshRange(*mesh, dim, mesh::MeshRangeType::ALL))
  {
    std::vector<std::size_t> cell_topology;
    for (auto& v : mesh::EntityRange(cell, 0))
      cell_topology.push_back(global_indices[v.index()]);
    std::sort(cell_topology.begin(), cell_topology.end());

    // Use first vertex to decide where to send this request
    std::size_t send_to_process
        = MPI::index_owner(_mpi_comm.comm(), cell_topology.front(), max_vertex);
    // Map to this process and local index by appending to send data
    cell_topology.push_back(cell.index());
    cell_topology.push_back(process_number);
    send_requests[send_to_process].insert(send_requests[send_to_process].end(),
                                          cell_topology.begin(),
                                          cell_topology.end());
  }

  std::vector<std::vector<std::size_t>> receive_requests(num_processes);
  MPI::all_to_all(_mpi_comm.comm(), send_requests, receive_requests);

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
    assert(receive_values[i].size() * vertices_per_cell
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
      assert(find_cell != cell_to_data.end());
      send_values[send_to_proc].push_back(find_cell->second);
      send_topology[send_to_proc].push_back(remote_index);
    }
  }

  MPI::all_to_all(_mpi_comm.comm(), send_topology, receive_topology);
  MPI::all_to_all(_mpi_comm.comm(), send_values, receive_values);

  // At this point, receive_topology should only list the local indices
  // and received values should have the appropriate values for each
  mesh::MeshFunction<T> mf(mesh, dim, 0);

  // Get reference to mesh function data array
  Eigen::Array<T, Eigen::Dynamic, 1>& mf_values = mf.values();

  for (std::size_t i = 0; i < receive_values.size(); ++i)
  {
    assert(receive_values[i].size() == receive_topology[i].size());
    for (std::size_t j = 0; j < receive_values[i].size(); ++j)
      mf_values[receive_topology[i][j]] = receive_values[i][j];
  }

  return mf;
}
//-----------------------------------------------------------------------------
template <typename T>
void HDF5File::write_mesh_function(const mesh::MeshFunction<T>& meshfunction,
                                   const std::string name)
{
  if (meshfunction.values().size() == 0)
    throw std::runtime_error("Cannot save empty mesh::MeshFunction.");

  const mesh::Mesh& mesh = *meshfunction.mesh();
  const int cell_dim = meshfunction.dim();

  // Write a mesh for the mesh::MeshFunction - this will also globally
  // number the entities if needed
  write(mesh, cell_dim, name);

  // Storage for output values
  std::vector<T> data_values;

  if (cell_dim == mesh.topology().dim() || MPI::size(_mpi_comm.comm()) == 1)
  {
    // No duplicates - ignore ghost cells if present
    data_values.assign(meshfunction.values().data(),
                       meshfunction.values().data()
                           + mesh.topology().index_map(cell_dim)->size_local());
  }
  else
  {
    // In parallel and not CellFunction
    data_values.reserve(mesh.num_entities(cell_dim));

    // Drop duplicate data
    const int tdim = mesh.topology().dim();
    const int mpi_rank = MPI::rank(_mpi_comm.comm());
    const std::map<std::int32_t, std::set<std::int32_t>> shared_entities
        = mesh.topology().index_map(cell_dim)->compute_shared_indices();

    std::set<int> non_local_entities;
    if (mesh.topology().index_map(tdim)->num_ghosts() == 0)
    {
      // No ghost cells
      // Exclude shared entities which are on lower rank processes
      for (auto sh = shared_entities.begin(); sh != shared_entities.end(); ++sh)
      {
        const int lowest_proc = *(sh->second.begin());
        if (lowest_proc < mpi_rank)
          non_local_entities.insert(sh->first);
      }
    }
    else
    {
      // Iterate through ghost cells, adding non-ghost entities which are
      // shared from lower rank process cells to a set for exclusion
      // from output
      const Eigen::Array<int, Eigen::Dynamic, 1>& cell_owners
          = mesh.topology().index_map(tdim)->ghost_owners();
      const std::int32_t ghost_offset_c
          = mesh.topology().index_map(tdim)->size_local();
      const std::int32_t ghost_offset_e
          = mesh.topology().index_map(cell_dim)->size_local();
      for (auto& c : mesh::MeshRange(mesh, tdim, mesh::MeshRangeType::GHOST))
      {
        assert(c.index() >= ghost_offset_c);
        const int cell_owner = cell_owners[c.index() - ghost_offset_c];
        for (auto& e : mesh::EntityRange(c, cell_dim))
        {
          const bool not_ghost = e.index() < ghost_offset_e;
          if (not_ghost and cell_owner < mpi_rank)
            non_local_entities.insert(e.index());
        }
      }
    }

    // Get reference to mesh function data array
    const Eigen::Array<T, Eigen::Dynamic, 1>& mf_values = meshfunction.values();
    for (auto& e : mesh::MeshRange(mesh, cell_dim))
    {
      if (non_local_entities.find(e.index()) == non_local_entities.end())
        data_values.push_back(mf_values[e.index()]);
    }
  }

  // Write values to HDF5
  std::vector<std::int64_t> global_size(
      1, MPI::sum(_mpi_comm.comm(), data_values.size()));
  const bool mpi_io = MPI::size(_mpi_comm.comm()) > 1 ? true : false;
  write_data(name + "/values", data_values, global_size, mpi_io);
}
//-----------------------------------------------------------------------------
void HDF5File::write(const function::Function& u, const std::string name,
                     double timestamp)
{
  assert(_hdf5_file_id > 0);
  if (!HDF5Interface::has_dataset(_hdf5_file_id, name))
  {
    write(u, name);
    const std::size_t vec_count = 1;

    // if (!HDF5Interface::has_dataset(_hdf5_file_id, name))
    //  throw std::runtime_error("Setting attribute on dataset. Dataset does
    //  not exist");

    // FIXME: is this check required?
    if (HDF5Interface::has_attribute(_hdf5_file_id, name, "count"))
      HDF5Interface::delete_attribute(_hdf5_file_id, name, "count");
    HDF5Interface::add_attribute(_hdf5_file_id, name, "count", vec_count);

    const std::string vec_name = name + "/vector_0";
    // FIXME: is this check required?
    if (HDF5Interface::has_attribute(_hdf5_file_id, vec_name, "timestamp"))
      HDF5Interface::delete_attribute(_hdf5_file_id, vec_name, "timestamp");
    HDF5Interface::add_attribute(_hdf5_file_id, vec_name, "timestamp",
                                 timestamp);
  }
  else
  {
    // HDF5Attribute attr(_hdf5_file_id, name);

    if (!HDF5Interface::has_attribute(_hdf5_file_id, name, "count"))
    {
      throw std::runtime_error(
          "Function dataset does not contain a series 'count' attribute");
    }

    // Get count of vectors in dataset, and increment
    std::size_t vec_count = HDF5Interface::get_attribute<std::size_t>(
        _hdf5_file_id, name, "count");

    std::string vec_name = name + "/vector_" + std::to_string(vec_count);
    ++vec_count;
    if (HDF5Interface::has_attribute(_hdf5_file_id, name, "count"))
      HDF5Interface::delete_attribute(_hdf5_file_id, name, "count");
    HDF5Interface::add_attribute(_hdf5_file_id, name, "count", vec_count);

    // Write new vector and save timestamp
    write(u.vector(), vec_name);
    if (HDF5Interface::has_attribute(_hdf5_file_id, vec_name, "timestamp"))
      HDF5Interface::delete_attribute(_hdf5_file_id, vec_name, "timestamp");
    HDF5Interface::add_attribute(_hdf5_file_id, vec_name, "timestamp",
                                 timestamp);
  }
}
//-----------------------------------------------------------------------------
void HDF5File::write(const function::Function& u, const std::string name)
{
  common::Timer t0("HDF5: write function::Function");
  assert(_hdf5_file_id > 0);

  // Get mesh and dofmap
  assert(u.function_space()->mesh());
  const mesh::Mesh& mesh = *u.function_space()->mesh();

  assert(u.function_space()->dofmap());
  const fem::DofMap& dofmap = *u.function_space()->dofmap();

  // FIXME:
  // Possibly sort cell_dofs into global cell order before writing?

  // Save data in compressed format with an index to mark out
  // the start of each row

  const std::size_t tdim = mesh.topology().dim();
  std::vector<std::int32_t> cell_dofs;
  std::vector<std::size_t> x_cell_dofs;
  const std::size_t n_cells = mesh.topology().index_map(tdim)->size_local();
  x_cell_dofs.reserve(n_cells);

  Eigen::Array<std::int64_t, Eigen::Dynamic, 1> local_to_global_map
      = dofmap.index_map->indices(true);

  for (std::size_t i = 0; i != n_cells; ++i)
  {
    x_cell_dofs.push_back(cell_dofs.size());
    auto cell_dofs_i = dofmap.cell_dofs(i);
    for (Eigen::Index j = 0; j < cell_dofs_i.size(); ++j)
    {
      auto p = cell_dofs_i[j];
      assert(p < (std::int32_t)local_to_global_map.size());
      cell_dofs.push_back(local_to_global_map[p]);
    }
  }

  // Add offset to CSR index to be seamless in parallel
  const std::size_t offset
      = MPI::global_offset(_mpi_comm.comm(), cell_dofs.size(), true);
  for (auto& x : x_cell_dofs)
    x += offset;

  const bool mpi_io = MPI::size(_mpi_comm.comm()) > 1 ? true : false;

  // Save DOFs on each cell
  std::vector<std::int64_t> global_size(
      1, MPI::sum(_mpi_comm.comm(), cell_dofs.size()));
  write_data(name + "/cell_dofs", cell_dofs, global_size, mpi_io);
  if (MPI::rank(_mpi_comm.comm()) == MPI::size(_mpi_comm.comm()) - 1)
    x_cell_dofs.push_back(global_size[0]);
  global_size[0] = mesh.num_entities_global(tdim) + 1;
  write_data(name + "/x_cell_dofs", x_cell_dofs, global_size, mpi_io);

  // Save cell ordering - copy to local vector and cut off ghosts
  auto map = mesh.topology().index_map(tdim);
  assert(map);
  const std::vector<std::int64_t> global_indices = map->global_indices(false);
  std::vector<std::size_t> cells(global_indices.begin(),
                                 global_indices.begin() + n_cells);

  global_size[0] = mesh.num_entities_global(tdim);
  write_data(name + "/cells", cells, global_size, mpi_io);

  HDF5Interface::add_attribute(_hdf5_file_id, name, "signature",
                               u.function_space()->element()->signature());

  // Save vector
  write(u.vector(), name + "/vector_0");
}
//-----------------------------------------------------------------------------
function::Function
HDF5File::read(std::shared_ptr<const function::FunctionSpace> V,
               const std::string name) const
{
  common::Timer t0("HDF5: read function::Function");
  assert(_hdf5_file_id > 0);

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
    throw std::runtime_error("Cannot read function from file"
                             "Group with name \""
                             + basename + "\" does not exist");
  }

  if (!HDF5Interface::has_dataset(_hdf5_file_id, cells_dataset_name))
  {
    throw std::runtime_error("Cannot read function from file"
                             "Dataset with name \""
                             + cells_dataset_name + "\" does not exist");
  }

  if (!HDF5Interface::has_dataset(_hdf5_file_id, cell_dofs_dataset_name))
  {
    throw std::runtime_error("Cannot read function from file"
                             "Dataset with name \""
                             + cell_dofs_dataset_name + "\" does not exist");
  }

  if (!HDF5Interface::has_dataset(_hdf5_file_id, x_cell_dofs_dataset_name))
  {
    throw std::runtime_error("Cannot read function from file"
                             "Dataset with name \""
                             + x_cell_dofs_dataset_name + "\" does not exist");
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
      throw std::runtime_error("Cannot read function from file"
                               "Dataset with name \""
                               + vector_dataset_name + "\" does not exist");
    }
  }

  // Create function
  function::Function u(V);

  // Get existing mesh and dofmap - these should be pre-existing
  // and set up by user when defining the function::Function
  assert(u.function_space()->mesh());
  const mesh::Mesh& mesh = *u.function_space()->mesh();
  assert(u.function_space()->dofmap());
  const fem::DofMap& dofmap = *u.function_space()->dofmap();

  // Get dimension of dataset
  const std::vector<std::int64_t> dataset_shape
      = HDF5Interface::get_dataset_shape(_hdf5_file_id, cells_dataset_name);
  const std::int64_t num_global_cells = dataset_shape[0];
  if (mesh.num_entities_global(mesh.topology().dim()) != num_global_cells)
  {
    throw std::runtime_error(
        "Cannot read function::Function from file. Number of "
        "global cells does not match.");
  }

  // Divide cells equally between processes
  const std::array<std::int64_t, 2> cell_range
      = MPI::local_range(_mpi_comm.comm(), num_global_cells);

  // Read cells
  std::vector<std::size_t> input_cells
      = HDF5Interface::read_dataset<std::size_t>(
          _hdf5_file_id, cells_dataset_name, cell_range);

  // Overlap reads of DOF indices, to get full range on each process
  std::vector<std::int64_t> x_cell_dofs
      = HDF5Interface::read_dataset<std::int64_t>(
          _hdf5_file_id, x_cell_dofs_dataset_name,
          {{cell_range[0], cell_range[1] + 1}});

  // Read cell-DOF maps
  std::vector<std::int64_t> input_cell_dofs
      = HDF5Interface::read_dataset<std::int64_t>(
          _hdf5_file_id, cell_dofs_dataset_name,
          {{x_cell_dofs.front(), x_cell_dofs.back()}});

  la::PETScVector& x = u.vector();

  const std::vector<std::int64_t> vector_shape
      = HDF5Interface::get_dataset_shape(_hdf5_file_id, vector_dataset_name);
  const std::int64_t num_global_dofs = vector_shape[0];
  assert(num_global_dofs == x.size());
  const std::array<std::int64_t, 2> input_vector_range
      = MPI::local_range(_mpi_comm.comm(), vector_shape[0]);

  std::vector<PetscScalar> input_values
      = HDF5Interface::read_dataset<PetscScalar>(
          _hdf5_file_id, vector_dataset_name, input_vector_range);

  // HDF5Utility::set_local_vector_values(_mpi_comm.comm(), x, mesh,
  // input_cells,
  //                                     input_cell_dofs, x_cell_dofs,
  //                                     input_values, input_vector_range,
  //                                     dofmap);
  HDF5Utility::set_local_vector_values(
      _mpi_comm.comm(), x, mesh, input_cells, input_cell_dofs, x_cell_dofs,
      input_values, input_vector_range, dofmap);

  return u;
}
//-----------------------------------------------------------------------------
void HDF5File::write(const mesh::MeshValueCollection<std::size_t>& mesh_values,
                     const std::string name)
{
  write_mesh_value_collection(mesh_values, name);
}
//-----------------------------------------------------------------------------
mesh::MeshValueCollection<std::size_t>
HDF5File::read_mvc_size_t(std::shared_ptr<const mesh::Mesh> mesh,
                          const std::string name) const
{
  return read_mesh_value_collection<std::size_t>(mesh, name);
}
//-----------------------------------------------------------------------------
void HDF5File::write(const mesh::MeshValueCollection<double>& mesh_values,
                     const std::string name)
{
  write_mesh_value_collection(mesh_values, name);
}
//-----------------------------------------------------------------------------
mesh::MeshValueCollection<double>
HDF5File::read_mvc_double(std::shared_ptr<const mesh::Mesh> mesh,
                          const std::string name) const
{
  return read_mesh_value_collection<double>(mesh, name);
}
//-----------------------------------------------------------------------------
void HDF5File::write(const mesh::MeshValueCollection<bool>& mesh_values,
                     const std::string name)
{
  // HDF5 does not implement bool, use int and copy

  mesh::MeshValueCollection<int> mvc_int(mesh_values.mesh(), mesh_values.dim());
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
mesh::MeshValueCollection<bool>
HDF5File::read_mvc_bool(std::shared_ptr<const mesh::Mesh> mesh,
                        const std::string name) const
{
  // HDF5 does not implement bool, use int and copy

  auto mvc_int = read_mesh_value_collection<int>(mesh, name);

  const std::map<std::pair<std::size_t, std::size_t>, int>& values
      = mvc_int.values();
  mesh::MeshValueCollection<bool> mvc(mesh, mvc_int.dim());
  for (auto mesh_value_it = values.begin(); mesh_value_it != values.end();
       ++mesh_value_it)
  {
    mvc.set_value(mesh_value_it->first.first, mesh_value_it->first.second,
                  (mesh_value_it->second != 0));
  }

  return mvc;
}
//-----------------------------------------------------------------------------
template <typename T>
void HDF5File::write_mesh_value_collection(
    const mesh::MeshValueCollection<T>& mesh_values, const std::string name)
{
  assert(_hdf5_file_id > 0);

  const std::size_t dim = mesh_values.dim();
  std::shared_ptr<const mesh::Mesh> mesh = mesh_values.mesh();

  const std::map<std::pair<std::size_t, std::size_t>, T>& values
      = mesh_values.values();

  const mesh::CellType entity_type
      = mesh::cell_entity_type(mesh->topology().cell_type(), dim);
  const std::size_t num_vertices_per_entity
      = (dim == 0) ? 1 : mesh::num_cell_vertices(entity_type);

  std::vector<std::size_t> topology;
  std::vector<T> value_data;
  topology.reserve(values.size() * num_vertices_per_entity);
  value_data.reserve(values.size());

  const std::size_t tdim = mesh->topology().dim();
  mesh->create_connectivity(tdim, dim);

  auto map = mesh->topology().index_map(0);
  assert(map);
  const std::vector<std::int64_t> global_indices = map->global_indices(false);

  for (auto& p : values)
  {
    // mesh::MeshEntity cell = mesh::Cell(*mesh, p.first.first);
    mesh::MeshEntity cell(*mesh, tdim, p.first.first);
    if (dim != tdim)
    {
      const int entity_local_idx = cell.entities(dim)[p.first.second];
      cell = mesh::MeshEntity(*mesh, dim, entity_local_idx);
    }
    for (auto& v : mesh::EntityRange(cell, 0))
      topology.push_back(global_indices[v.index()]);
    value_data.push_back(p.second);
  }

  const bool mpi_io = MPI::size(_mpi_comm.comm()) > 1 ? true : false;
  std::vector<std::int64_t> global_size(2);

  global_size[0] = MPI::sum(_mpi_comm.comm(), values.size());
  global_size[1] = num_vertices_per_entity;

  write_data(name + "/topology", topology, global_size, mpi_io);

  global_size[1] = 1;
  write_data(name + "/values", value_data, global_size, mpi_io);
  HDF5Interface::add_attribute(_hdf5_file_id, name, "dimension",
                               mesh_values.dim());
}
//-----------------------------------------------------------------------------
template <typename T>
mesh::MeshValueCollection<T>
HDF5File::read_mesh_value_collection(std::shared_ptr<const mesh::Mesh> mesh,
                                     const std::string name) const
{
  common::Timer t1("HDF5: read mesh value collection");
  assert(_hdf5_file_id > 0);

  if (!HDF5Interface::has_group(_hdf5_file_id, name))
  {
    throw std::runtime_error("Cannot open MeshValueCollection dataset. "
                             "Group \""
                             + name + "\" not found in file");
  }

  std::size_t dim = HDF5Interface::get_attribute<std::size_t>(
      _hdf5_file_id, name, "dimension");
  assert(mesh);
  const mesh::CellType entity_type
      = mesh::cell_entity_type(mesh->topology().cell_type(), dim);
  const std::size_t num_verts_per_entity
      = mesh::cell_num_entities(entity_type, 0);

  const std::string values_name = name + "/values";
  const std::string topology_name = name + "/topology";

  if (!HDF5Interface::has_dataset(_hdf5_file_id, values_name))
  {
    throw std::runtime_error("Cannot open MeshValueCollection dataset. "
                             "Group \""
                             + values_name + "\" not found in file");
  }

  if (!HDF5Interface::has_dataset(_hdf5_file_id, topology_name))
  {
    throw std::runtime_error("Cannot open MeshValueCollection dataset. "
                             "Group \""
                             + topology_name + "\" not found in file");
  }

  // Check both datasets have the same number of entries
  const std::vector<std::int64_t> values_shape
      = HDF5Interface::get_dataset_shape(_hdf5_file_id, values_name);
  const std::vector<std::int64_t> topology_shape
      = HDF5Interface::get_dataset_shape(_hdf5_file_id, topology_name);
  assert(values_shape[0] == topology_shape[0]);

  // Divide range between processes
  const std::array<std::int64_t, 2> data_range
      = MPI::local_range(_mpi_comm.comm(), values_shape[0]);

  // Read local range of values and entities
  std::vector<T> values_data
      = HDF5Interface::read_dataset<T>(_hdf5_file_id, values_name, data_range);
  std::vector<std::size_t> topology_data
      = HDF5Interface::read_dataset<std::size_t>(_hdf5_file_id, topology_name,
                                                 data_range);

  /// Basically need to tabulate all entities by vertex, and get their
  /// local index, transmit them to a 'sorting' host.  Also send the
  /// read data to the 'sorting' hosts.

  // Ensure the mesh dimension is initialised
  mesh->create_entities(dim);
  std::size_t global_vertex_range = mesh->num_entities_global(0);
  std::vector<std::size_t> v(num_verts_per_entity);
  const std::size_t num_processes = MPI::size(_mpi_comm.comm());

  // Calculate map from entity vertices to {process, local index}
  std::map<std::vector<std::size_t>, std::vector<std::size_t>> entity_map;

  std::vector<std::vector<std::size_t>> send_entities(num_processes);
  std::vector<std::vector<std::size_t>> recv_entities(num_processes);

  auto map = mesh->topology().index_map(0);
  assert(map);
  const std::vector<std::int64_t> global_indices = map->global_indices(false);

  for (auto& m : mesh::MeshRange(*mesh, dim))
  {
    if (dim == 0)
      v[0] = global_indices[m.index()];
    else
    {
      v.clear();
      for (auto& vtx : mesh::EntityRange(m, 0))
        v.push_back(global_indices[vtx.index()]);
      std::sort(v.begin(), v.end());
    }

    std::size_t dest
        = MPI::index_owner(_mpi_comm.comm(), v[0], global_vertex_range);
    send_entities[dest].push_back(m.index());
    send_entities[dest].insert(send_entities[dest].end(), v.begin(), v.end());
  }

  MPI::all_to_all(_mpi_comm.comm(), send_entities, recv_entities);

  for (std::size_t i = 0; i != num_processes; ++i)
  {
    for (auto it = recv_entities[i].cbegin(); it != recv_entities[i].cend();
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

  // Send data from mesh::MeshValueCollection to sorting process

  std::vector<std::vector<T>> send_data(num_processes);
  std::vector<std::vector<T>> recv_data(num_processes);
  // Reset send/recv arrays
  send_entities = std::vector<std::vector<std::size_t>>(num_processes);
  recv_entities = std::vector<std::vector<std::size_t>>(num_processes);

  std::size_t i = 0;
  for (auto it = topology_data.begin(); it != topology_data.end();
       it += num_verts_per_entity)
  {
    std::partial_sort_copy(it, it + num_verts_per_entity, v.begin(), v.end());
    std::size_t dest
        = MPI::index_owner(_mpi_comm.comm(), v[0], global_vertex_range);
    send_entities[dest].insert(send_entities[dest].end(), v.begin(), v.end());
    send_data[dest].push_back(values_data[i]);
    ++i;
  }

  MPI::all_to_all(_mpi_comm.comm(), send_entities, recv_entities);
  MPI::all_to_all(_mpi_comm.comm(), send_data, recv_data);

  // Reset send arrays
  send_data = std::vector<std::vector<T>>(num_processes);
  send_entities = std::vector<std::vector<std::size_t>>(num_processes);

  // Locate entity in map, and send back to data to owning processes
  for (std::size_t i = 0; i != num_processes; ++i)
  {
    assert(recv_data[i].size() * num_verts_per_entity
           == recv_entities[i].size());

    for (std::size_t j = 0; j != recv_data[i].size(); ++j)
    {
      auto it = recv_entities[i].begin() + j * num_verts_per_entity;
      std::copy(it, it + num_verts_per_entity, v.begin());
      auto map_it = entity_map.find(v);

      if (map_it == entity_map.end())
      {
        throw std::runtime_error("Cannot find entity in map when reading "
                                 "mesh::MeshValueCollection");
      }
      for (auto p = map_it->second.begin(); p != map_it->second.end(); p += 2)
      {
        const std::size_t dest = *p;
        assert(dest < num_processes);
        send_entities[dest].push_back(*(p + 1));
        send_data[dest].push_back(recv_data[i][j]);
      }
    }
  }

  // Send to owning processes and set in mesh::MeshValueCollection
  MPI::all_to_all(_mpi_comm.comm(), send_entities, recv_entities);
  MPI::all_to_all(_mpi_comm.comm(), send_data, recv_data);

  mesh::MeshValueCollection<T> mvc(mesh, dim);
  for (std::size_t i = 0; i != num_processes; ++i)
  {
    assert(recv_entities[i].size() == recv_data[i].size());
    for (std::size_t j = 0; j != recv_data[i].size(); ++j)
      mvc.set_value(recv_entities[i][j], recv_data[i][j]);
  }

  return mvc;
}
//-----------------------------------------------------------------------------
mesh::Mesh HDF5File::read_mesh(const std::string data_path,
                               bool use_partition_from_file,
                               const mesh::GhostMode ghost_mode) const
{
  // Read local mesh data
  const auto [cell_type, points, cells, global_cell_indices, cell_distribution]
      = read_mesh_data(data_path);

  if (use_partition_from_file)
  {
    // Check that number of MPI processes matches partitioning
    if (MPI::size(_mpi_comm.comm()) != (cell_distribution.size() - 1))
    {
      throw std::runtime_error("Different number of processes used when "
                               "writing. Cannot restore partitioning");
    }

    const std::int32_t num_local_cells = global_cell_indices.size();
    std::vector<int> part(num_local_cells);

    // Get offset for this process
    const std::int64_t offset
        = dolfinx::MPI::global_offset(_mpi_comm.comm(), num_local_cells, true);

    // Convert cell distribution to an array of proces destination for each
    // local cell
    for (std::int32_t i = 0; i < num_local_cells; ++i)
    {
      auto it = std::upper_bound(cell_distribution.begin(),
                                 cell_distribution.end(), i + offset);

      part[i] = static_cast<int>(it - cell_distribution.begin() - 1);
    }

    std::map<std::int64_t, std::vector<int>> ghost_procs;
    ghost_procs = mesh::Partitioning::compute_halo_cells(_mpi_comm.comm(), part,
                                                         cell_type, cells);

    mesh::PartitionData cell_partition(part, ghost_procs);

    return mesh::Partitioning::build_from_partition(
        _mpi_comm.comm(), cell_type, points, cells, global_cell_indices,
        ghost_mode, cell_partition);
  }
  else
  {
    return mesh::Partitioning::build_distributed_mesh(
        _mpi_comm.comm(), cell_type, points, cells, global_cell_indices,
        ghost_mode);
  }
}
//-----------------------------------------------------------------------------
std::tuple<
    mesh::CellType,
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
    Eigen::Array<std::int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
    std::vector<std::int64_t>, std::vector<std::int64_t>>
HDF5File::read_mesh_data(const std::string data_path) const
{

  assert(_hdf5_file_id > 0);

  // Check that topology data set is found in HDF5 file
  const std::string topology_path = data_path + "/topology";
  if (!HDF5Interface::has_dataset(_hdf5_file_id, topology_path))
  {
    throw std::runtime_error("Cannot read topology dataset. "
                             "Dataset \""
                             + topology_path + "\" not found");
  }

  // --- Topology ---

  // Get topology data
  std::string cell_type_str;
  if (HDF5Interface::has_attribute(_hdf5_file_id, topology_path, "celltype"))
  {
    cell_type_str = HDF5Interface::get_attribute<std::string>(
        _hdf5_file_id, topology_path, "celltype");
  }

  // Create CellType from string
  mesh::CellType cell_type = mesh::to_type(cell_type_str);

  // Check that coordinate data set is found in HDF5 file
  const std::string geometry_path = data_path + "/coordinates";
  if (!HDF5Interface::has_dataset(_hdf5_file_id, geometry_path))
  {
    throw std::runtime_error("Cannot read geometry dataset. "
                             "Dataset \""
                             + geometry_path + "\" not found");
  }

  // Get dimensions of coordinate dataset
  std::vector<std::int64_t> coords_shape
      = HDF5Interface::get_dataset_shape(_hdf5_file_id, geometry_path);
  assert(coords_shape.size() < 3);
  if (coords_shape.size() == 1)
  {
    throw std::runtime_error("Cannot determine geometric dimension from "
                             "one-dimensional array storage in HDF5 file");
  }
  else if (coords_shape.size() > 2)
  {
    throw std::runtime_error("Cannot determine geometric dimension from "
                             "high-rank array storage in HDF5 file");
  }

  // Extract geometric dimension
  int gdim = coords_shape[1];

  // Discover shape of the topology data set in HDF5 file
  std::vector<std::int64_t> topology_shape
      = HDF5Interface::get_dataset_shape(_hdf5_file_id, topology_path);

  // Number of nodes per element != vertices per element for higher order
  // meshes.
  const int num_nodes_per_cell = topology_shape[1];

  // Compute number of global cells (handle case that topology may be
  // arranged a 1D or 2D array)
  std::int64_t num_global_cells = 0;
  if (topology_shape.size() == 1)
  {
    assert(topology_shape[0] % num_nodes_per_cell == 0);
    num_global_cells = topology_shape[0] / num_nodes_per_cell;
  }
  else if (topology_shape.size() == 2)
  {
    num_global_cells = topology_shape[0];
  }
  else
  {
    throw std::runtime_error("Topology in HDF5 file has wrong shape");
  }

  // Get partition from file, if available
  std::vector<std::int64_t> cell_distribution;
  if (HDF5Interface::has_attribute(_hdf5_file_id, topology_path, "partition"))
  {
    cell_distribution = HDF5Interface::get_attribute<std::vector<std::int64_t>>(
        _hdf5_file_id, topology_path, "partition");
  }

  // Prepare range of cells to read on this process
  std::array<std::int64_t, 2> cell_range;

  // Check whether number of MPI processes matches partitioning, and
  // restore if possible
  if (MPI::size(_mpi_comm.comm()) == cell_distribution.size())
  {
    cell_distribution.push_back(num_global_cells);
    const std::size_t proc = MPI::rank(_mpi_comm.comm());
    cell_range = {{cell_distribution[proc], cell_distribution[proc + 1]}};
  }
  else
  {
    // Divide up cells approximately equally between processes
    cell_range = MPI::local_range(_mpi_comm.comm(), num_global_cells);
  }

  // Get number of cells to read on this process
  const int num_local_cells = cell_range[1] - cell_range[0];

  // Modify range of array to read for flat HDF5 storage
  std::array<std::int64_t, 2> cell_data_range = cell_range;
  if (topology_shape.size() == 1)
  {
    cell_data_range[0] *= num_nodes_per_cell;
    cell_data_range[1] *= num_nodes_per_cell;
  }

  // Read a block of cells
  std::vector<std::int64_t> topology_data
      = HDF5Interface::read_dataset<std::int64_t>(_hdf5_file_id, topology_path,
                                                  cell_data_range);

  // FIXME: explain this more clearly.
  // Reconstruct mesh_name from topology_name - needed for
  // cell_indices
  std::string mesh_name = topology_path.substr(0, topology_path.rfind("/"));

  // Look for cell indices in dataset, and use if available.
  // Otherwise renumber from zero across processes
  std::vector<std::int64_t> global_cell_indices;
  global_cell_indices.clear();
  const std::string cell_indices_name = mesh_name + "/cell_indices";
  if (HDF5Interface::has_dataset(_hdf5_file_id, cell_indices_name))
  {
    global_cell_indices = HDF5Interface::read_dataset<std::int64_t>(
        _hdf5_file_id, cell_indices_name, cell_range);
  }
  else
  {
    global_cell_indices.resize(num_local_cells);
    std::iota(global_cell_indices.begin(), global_cell_indices.end(),
              cell_range[0]);
  }

  Eigen::Map<Eigen::Array<std::int64_t, Eigen::Dynamic, Eigen::Dynamic,
                          Eigen::RowMajor>>
      cells(topology_data.data(), num_local_cells, num_nodes_per_cell);

  cells = io::cells::permute_ordering(
      cells, io::cells::vtk_to_dolfin(cell_type, num_nodes_per_cell));

  // --- Coordinates ---

  // Get dimensions of coordinate dataset
  std::int64_t num_global_points = 0;
  if (coords_shape.size() == 1)
  {
    assert(coords_shape[0] % gdim == 0);
    num_global_points = coords_shape[0] / gdim;
  }
  else if (coords_shape.size() == 2)
  {
    assert((int)coords_shape[1] == gdim);
    num_global_points = coords_shape[0];
  }
  else
  {
    throw std::runtime_error("Topology in HDF5 file has wrong shape");
  }

  // Divide point range into equal blocks for each process
  std::array<std::int64_t, 2> vertex_range
      = MPI::local_range(_mpi_comm.comm(), num_global_points);
  const std::size_t num_local_points = vertex_range[1] - vertex_range[0];

  // Modify vertex data range for flat storage
  std::array<std::int64_t, 2> vertex_data_range = vertex_range;
  if (coords_shape.size() == 1)
  {
    vertex_data_range[0] *= gdim;
    vertex_data_range[1] *= gdim;
  }

  // Read vertex data to temporary vector
  std::vector<double> coordinates_data = HDF5Interface::read_dataset<double>(
      _hdf5_file_id, geometry_path, vertex_data_range);
  assert(coordinates_data.size() == num_local_points * gdim);
  Eigen::Map<
      Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      points(coordinates_data.data(), num_local_points, gdim);

  return std::tuple(cell_type, std::move(points), std::move(cells),
                    std::move(global_cell_indices),
                    std::move(cell_distribution));
}
//-----------------------------------------------------------------------------
bool HDF5File::has_dataset(const std::string dataset_name) const
{
  assert(_hdf5_file_id > 0);
  return HDF5Interface::has_dataset(_hdf5_file_id, dataset_name);
}
//-----------------------------------------------------------------------------
void HDF5File::set_mpi_atomicity(bool atomic)
{
  assert(_hdf5_file_id > 0);
  HDF5Interface::set_mpi_atomicity(_hdf5_file_id, atomic);
}
//-----------------------------------------------------------------------------
bool HDF5File::get_mpi_atomicity() const
{
  assert(_hdf5_file_id > 0);
  return HDF5Interface::get_mpi_atomicity(_hdf5_file_id);
}
//-----------------------------------------------------------------------------
