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
// Last changed: 2013-11-05

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
HDF5File::HDF5File(const std::string filename, const std::string file_mode,
                   bool use_mpiio)
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
  close();
}
//-----------------------------------------------------------------------------
void HDF5File::close()
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
void HDF5File::read(GenericVector& x, const std::string dataset_name,
                    const bool use_partition_from_file) const
{
  dolfin_assert(hdf5_file_open);

  // Check for data set exists
  if (!HDF5Interface::has_dataset(hdf5_file_id, dataset_name))
    error("Data set with name \"%s\" does not exist", dataset_name.c_str());

  // Get dataset rank
  const std::size_t rank = HDF5Interface::dataset_rank(hdf5_file_id,
                                                       dataset_name);
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
      HDF5Interface::get_attribute(hdf5_file_id, dataset_name, "partition",
                                   partitions);

      // Check that number of MPI processes matches partitioning
      if (MPI::num_processes() != partitions.size())
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
      = HDF5Utility::reorder_vertices_by_global_indices(mesh);

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
      // Drop duplicate topology for shared entities of less than mesh
      // dimension

      // If not already numbered, number entities of order cell_dim so
      // we can get shared_entities
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
          std::set<unsigned int>::const_iterator lowest_proc
            = sh->second.begin();
          if (*lowest_proc > my_rank)
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

    // For cells, write the global cell index
    if (cell_dim == mesh.topology().dim())
    {
      const std::string cell_index_dataset = name + "/cell_indices";
      global_size.pop_back();
      const std::vector<std::size_t>& cells =
        mesh.topology().global_indices(mesh.topology().dim());
      write_data(cell_index_dataset, cells, global_size);
    }

    // Add cell type attribute
    HDF5Interface::add_attribute(hdf5_file_id, topology_dataset, "celltype",
                    CellType::type2string((CellType::Type)cell_dim));

    // Add partitioning attribute to dataset
    std::vector<std::size_t> partitions;
    const std::size_t topology_offset
      = MPI::global_offset(topological_data.size()/(cell_dim + 1), true);
    MPI::gather(topology_offset, partitions);
    MPI::broadcast(partitions);
    HDF5Interface::add_attribute(hdf5_file_id, topology_dataset,
                                 "partition", partitions);
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
  const Mesh& mesh = *meshfunction.mesh();
  const std::size_t cell_dim = meshfunction.dim();

  // HDF5 does not support a boolean type,
  // so copy to int with values 1 and 0
  MeshFunction<int> mf(mesh, cell_dim);
  for (MeshEntityIterator cell(mesh, cell_dim); !cell.end(); ++cell)
    mf[cell->index()] = (meshfunction[cell->index()] ? 1 : 0);

  write_mesh_function(mf, name);
}
//-----------------------------------------------------------------------------
void HDF5File::read(MeshFunction<bool>& meshfunction,
                    const std::string name) const
{
  const Mesh& mesh = *meshfunction.mesh();
  const std::size_t cell_dim = meshfunction.dim();

  // HDF5 does not support bool, so use int instead
  MeshFunction<int> mf(mesh, cell_dim);
  read_mesh_function(mf, name);

  for (MeshEntityIterator cell(mesh, cell_dim); !cell.end(); ++cell)
    meshfunction[cell->index()] = (mf[cell->index()] != 0);
}
//-----------------------------------------------------------------------------
template <typename T>
void HDF5File::read_mesh_function(MeshFunction<T>& meshfunction,
                                  const std::string mesh_name) const
{
  const Mesh& mesh = *meshfunction.mesh();

  dolfin_assert(hdf5_file_open);

  const std::string topology_name = mesh_name + "/topology";

  if (!HDF5Interface::has_dataset(hdf5_file_id, topology_name))
  {
    dolfin_error("HDF5File.cpp",
                 "read topology dataset",
                 "Dataset \"%s\" not found", topology_name.c_str());
  }

  // Look for Coordinates dataset - but not used
  const std::string coordinates_name = mesh_name + "/coordinates";
  if (!HDF5Interface::has_dataset(hdf5_file_id, coordinates_name))
  {
    dolfin_error("HDF5File.cpp",
                 "read coordinates dataset",
                 "Dataset \"%s\" not found", coordinates_name.c_str());
  }

  // Look for Values dataset
  const std::string values_name = mesh_name + "/values";
  if (!HDF5Interface::has_dataset(hdf5_file_id, values_name))
  {
    dolfin_error("HDF5File.cpp",
                 "read values dataset",
                 "Dataset \"%s\" not found", values_name.c_str());
  }

  // --- Topology ---
  // Discover size of topology dataset
  const std::vector<std::size_t> topology_dim
      = HDF5Interface::get_dataset_size(hdf5_file_id, topology_name);

  // Some consistency checks

  const std::size_t num_global_cells = topology_dim[0];
  const std::size_t vert_per_cell = topology_dim[1];
  const std::size_t cell_dim = vert_per_cell - 1;

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
  DistributedMeshTools::number_entities(mesh, cell_dim);

  if (num_global_cells != mesh.size_global(cell_dim))
  {
    dolfin_error("HDF5File.cpp",
                 "read meshfunction topology",
                 "Mesh dimension mismatch");
  }

  // Divide up cells ~equally between processes
  const std::pair<std::size_t, std::size_t> cell_range
    = MPI::local_range(num_global_cells);
  const std::size_t num_read_cells = cell_range.second - cell_range.first;

  // Read a block of cells
  std::vector<std::size_t> topology_data;
  topology_data.reserve(num_read_cells*vert_per_cell);
  HDF5Interface::read_dataset(hdf5_file_id, topology_name, cell_range,
                              topology_data);

  boost::multi_array_ref<std::size_t, 2>
    topology_array(topology_data.data(),
                   boost::extents[num_read_cells][vert_per_cell]);

  std::vector<T> value_data;
  value_data.reserve(num_read_cells);
  HDF5Interface::read_dataset(hdf5_file_id, values_name, cell_range,
                              value_data);

  // Now send the read data to each process on the basis of the first
  // vertex of the entity, since we do not know the global_index
  const std::size_t num_processes = MPI::num_processes();
  const std::size_t max_vertex = mesh.size_global(0);

  std::vector<std::vector<std::size_t> > send_topology(num_processes);
  std::vector<std::vector<std::size_t> > receive_topology(num_processes);
  std::vector<std::vector<T> > send_values(num_processes);
  std::vector<std::vector<T> > receive_values(num_processes);

  for (std::size_t i = 0; i < num_read_cells ; ++i)
  {
    std::vector<std::size_t> cell_topology(topology_array[i].begin(),
                                           topology_array[i].end());
    std::sort(cell_topology.begin(), cell_topology.end());

    // Use first vertex to decide where to send this data
    const std::size_t send_to_process
      = MPI::index_owner(cell_topology.front(), max_vertex);

    send_topology[send_to_process].insert(send_topology[send_to_process].end(),
                              cell_topology.begin(), cell_topology.end());
    send_values[send_to_process].push_back(value_data[i]);
  }

  MPI::all_to_all(send_topology, receive_topology);
  MPI::all_to_all(send_values, receive_values);

  // Generate requests for data from remote processes, based on the
  // first vertex of the MeshEntities which belong on this process
  // Send our process number, and our local index, so it can come back
  // directly to the right place
  std::vector<std::vector<std::size_t> > send_requests(num_processes);
  std::vector<std::vector<std::size_t> > receive_requests(num_processes);

  const std::size_t process_number = MPI::process_number();

  for (MeshEntityIterator cell(mesh, cell_dim); !cell.end(); ++cell)
  {
    std::vector<std::size_t> cell_topology;
    for (VertexIterator v(*cell); !v.end(); ++v)
    {
      cell_topology.push_back(v->global_index());
    }
    std::sort(cell_topology.begin(), cell_topology.end());

    // Use first vertex to decide where to send this request
    std::size_t send_to_process = MPI::index_owner(cell_topology.front(),
                                                   max_vertex);
    // Map to this process and local index by appending to send data
    cell_topology.push_back(cell->index());
    cell_topology.push_back(process_number);
    send_requests[send_to_process].insert(send_requests[send_to_process].end(),
                                          cell_topology.begin(),
                                          cell_topology.end());
  }

  MPI::all_to_all(send_requests, receive_requests);

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
    dolfin_assert(receive_values[i].size()*vert_per_cell
                  == receive_topology[i].size());
    std::vector<std::size_t>::iterator p = receive_topology[i].begin();
    for (std::size_t j = 0; j < receive_values[i].size(); ++j)
    {
      const std::vector<std::size_t> cell(p, p + vert_per_cell);
      cell_to_data[cell] = receive_values[i][j];
      p += vert_per_cell;
    }
  }

  // Clear vectors for reuse - now to send values and indices to final
  // destination
  send_topology = std::vector<std::vector<std::size_t> >(num_processes);
  send_values = std::vector<std::vector<T> >(num_processes);

  // Go through requests, which are stacked as [vertex, vertex, ...]
  // [index] [proc] etc.  Use the vertices as the key for the map
  // (above) to retrieve the data to send to proc
  for (std::size_t i = 0; i < receive_requests.size(); ++i)
  {
    for (std::vector<std::size_t>::iterator p = receive_requests[i].begin();
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

  for (std::size_t i = 0; i < receive_values.size(); ++i)
  {
    dolfin_assert(receive_values[i].size() == receive_topology[i].size());
    for (std::size_t j = 0; j < receive_values[i].size(); ++j)
    {
      meshfunction[receive_topology[i][j]] = receive_values[i][j];
    }
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

  if (cell_dim == mesh.topology().dim() || MPI::num_processes() == 1)
  {
    // No duplicates
    data_values.assign(meshfunction.values(),
                       meshfunction.values() + meshfunction.size());
  }
  else
  {
    data_values.reserve(mesh.size(cell_dim));

    // Drop duplicate data
    const std::size_t my_rank = MPI::process_number();
    const std::map<unsigned int, std::set<unsigned int> >& shared_entities
      = mesh.topology().shared_entities(cell_dim);

    for (std::size_t i = 0; i < meshfunction.size(); ++i)
    {
      std::map<unsigned int, std::set<unsigned int> >::const_iterator sh
        = shared_entities.find(i);

      // If unshared, or shared and locally owned, append to vector
      if (sh == shared_entities.end())
        data_values.push_back(meshfunction[i]);
      else
      {
        std::set<unsigned int>::iterator lowest_proc = sh->second.begin();
        if (*lowest_proc > my_rank)
          data_values.push_back(meshfunction[i]);
      }
    }
  }

  // Write values to HDF5
  std::vector<std::size_t> global_size(1, MPI::sum(data_values.size()));
  write_data(name + "/values", data_values, global_size);
}
//-----------------------------------------------------------------------------
void HDF5File::write(const Function& u,  const std::string name,
                     double timestamp)
{
  if (!HDF5Interface::has_dataset(hdf5_file_id, name))
  {
    write(u, name);
    std::vector<double> vectime(1, timestamp);
    attributes(name).set("series", vectime);
  }
  else
  {
    HDF5Attribute attr = attributes(name);
    if (!attr.exists("series"))
    {
      dolfin_error("HDF5File.cpp",
                   "append to series",
                   "Function dataset does not contain a 'series' attribute");
    }

    std::vector<double> vectime;
    attr.get("series", vectime);

    std::size_t nvec = vectime.size();
    std::string vecname = name
      + "/vector_" + boost::lexical_cast<std::string>(nvec);

    vectime.push_back(timestamp);
    attr.set("series", vectime);

    write(*u.vector(), vecname);
  }
}
//-----------------------------------------------------------------------------
void HDF5File::write(const Function& u, const std::string name)
{
  Timer t0("HDF5: write Function");

  // Get mesh and dofmap
  dolfin_assert(u.function_space()->mesh());
  const Mesh& mesh = *u.function_space()->mesh();

  dolfin_assert(u.function_space()->dofmap());
  const GenericDofMap& dofmap = *u.function_space()->dofmap();

  // FIXME:
  // Possibly sort cell_dofs into global cell order before writing?

  // Save data in compressed format with an index to mark out
  // the start of each row

  std::vector<dolfin::la_index> cell_dofs;
  std::vector<std::size_t> x_cell_dofs;
  x_cell_dofs.reserve(mesh.num_cells());

  for (std::size_t i = 0; i != mesh.num_cells(); ++i)
  {
    x_cell_dofs.push_back(cell_dofs.size());
    const std::vector<dolfin::la_index>& cell_dofs_i = dofmap.cell_dofs(i);
    cell_dofs.insert(cell_dofs.end(), cell_dofs_i.begin(), cell_dofs_i.end());
  }

  // Add offset to CSR index to be seamless in parallel
  std::size_t offset = MPI::global_offset(cell_dofs.size(), true);
  std::transform(x_cell_dofs.begin(),
                 x_cell_dofs.end(),
                 x_cell_dofs.begin(),
                 std::bind2nd(std::plus<std::size_t>(), offset));

  // Save DOFs on each cell
  std::vector<std::size_t> global_size(1, MPI::sum(cell_dofs.size()));
  write_data(name + "/cell_dofs", cell_dofs, global_size);
  if (MPI::process_number() == MPI::num_processes() - 1)
    x_cell_dofs.push_back(global_size[0]);
  global_size[0] = mesh.size_global(mesh.topology().dim()) + 1;
  write_data(name + "/x_cell_dofs", x_cell_dofs, global_size);

  // Save cell ordering
  const std::vector<std::size_t>& cells =
    mesh.topology().global_indices(mesh.topology().dim());
  global_size[0] = mesh.size_global(mesh.topology().dim());
  write_data(name + "/cells", cells, global_size);

  // Save vector
  write(*u.vector(), name + "/vector");
}
//-----------------------------------------------------------------------------
void HDF5File::read(Function& u, const std::string name)
{
  Timer t0("HDF5: read Function");

  dolfin_assert(hdf5_file_open);

  // FIXME: This routine is long and involves a lot of MPI, but it
  // should work for the general case of reading a function that was
  // written from a different number of processes.  Memory efficiency
  // could be improved by limiting the scope of some of the temporary
  // variables

  std::string basename = name;
  std::string vector_dataset_name = name + "/vector";

  // Check that the name we have been given corresponds to a "group"
  // If not, then maybe we have been given the vector dataset name
  // directly, so the group name should be one level up.
  if (!HDF5Interface::has_group(hdf5_file_id, basename))
  {
    basename = name.substr(0, name.rfind("/"));
    vector_dataset_name = name;
  }

  const std::string cells_dataset_name = basename + "/cells";
  const std::string cell_dofs_dataset_name = basename + "/cell_dofs";
  const std::string x_cell_dofs_dataset_name = basename + "/x_cell_dofs";

  // Check datasets exist
  if (!HDF5Interface::has_group(hdf5_file_id, name))
    error("Group with name \"%s\" does not exist", name.c_str());
  if (!HDF5Interface::has_dataset(hdf5_file_id, cells_dataset_name))
    error("Dataset with name \"%s\" does not exist",
          cells_dataset_name.c_str());
  if (!HDF5Interface::has_dataset(hdf5_file_id, cell_dofs_dataset_name))
    error("Dataset with name \"%s\" does not exist",
          cell_dofs_dataset_name.c_str());
  if (!HDF5Interface::has_dataset(hdf5_file_id, x_cell_dofs_dataset_name))
    error("Dataset with name \"%s\" does not exist",
          x_cell_dofs_dataset_name.c_str());
  if (!HDF5Interface::has_dataset(hdf5_file_id, vector_dataset_name))
    error("Dataset with name \"%s\" does not exist",
          vector_dataset_name.c_str());

  // Get existing mesh and dofmap - these should be pre-existing
  // and set up by user when defining the Function
  dolfin_assert(u.function_space()->mesh());
  const Mesh& mesh = *u.function_space()->mesh();
  dolfin_assert(u.function_space()->dofmap());
  const GenericDofMap& dofmap = *u.function_space()->dofmap();

  // Get dimension of dataset
  const std::vector<std::size_t> dataset_size =
    HDF5Interface::get_dataset_size(hdf5_file_id, cells_dataset_name);
  const std::size_t num_global_cells = dataset_size[0];
  if (mesh.size_global(mesh.topology().dim())
     != num_global_cells)
  {
    dolfin_error("HDF5File.cpp",
                 "read Function from file",
                 "Number of global cells does not match");
  }

  // Divide cells equally between processes
  const std::pair<std::size_t, std::size_t> cell_range
    = MPI::local_range(num_global_cells);

  // Read cells
  std::vector<std::size_t> input_cells;
  HDF5Interface::read_dataset(hdf5_file_id, cells_dataset_name,
                              cell_range, input_cells);

  // Overlap reads of DOF indices, to get full range on each process
  std::vector<std::size_t> x_cell_dofs;
  HDF5Interface::read_dataset(hdf5_file_id, x_cell_dofs_dataset_name,
                              std::make_pair(cell_range.first,
                                             cell_range.second + 1),
                              x_cell_dofs);

  // Read cell-DOF maps
  std::vector<dolfin::la_index> input_cell_dofs;
  HDF5Interface::read_dataset(hdf5_file_id, cell_dofs_dataset_name,
                              std::make_pair(x_cell_dofs.front(),
                                             x_cell_dofs.back()),
                              input_cell_dofs);

  GenericVector& x = *u.vector();

  const std::vector<std::size_t> vector_size =
    HDF5Interface::get_dataset_size(hdf5_file_id, vector_dataset_name);
  const std::size_t num_global_dofs = vector_size[0];
  dolfin_assert(num_global_dofs == x.size(0));
  const std::pair<dolfin::la_index, dolfin::la_index>
    input_vector_range = MPI::local_range(vector_size[0]);

  std::vector<double> input_values;
  HDF5Interface::read_dataset(hdf5_file_id, vector_dataset_name,
                              input_vector_range,
                              input_values);

  // Calculate one (global cell, local_dof_index) to associate
  // with each item in the vector on this process

  std::vector<std::size_t> global_cells;
  std::vector<std::size_t> remote_local_dofi;

  HDF5Utility::map_gdof_to_cell(input_cells, input_cell_dofs,
                                x_cell_dofs, input_vector_range,
                                global_cells, remote_local_dofi);

  // At this point, each process has a set of data, and for
  // each value, a global_cell and local_dof to send it to.
  // However, it is not known which processes the cells
  // are actually on.

  // Find where the needed cells are held
  std::vector<std::pair<std::size_t, std::size_t > >
    cell_ownership = HDF5Utility::cell_owners(mesh, global_cells);

  // Having found the cell location, the actual global_dof index
  // held by that (cell, local_dof) is needed on the process
  // which holds the data values

  std::vector<dolfin::la_index> global_dof;
  HDF5Utility::get_global_dof(cell_ownership, remote_local_dofi,
                              input_vector_range, dofmap, global_dof);


  const std::size_t num_processes = MPI::num_processes();

  // Shift to dividing things into the vector range of Function Vector
  const std::pair<dolfin::la_index, dolfin::la_index>
    vector_range = x.local_range();

  std::vector<std::vector<double> > receive_values(num_processes);
  std::vector<std::vector<dolfin::la_index> > receive_indices(num_processes);
  {
    std::vector<std::vector<double> > send_values(num_processes);
    std::vector<std::vector<dolfin::la_index> > send_indices(num_processes);
    const std::size_t
      n_vector_vals = input_vector_range.second - input_vector_range.first;
    std::vector<dolfin::la_index> all_vec_range;
    MPI::gather(vector_range.second, all_vec_range);
    MPI::broadcast(all_vec_range);

    for (std::size_t i = 0; i != n_vector_vals; ++i)
    {
      const std::size_t dest = std::upper_bound(all_vec_range.begin(),
                                                all_vec_range.end(),
                                                global_dof[i])
                                              - all_vec_range.begin();
      dolfin_assert(dest < num_processes);
      dolfin_assert(i < input_values.size());
      send_indices[dest].push_back(global_dof[i]);
      send_values[dest].push_back(input_values[i]);
    }

    MPI::all_to_all(send_values, receive_values);
    MPI::all_to_all(send_indices, receive_indices);
  }

  std::vector<double>
    vector_values(vector_range.second - vector_range.first);

  for (std::size_t i = 0; i != num_processes; ++i)
  {
    const std::vector<double>& rval = receive_values[i];
    const std::vector<dolfin::la_index>& rindex = receive_indices[i];
    dolfin_assert(rval.size() == rindex.size());
    for (std::size_t j = 0; j != rindex.size(); ++j)
    {
      dolfin_assert(rindex[j] >= vector_range.first);
      dolfin_assert(rindex[j] < vector_range.second);
      vector_values[rindex[j] - vector_range.first]
        = rval[j];
    }
  }

  x.set_local(vector_values);
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
  for (std::map<std::pair<std::size_t, std::size_t>,
                bool>::const_iterator mesh_value_it = values.begin();
       mesh_value_it != values.end(); ++mesh_value_it)
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
  for (std::map<std::pair<std::size_t, std::size_t>,
                int>::const_iterator mesh_value_it = values.begin();
      mesh_value_it != values.end(); ++mesh_value_it)
  {
    mesh_values.set_value(mesh_value_it->first.first,
                          mesh_value_it->first.second,
                          (mesh_value_it->second != 0));
  }

}
//-----------------------------------------------------------------------------
template <typename T>
void HDF5File::write_mesh_value_collection(const MeshValueCollection<T>& mesh_values, const std::string name)
{
  const std::map<std::pair<std::size_t, std::size_t>, T>& values
    = mesh_values.values();

  const Mesh& mesh = *mesh_values.mesh();
  const std::vector<std::size_t>& global_cell_index
    = mesh.topology().global_indices(mesh.topology().dim());

  std::vector<T> data_values;
  std::vector<std::size_t> entities;
  std::vector<std::size_t> cells;

  for (typename std::map<std::pair<std::size_t, std::size_t>,
         T>::const_iterator
         p = values.begin(); p != values.end(); ++p)
  {
    cells.push_back(global_cell_index[p->first.first]);
    entities.push_back(p->first.second);
    data_values.push_back(p->second);
  }

  std::vector<std::size_t> global_size(1, MPI::sum(data_values.size()));
  write_data(name + "/values", data_values, global_size);
  write_data(name + "/entities", entities, global_size);
  write_data(name + "/cells", cells, global_size);

  HDF5Interface::add_attribute(hdf5_file_id, name, "dimension",
                               mesh_values.dim());
}
//-----------------------------------------------------------------------------
template <typename T>
void HDF5File::read_mesh_value_collection(MeshValueCollection<T>& mesh_vc,
                                          const std::string name) const
{
  dolfin_assert(hdf5_file_open);
  mesh_vc.clear();
  if (!HDF5Interface::has_group(hdf5_file_id, name))
  {
    dolfin_error("HDF5File.cpp",
                 "open MeshValueCollection dataset",
                 "Group \"%s\" not found in file", name.c_str());
  }

  std::size_t dim = 0;
  HDF5Interface::get_attribute(hdf5_file_id, name, "dimension", dim);

  const std::string values_name = name + "/values";
  const std::string entities_name = name + "/entities";
  const std::string cells_name = name + "/cells";

  if (!HDF5Interface::has_dataset(hdf5_file_id, values_name))
  {
    dolfin_error("HDF5File.cpp",
                 "open MeshValueCollection dataset",
                 "Dataset \"%s\" not found in file", values_name.c_str());
  }

  if (!HDF5Interface::has_dataset(hdf5_file_id, entities_name))
  {
    dolfin_error("HDF5File.cpp",
                 "open MeshValueCollection dataset",
                 "Dataset \"%s\" not found in file", entities_name.c_str());
  }

  if (!HDF5Interface::has_dataset(hdf5_file_id, cells_name))
  {
    dolfin_error("HDF5File.cpp",
                 "open MeshValueCollection dataset",
                 "Dataset \"%s\" not found in file", cells_name.c_str());
  }

  // Check all datasets have the same size
  const std::vector<std::size_t> values_dim
      = HDF5Interface::get_dataset_size(hdf5_file_id, values_name);
  const std::vector<std::size_t> entities_dim
      = HDF5Interface::get_dataset_size(hdf5_file_id, entities_name);
  const std::vector<std::size_t> cells_dim
      = HDF5Interface::get_dataset_size(hdf5_file_id, cells_name);
  dolfin_assert(values_dim[0] == entities_dim[0]);
  dolfin_assert(values_dim[0] == cells_dim[0]);

  // Check size of dataset. If small enough, just read on all processes...

  // FIXME: optimise value
  const std::size_t max_data_one = 1048576; // arbtirary 1M

  if (values_dim[0] < max_data_one)
  {
    // read on all processes
    const std::pair<std::size_t, std::size_t> range(0, values_dim[0]);
    const std::size_t local_size = range.second - range.first;

    std::vector<T> values_data;
    values_data.reserve(local_size);
    HDF5Interface::read_dataset(hdf5_file_id, values_name, range, values_data);
    std::vector<std::size_t> entities_data;
    entities_data.reserve(local_size);
    HDF5Interface::read_dataset(hdf5_file_id, entities_name, range,
                                entities_data);
    std::vector<std::size_t> cells_data;
    cells_data.reserve(local_size);
    HDF5Interface::read_dataset(hdf5_file_id, cells_name, range, cells_data);

    // Get global mapping to restore values
    const Mesh& mesh = *mesh_vc.mesh();
    const std::vector<std::size_t>& global_cell_index
      = mesh.topology().global_indices(mesh.topology().dim());

    // Reference to actual map of MeshValueCollection
    std::map<std::pair<std::size_t, std::size_t>, T>& mvc_map
      = mesh_vc.values();

    // Find cells which are on this process
    for (std::size_t i = 0; i < cells_data.size(); ++i)
    {
      const std::vector<std::size_t>::const_iterator lidx
        = std::find(global_cell_index.begin(), global_cell_index.end(),
                    cells_data[i]);
      if (lidx != global_cell_index.end())
      {
        const std::size_t local_index = lidx - global_cell_index.begin();
        mvc_map[std::make_pair(local_index, entities_data[i])]
          = values_data[i];
      }
    }

  }
  else
  {
    const Mesh& mesh = *mesh_vc.mesh();

    // Divide range between processes
    const std::pair<std::size_t, std::size_t> data_range
      = MPI::local_range(values_dim[0]);
    const std::size_t local_size = data_range.second - data_range.first;

    // Read local range of values, entities and cells
    std::vector<T> values_data;
    values_data.reserve(local_size);
    HDF5Interface::read_dataset(hdf5_file_id, values_name, data_range,
                                values_data);
    std::vector<std::size_t> entities_data;
    entities_data.reserve(local_size);
    HDF5Interface::read_dataset(hdf5_file_id, entities_name, data_range,
                                entities_data);
    std::vector<std::size_t> cells_data;
    cells_data.reserve(local_size);
    HDF5Interface::read_dataset(hdf5_file_id, cells_name, data_range,
                                cells_data);

    std::vector<std::pair<std::size_t, std::size_t> > cell_ownership;
    cell_ownership = HDF5Utility::cell_owners(mesh, cells_data);

    const std::size_t num_processes = MPI::num_processes();
    std::vector<std::vector<std::size_t> > send_entities(num_processes);
    std::vector<std::vector<std::size_t> > send_local(num_processes);
    std::vector<std::vector<T> > send_values(num_processes);
    std::vector<std::vector<std::size_t> > recv_entities(num_processes);
    std::vector<std::vector<std::size_t> > recv_local(num_processes);
    std::vector<std::vector<T> > recv_values(num_processes);

    for (std::size_t i = 0; i != cells_data.size(); ++i)
    {
      const std::size_t dest = cell_ownership[i].first;
      send_local[dest].push_back(cell_ownership[i].second);
      send_entities[dest].push_back(entities_data[i]);
      send_values[dest].push_back(values_data[i]);
    }

    MPI::all_to_all(send_entities, recv_entities);
    MPI::all_to_all(send_local, recv_local);
    MPI::all_to_all(send_values, recv_values);

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
void HDF5File::read(Mesh& input_mesh, const std::string mesh_name) const
{
  Timer t("HDF5: read mesh");

  dolfin_assert(hdf5_file_open);

  const std::string topology_name = mesh_name + "/topology";
  if (!HDF5Interface::has_dataset(hdf5_file_id, topology_name))
  {
    dolfin_error("HDF5File.cpp",
                 "read topology dataset",
                 "Dataset \"%s\" not found", topology_name.c_str());
  }

  const std::string coordinates_name = mesh_name + "/coordinates";
  if (!HDF5Interface::has_dataset(hdf5_file_id, coordinates_name))
  {
    dolfin_error("HDF5File.cpp",
                 "read coordinates dataset",
                 "Dataset \"%s\" not found", coordinates_name.c_str());
  }

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

  // Get partition from file
  std::vector<std::size_t> partitions;
  HDF5Interface::get_attribute(hdf5_file_id, topology_name, "partition",
                               partitions);

  std::pair<std::size_t, std::size_t> cell_range;

  // Check whether number of MPI processes matches partitioning, and
  // restore if possible
  if (MPI::num_processes() == partitions.size())
  {
    partitions.push_back(num_global_cells);
    const std::size_t proc = MPI::process_number();
    cell_range = std::make_pair(partitions[proc], partitions[proc + 1]);
  }
  else
  {
    // Divide up cells ~equally between processes
    cell_range = MPI::local_range(num_global_cells);
  }

  const std::size_t num_local_cells = cell_range.second - cell_range.first;

  // Read a block of cells
  std::vector<std::size_t> topology_data;
  topology_data.reserve(num_local_cells*num_vertices_per_cell);
  HDF5Interface::read_dataset(hdf5_file_id, topology_name, cell_range,
                              topology_data);

  // Look for cell indices in dataset, and use if available
  mesh_data.global_cell_indices.reserve(num_local_cells);
  const std::string cell_indices_name = mesh_name + "/cell_indices";
  if (HDF5Interface::has_dataset(hdf5_file_id, cell_indices_name))
  {
    HDF5Interface::read_dataset(hdf5_file_id, cell_indices_name,
                                cell_range, mesh_data.global_cell_indices);
  }
  else
  {
    for (std::size_t i = 0; i < num_local_cells; i++)
      mesh_data.global_cell_indices.push_back(cell_range.first + i);
  }

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
  const std::pair<std::size_t, std::size_t> vertex_range
    = MPI::local_range(mesh_data.num_global_vertices);
  const std::size_t num_local_vertices
    = vertex_range.second - vertex_range.first;

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
  for (std::size_t i = 0; i < mesh_data.vertex_coordinates.size(); ++i)
    mesh_data.vertex_indices[i] = vertex_range.first + i;

  // Build distributed mesh

  t.stop();

  if (MPI::num_processes() == 1)
    HDF5Utility::build_local_mesh(input_mesh, mesh_data);
  else
    MeshPartitioning::build_distributed_mesh(input_mesh, mesh_data);
}
//-----------------------------------------------------------------------------
bool HDF5File::has_dataset(const std::string dataset_name) const
{
  dolfin_assert(hdf5_file_open);
  return HDF5Interface::has_dataset(hdf5_file_id, dataset_name);
}
//-----------------------------------------------------------------------------
HDF5Attribute HDF5File::attributes(const std::string dataset_name)
{
  dolfin_assert(hdf5_file_open);
  return HDF5Attribute(hdf5_file_id, dataset_name);
}
//-----------------------------------------------------------------------------

#endif
