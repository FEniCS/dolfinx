// Copyright (C) 2012 Chris N. Richardson
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
// First added:  2012-05-22
// Last changed: 2013-02-21

#ifndef __DOLFIN_HDF5FILE_H
#define __DOLFIN_HDF5FILE_H

#ifdef HAS_HDF5

#include <string>
#include <utility>
#include <vector>
#include "dolfin/common/Timer.h"
#include "dolfin/common/Variable.h"
#include "HDF5Interface.h"

namespace dolfin
{

  class Function;
  class GenericVector;
  class LocalMeshData;
  class Mesh;

  class HDF5File : public Variable
  {
  public:

    /// Constructor. file_mode should "a" (append), "w" (write) ot "r"
    /// (read).
    HDF5File(const std::string filename, const std::string file_mode,
             bool use_mpiio=true);

    /// Destructor
    ~HDF5File();

    /// Write Vector to file in a format suitable for re-reading
    void write(const GenericVector& x, const std::string name);

    /// Write Mesh to file in a format suitable for re-reading
    void write(const Mesh& mesh, const std::string name);

    /// Write Mesh of given cell dimension to file 
    /// in a format suitable for re-reading
    void write(const Mesh& mesh, const std::size_t cell_dim,
               const std::string name);

    /// Write Mesh to file for visualisation (may contain duplicate
    /// entities and will not preserve global indices)
    void write_visualisation(const Mesh& mesh, const std::string name);

    /// Write Mesh of given cell dimension to file for visualisation (may
    /// contain duplicate entities and will not preserve global indices)
    void write_visualisation(const Mesh& mesh, const std::size_t cell_dim,
                             const std::string name);

    /// Read vector from file
    void read(GenericVector& x, const std::string dataset_name,
              const bool use_partition_from_file=true);

    /// Read Mesh from file
    void read(Mesh& mesh, const std::string name);

    /// Check if dataset exists in HDF5 file
    bool has_dataset(const std::string dataset_name) const;

    /// Flush buffered I/O to disk
    void flush();

  private:

    // Friend
    friend class XDMFFile;

    // Read a mesh and repartition (if running in parallel)
    void read_mesh_repartition(Mesh &input_mesh,
                               const std::string coordinates_name,
                               const std::string topology_name);

    // Convert LocalMeshData into a Mesh, when running serially
    void build_local_mesh(Mesh &mesh, const LocalMeshData& mesh_data) const;

    // Get description of cells to be written to file
    const std::string cell_type(const std::size_t cell_dim, const Mesh& mesh);

    // Write contiguous data to HDF5 data set. Data is flattened into
    // a 1D array, e.g. [x0, y0, z0, x1, y1, z1] for a vector in 3D
    template <typename T>
    void write_data(const std::string dataset_name,
                    const std::vector<T>& data,
                    const std::vector<std::size_t> global_size);

    // Search dataset names for one beginning with search_term
    static std::string search_list(const std::vector<std::string>& list,
                                   const std::string& search_term);

    // Reorder vertices into global index order, so they can be saved
    // correctly for HDF5 mesh output
    std::vector<double> 
      reorder_vertices_by_global_indices(const Mesh& mesh) const;

    // Reorder data values into global index order
    template <typename T>
    void reorder_values_by_global_indices(const Mesh& mesh, std::vector<T>& data, 
                                          std::vector<std::size_t>& global_size) const;

    // HDF5 file descriptor/handle
    bool hdf5_file_open;
    hid_t hdf5_file_id;

    // Parallel mode
    const bool mpi_io;
  };

  //---------------------------------------------------------------------------

  template <typename T>
  void HDF5File::reorder_values_by_global_indices(const Mesh& mesh, std::vector<T>& data, 
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
    std::vector<std::vector<T> > send_buffer_values(num_processes);
    // Reference to data to send, reorganised as a 2D boost::multi_array
    boost::multi_array_ref<T, 2> data_array(data.data(), boost::extents[mesh.num_vertices()][width]);

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
    std::vector<std::vector<T> > receive_buffer_values;
    MPI::all_to_all(send_buffer_values, receive_buffer_values);

    // Build vectors of ordered values
    std::vector<T> ordered_values(width*(local_range.second - local_range.first));
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
  //---------------------------------------------------------------------------
  template <typename T>
  void HDF5File::write_data(const std::string dataset_name,
                            const std::vector<T>& data,
                            const std::vector<std::size_t> global_size)
  {
    dolfin_assert(hdf5_file_open);

    //FIXME: Get groups from dataset_name and recursively create groups
    const std::string group_name(dataset_name, 0, dataset_name.rfind('/'));

    // Check that group exists and create if required
    if (!HDF5Interface::has_group(hdf5_file_id, group_name))
      HDF5Interface::add_group(hdf5_file_id, group_name);

    dolfin_assert(global_size.size() > 0);

    // Get number of 'items'
    std::size_t num_local_items = 1;
    for (std::size_t i = 1; i < global_size.size(); ++i)
      num_local_items *= global_size[i];
    num_local_items = data.size()/num_local_items;

    // Compute offet
    const std::size_t offset = MPI::global_offset(num_local_items, true);
    std::pair<std::size_t, std::size_t> range(offset, offset + num_local_items);

    const bool chunking = parameters["chunking"];
    // Write data to HDF5 file
    HDF5Interface::write_dataset(hdf5_file_id, dataset_name, data,
                                 range, global_size, mpi_io, chunking);
  }
  //---------------------------------------------------------------------------

}

#endif
#endif
