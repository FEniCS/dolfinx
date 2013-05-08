// Copyright (C) 2013 Chris N Richardson
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
// First added:  2013-05-08
// Last changed: 2013-05-08

#ifdef HAS_HDF5

#include <iostream>
#include <boost/multi_array.hpp>

#include <dolfin/common/MPI.h>
#include <dolfin/common/Timer.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/LocalMeshData.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshEditor.h>
#include <dolfin/mesh/MeshEntityIterator.h>
#include <dolfin/mesh/Vertex.h>

#include "HDF5Utility.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void HDF5Utility::compute_global_mapping(std::vector<std::pair<std::size_t, std::size_t> >& global_owner, const Mesh& mesh)
{
  const std::size_t n_global_cells = mesh.size_global(mesh.topology().dim());
  const std::size_t num_processes = MPI::num_processes();

  // Communicate global ownership of cells to matching process
  const std::pair<std::size_t, std::size_t> range = MPI::local_range(n_global_cells);
  global_owner.resize(range.second - range.first);

  std::vector<std::vector<std::size_t> > send_owned_global(num_processes);
  std::vector<std::vector<std::size_t> > owned_global(num_processes);
  for(CellIterator mesh_cell(mesh); !mesh_cell.end(); ++mesh_cell)
  {
    const std::size_t global_i = mesh_cell->global_index();
    const std::size_t local_i = mesh_cell->index();
    const std::size_t po_proc = MPI::index_owner(global_i, n_global_cells);
    send_owned_global[po_proc].push_back(global_i);
    send_owned_global[po_proc].push_back(local_i);
  }
  MPI::all_to_all(send_owned_global, owned_global);

  std::size_t count = 0;
  // Construct mapping from global_index(partial range held) to owning process and remote local_index
  for(std::vector<std::vector<std::size_t> >::iterator owner = owned_global.begin();
      owner != owned_global.end(); ++owner)
    for(std::vector<std::size_t>::iterator r = owner->begin(); r != owner->end(); r += 2)
    {
      const std::size_t proc = owner - owned_global.begin();
      const std::size_t idx = *r - range.first;
      global_owner[idx].first = proc;    // owning process
      global_owner[idx].second = *(r+1); // local index on owning process
      count++;
    }
    // All cells in range should be accounted for
  dolfin_assert(count == range.second - range.first);
}

//--------------------------------------------------------------------------------
void HDF5Utility::build_local_mesh(Mesh &mesh, const LocalMeshData& mesh_data)
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
std::vector<double> HDF5Utility::reorder_vertices_by_global_indices(const Mesh& mesh)
{
  std::vector<std::size_t> global_size(2);
  global_size[0] = MPI::sum(mesh.num_vertices()); //including duplicates
  global_size[1] = mesh.geometry().dim();

  std::vector<double> ordered_coordinates(mesh.coordinates());
  reorder_values_by_global_indices(mesh, ordered_coordinates, global_size);
  return ordered_coordinates;
}
//---------------------------------------------------------------------------
void HDF5Utility::reorder_values_by_global_indices(const Mesh& mesh, std::vector<double>& data, 
                                                   std::vector<std::size_t>& global_size)
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

#endif
