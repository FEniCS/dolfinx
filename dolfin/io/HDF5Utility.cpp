// Copyright (C) 2013 Chris N. Richardson
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
// Last changed: 2014-02-06

#ifdef HAS_HDF5

#include <iostream>
#include <boost/multi_array.hpp>

#include <dolfin/common/MPI.h>
#include <dolfin/common/Timer.h>
#include <dolfin/fem/GenericDofMap.h>
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
void HDF5Utility::map_gdof_to_cell(
  const MPI_Comm mpi_comm,
  const std::vector<std::size_t>& input_cells,
  const std::vector<dolfin::la_index>& input_cell_dofs,
  const std::vector<std::size_t>& x_cell_dofs,
  const std::pair<dolfin::la_index, dolfin::la_index> vector_range,
  std::vector<std::size_t>& global_cells,
  std::vector<std::size_t>& remote_local_dofi)
{
  // Go through all locally held cells (as read from file)
  // and make mapping from global DOF index back to the cell and local index.
  // Some overwriting will occur if multiple cells refer to the same DOF

  const std::size_t num_processes = MPI::size(mpi_comm);
  std::vector<dolfin::la_index> all_vec_range;
  std::vector<dolfin::la_index> vector_range_second(1, vector_range.second);
  MPI::gather(mpi_comm, vector_range_second, all_vec_range);
  MPI::broadcast(mpi_comm, all_vec_range);

  std::map<dolfin::la_index,
           std::pair<std::size_t, std::size_t>> dof_to_cell;
  const std::size_t offset = x_cell_dofs[0];
  for (std::size_t i = 0; i < x_cell_dofs.size() - 1; ++i)
  {
    for (std::size_t j = x_cell_dofs[i] ; j != x_cell_dofs[i + 1]; ++j)
    {
      const unsigned char local_dof_i = j - x_cell_dofs[i];
      const dolfin::la_index global_dof = input_cell_dofs[j - offset];
      dof_to_cell[global_dof] = std::make_pair(input_cells[i], local_dof_i);
    }
  }

  // Transfer dof_to_cell map to processes which hold the
  // vector data for that DOF
  std::vector<std::vector<dolfin::la_index>> send_dofs(num_processes);
  std::vector<std::vector<std::size_t>> send_cell_dofs(num_processes);
  for (std::map<dolfin::la_index,
        std::pair<std::size_t, std::size_t>>::const_iterator
        p = dof_to_cell.begin(); p != dof_to_cell.end(); ++p)
  {
    const std::size_t dest = std::upper_bound(all_vec_range.begin(),
                                              all_vec_range.end(),
                                              p->first)
                                            - all_vec_range.begin();
    send_dofs[dest].push_back(p->first);
    send_cell_dofs[dest].push_back(p->second.first);
    send_cell_dofs[dest].push_back(p->second.second);
  }

  std::vector<std::vector<dolfin::la_index>> receive_dofs(num_processes);
  std::vector<std::vector<std::size_t>> receive_cell_dofs(num_processes);
  MPI::all_to_all(mpi_comm, send_dofs, receive_dofs);
  MPI::all_to_all(mpi_comm, send_cell_dofs, receive_cell_dofs);

  // Unpack associated cell and local_dofs into vector
  // There may be some overwriting due to receiving an
  // index for the same DOF from multiple cells on different processes

  global_cells.resize(vector_range.second - vector_range.first);
  remote_local_dofi.resize(vector_range.second - vector_range.first);
  for (std::size_t i = 0; i < num_processes; ++i)
  {
    std::vector<dolfin::la_index>& rdofs = receive_dofs[i];
    std::vector<std::size_t>& rcelldofs = receive_cell_dofs[i];
    dolfin_assert(rcelldofs.size() == 2*rdofs.size());
    for (std::size_t j = 0; j < rdofs.size(); ++j)
    {
      dolfin_assert(rdofs[j] >= vector_range.first);
      dolfin_assert(rdofs[j] < vector_range.second);
      global_cells[rdofs[j] - vector_range.first] = rcelldofs[2*j];
      remote_local_dofi[rdofs[j] - vector_range.first] = rcelldofs[2*j + 1];
    }
  }
}
//-----------------------------------------------------------------------------
void HDF5Utility::get_global_dof(
  const MPI_Comm mpi_comm,
  const std::vector<std::pair<std::size_t, std::size_t>>& cell_ownership,
  const std::vector<std::size_t>& remote_local_dofi,
  const std::pair<std::size_t, std::size_t> vector_range,
  const GenericDofMap& dofmap,
  std::vector<dolfin::la_index>& global_dof)
{
  const std::size_t num_processes = MPI::size(mpi_comm);
  std::vector<std::vector<std::size_t>> send_cell_dofs(num_processes);
  const std::size_t n_vector_vals = vector_range.second - vector_range.first;
  global_dof.resize(n_vector_vals);

  for (std::size_t i = 0; i != n_vector_vals; ++i)
  {
    const std::size_t dest = cell_ownership[i].first;
    // Send local index (on remote) and cell local_dof index
    send_cell_dofs[dest].push_back(cell_ownership[i].second);
    send_cell_dofs[dest].push_back(remote_local_dofi[i]);
  }

  std::vector<std::vector<std::size_t>> receive_cell_dofs(num_processes);
  MPI::all_to_all(mpi_comm, send_cell_dofs, receive_cell_dofs);

  std::vector<std::size_t> local_to_global_map;
  dofmap.tabulate_local_to_global_dofs(local_to_global_map);

  // Return back the global dof to the process the request came from
  std::vector<std::vector<dolfin::la_index>>
    send_global_dof_back(num_processes);
  for (std::size_t i = 0; i < num_processes; ++i)
  {
    const std::vector<std::size_t>& rdof = receive_cell_dofs[i];
    for (std::size_t j = 0; j < rdof.size(); j += 2)
    {
      const ArrayView<const dolfin::la_index> dmap = dofmap.cell_dofs(rdof[j]);
      dolfin_assert(rdof[j + 1] < dmap.size());
      const dolfin::la_index local_index = dmap[rdof[j + 1]];
      dolfin_assert(local_index >= 0);
      dolfin_assert((std::size_t)local_index < local_to_global_map.size());
      send_global_dof_back[i].push_back(local_to_global_map[local_index]);
    }
  }

  std::vector<std::vector<dolfin::la_index>>
    receive_global_dof_back(num_processes);
  MPI::all_to_all(mpi_comm, send_global_dof_back, receive_global_dof_back);

  // Go through the received data in the same order as when it was
  // sent out as a request, pulling out the global_dof for each vector
  // position

  std::vector<std::size_t> pos(num_processes, 0);
  for (std::size_t i = 0; i != n_vector_vals; ++i)
  {
    const std::size_t src = cell_ownership[i].first;
    dolfin_assert(src < num_processes);
    const std::vector<dolfin::la_index>& rgdof = receive_global_dof_back[src];
    dolfin_assert(pos[src] < rgdof.size());
    global_dof[i] = rgdof[pos[src]];
    pos[src]++;
  }

}
//-----------------------------------------------------------------------------
std::vector<std::pair<std::size_t, std::size_t>>
HDF5Utility::cell_owners(const Mesh& mesh, const std::vector<std::size_t> cells)
{
  // MPI communicator
  const MPI_Comm mpi_comm = mesh.mpi_comm();

  const std::size_t num_processes = MPI::size(mpi_comm);
  const std::size_t num_global_cells
    = mesh.size_global(mesh.topology().dim());
  const std::pair<std::size_t, std::size_t> cell_range
    = MPI::local_range(mpi_comm, num_global_cells);

  // Find the ownership and local index for
  // all cells in MPI::local_range(num_global_cells)
  std::vector<std::pair<std::size_t, std::size_t>> cell_locations;
  cell_owners_in_range(cell_locations, mesh);

  // Requested cells (given in "cells" argument) are now known to be
  // on the "matching" process given by MPI::index_owner
  std::vector<std::vector<std::size_t>> receive_input_cells(num_processes);
  {
    std::vector<std::vector<std::size_t>> send_input_cells(num_processes);
    for (std::vector<std::size_t>::const_iterator c = cells.begin();
         c != cells.end(); ++c)
    {
      const std::size_t dest = MPI::index_owner(mpi_comm, *c, num_global_cells);
      send_input_cells[dest].push_back(*c);
    }
    MPI::all_to_all(mpi_comm, send_input_cells, receive_input_cells);
  }

  // Reflect back to sending process with cell owner and local index
  std::vector<std::vector<std::size_t>> send_cells(num_processes);
  for (std::size_t i = 0; i != num_processes; ++i)
  {
    const std::vector<std::size_t>& rcells = receive_input_cells[i];
    for (std::size_t j = 0; j < rcells.size(); ++j)
    {
      dolfin_assert(rcells[j] >= cell_range.first);
      dolfin_assert(rcells[j] < cell_range.second);
      std::pair<std::size_t, std::size_t>& loc
        = cell_locations[rcells[j] - cell_range.first];
      send_cells[i].push_back(loc.first);
      send_cells[i].push_back(loc.second);
    }
  }

  std::vector<std::vector<std::size_t>> receive_cells(num_processes);
  MPI::all_to_all(mpi_comm, send_cells, receive_cells);

  std::vector<std::pair<std::size_t, std::size_t>>
    output_cell_locations(cells.size());

  // Index to walk through data reflected back
  std::vector<std::size_t> pos(num_processes, 0);
  for (std::size_t i = 0; i < cells.size(); ++i)
  {
    const std::size_t src = MPI::index_owner(mpi_comm, cells[i],
                                             num_global_cells);
    const std::vector<std::size_t>& rcell = receive_cells[src];
    dolfin_assert(pos[src] < rcell.size()/2);
    output_cell_locations[i].first  = rcell[2*pos[src]];
    output_cell_locations[i].second = rcell[2*pos[src] + 1];
    pos[src]++;
  }

  return output_cell_locations;
}
//-----------------------------------------------------------------------------
void HDF5Utility::cell_owners_in_range(std::vector<std::pair<std::size_t,
                                       std::size_t>>& global_owner,
                                       const Mesh& mesh)
{
  // MPI communicator
  const MPI_Comm mpi_comm = mesh.mpi_comm();

  const std::size_t n_global_cells = mesh.size_global(mesh.topology().dim());
  const std::size_t num_processes = MPI::size(mpi_comm);

  // Communicate global ownership of cells to matching process
  const std::pair<std::size_t, std::size_t> range
    = MPI::local_range(mpi_comm, n_global_cells);
  global_owner.resize(range.second - range.first);

  std::vector<std::vector<std::size_t>> send_owned_global(num_processes);
  for (CellIterator mesh_cell(mesh); !mesh_cell.end(); ++mesh_cell)
  {
    const std::size_t global_i = mesh_cell->global_index();
    const std::size_t local_i = mesh_cell->index();
    const std::size_t po_proc = MPI::index_owner(mpi_comm, global_i,
                                                 n_global_cells);
    send_owned_global[po_proc].push_back(global_i);
    send_owned_global[po_proc].push_back(local_i);
  }

  std::vector<std::vector<std::size_t>> owned_global(num_processes);
  MPI::all_to_all(mpi_comm, send_owned_global, owned_global);

  std::size_t count = 0;
  // Construct mapping from global_index(partial range held) to owning
  // process and remote local_index
  for (std::vector<std::vector<std::size_t>>::iterator owner
         = owned_global.begin(); owner != owned_global.end(); ++owner)
  {
    for (std::vector<std::size_t>::iterator r = owner->begin();
         r != owner->end(); r += 2)
    {
      const std::size_t proc = owner - owned_global.begin();
      const std::size_t idx = *r - range.first;
      global_owner[idx].first = proc;    // owning process
      global_owner[idx].second = *(r+1); // local index on owning process
      count++;
    }
  }

  // All cells in range should be accounted for
  dolfin_assert(count == range.second - range.first);
}
//-----------------------------------------------------------------------------
void HDF5Utility::build_local_mesh(Mesh& mesh, const LocalMeshData& mesh_data)
{
  // NOTE: This function is only used when running in serial

  // Create mesh for editing
  MeshEditor editor;
  dolfin_assert(mesh_data.tdim != 0);
  editor.open(mesh, mesh_data.cell_type, mesh_data.tdim, mesh_data.geometry.dim);

  // Iterate over vertices and add to mesh
  editor.init_vertices_global(mesh_data.geometry.num_global_vertices,
                              mesh_data.geometry.num_global_vertices);
  for (std::int64_t i = 0; i < mesh_data.geometry.num_global_vertices; ++i)
  {
    const std::size_t index = mesh_data.geometry.vertex_indices[i];
    const std::vector<double> coords(mesh_data.geometry.vertex_coordinates[i].begin(),
                                     mesh_data.geometry.vertex_coordinates[i].end());
    Point p(mesh_data.geometry.dim, coords.data());
    editor.add_vertex(index, p);
  }

  // Iterate over cells and add to mesh
  editor.init_cells_global(mesh_data.num_global_cells,
                           mesh_data.num_global_cells);

  for (std::int64_t i = 0; i < mesh_data.num_global_cells; ++i)
  {
    const std::size_t index = mesh_data.global_cell_indices[i];
    const std::vector<std::size_t> v(mesh_data.cell_vertices[i].begin(),
                                     mesh_data.cell_vertices[i].end());
    editor.add_cell(index, v);
  }

  // Close mesh editor
  editor.close();
}
//-----------------------------------------------------------------------------
#endif
