// Copyright (C) 2013 Chris N. Richardson
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "HDF5Utility.h"
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/la/PETScVector.h>
#include <dolfinx/la/utils.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/MeshEntity.h>
#include <dolfinx/mesh/MeshIterator.h>
#include <iostream>
#include <petscvec.h>

using namespace dolfinx;
using namespace dolfinx::io;

//-----------------------------------------------------------------------------
std::pair<std::vector<std::size_t>, std::vector<std::size_t>>
HDF5Utility::map_gdof_to_cell(const MPI_Comm mpi_comm,
                              const std::vector<std::size_t>& input_cells,
                              const std::vector<std::int64_t>& input_cell_dofs,
                              const std::vector<std::int64_t>& x_cell_dofs,
                              const std::array<std::int64_t, 2> vector_range)
{
  // Go through all locally held cells (as read from file)
  // and make mapping from global DOF index back to the cell and local index.
  // Some overwriting will occur if multiple cells refer to the same DOF

  const std::size_t num_processes = MPI::size(mpi_comm);
  std::vector<PetscInt> all_vec_range;
  std::vector<PetscInt> vector_range_second(1, vector_range[1]);
  MPI::gather(mpi_comm, vector_range_second, all_vec_range);
  MPI::broadcast(mpi_comm, all_vec_range);

  std::map<PetscInt, std::pair<std::size_t, std::size_t>> dof_to_cell;
  const std::size_t offset = x_cell_dofs[0];
  for (std::size_t i = 0; i < x_cell_dofs.size() - 1; ++i)
  {
    for (std::int64_t j = x_cell_dofs[i]; j != x_cell_dofs[i + 1]; ++j)
    {
      const unsigned char local_dof_i = j - x_cell_dofs[i];
      const PetscInt global_dof = input_cell_dofs[j - offset];
      dof_to_cell[global_dof] = std::pair(input_cells[i], local_dof_i);
    }
  }

  // Transfer dof_to_cell map to processes which hold the
  // vector data for that DOF
  std::vector<std::vector<PetscInt>> send_dofs(num_processes);
  std::vector<std::vector<std::size_t>> send_cell_dofs(num_processes);
  for (auto p = dof_to_cell.cbegin(); p != dof_to_cell.cend(); ++p)
  {
    const std::size_t dest
        = std::upper_bound(all_vec_range.begin(), all_vec_range.end(), p->first)
          - all_vec_range.begin();
    send_dofs[dest].push_back(p->first);
    send_cell_dofs[dest].push_back(p->second.first);
    send_cell_dofs[dest].push_back(p->second.second);
  }

  std::vector<std::vector<PetscInt>> receive_dofs(num_processes);
  std::vector<std::vector<std::size_t>> receive_cell_dofs(num_processes);
  MPI::all_to_all(mpi_comm, send_dofs, receive_dofs);
  MPI::all_to_all(mpi_comm, send_cell_dofs, receive_cell_dofs);

  // Unpack associated cell and local_dofs into vector There may be some
  // overwriting due to receiving an index for the same DOF from
  // multiple cells on different processes

  std::vector<std::size_t> global_cells(vector_range[1] - vector_range[0]);
  std::vector<std::size_t> remote_local_dofi(vector_range[1] - vector_range[0]);
  for (std::size_t i = 0; i < num_processes; ++i)
  {
    std::vector<PetscInt>& rdofs = receive_dofs[i];
    std::vector<std::size_t>& rcelldofs = receive_cell_dofs[i];
    assert(rcelldofs.size() == 2 * rdofs.size());
    for (std::size_t j = 0; j < rdofs.size(); ++j)
    {
      assert(rdofs[j] >= vector_range[0]);
      assert(rdofs[j] < vector_range[1]);
      global_cells[rdofs[j] - vector_range[0]] = rcelldofs[2 * j];
      remote_local_dofi[rdofs[j] - vector_range[0]] = rcelldofs[2 * j + 1];
    }
  }

  return std::pair(std::move(global_cells), std::move(remote_local_dofi));
}
//-----------------------------------------------------------------------------
std::vector<std::int64_t> HDF5Utility::get_global_dof(
    const MPI_Comm mpi_comm,
    const std::vector<std::pair<std::size_t, std::size_t>>& cell_ownership,
    const std::vector<std::size_t>& remote_local_dofi,
    const std::array<std::int64_t, 2> vector_range, const fem::DofMap& dofmap)
{
  const std::size_t num_processes = MPI::size(mpi_comm);
  std::vector<std::vector<std::size_t>> send_cell_dofs(num_processes);
  const std::size_t n_vector_vals = vector_range[1] - vector_range[0];

  std::vector<std::int64_t> global_dof(n_vector_vals);
  for (std::size_t i = 0; i != n_vector_vals; ++i)
  {
    const std::size_t dest = cell_ownership[i].first;
    // Send local index (on remote) and cell local_dof index
    send_cell_dofs[dest].push_back(cell_ownership[i].second);
    send_cell_dofs[dest].push_back(remote_local_dofi[i]);
  }

  std::vector<std::vector<std::size_t>> receive_cell_dofs(num_processes);
  MPI::all_to_all(mpi_comm, send_cell_dofs, receive_cell_dofs);

  Eigen::Array<std::int64_t, Eigen::Dynamic, 1> local_to_global_map
      = dofmap.index_map->indices(true);

  // Return back the global dof to the process the request came from
  std::vector<std::vector<PetscInt>> send_global_dof_back(num_processes);
  for (std::size_t i = 0; i < num_processes; ++i)
  {
    const std::vector<std::size_t>& rdof = receive_cell_dofs[i];
    for (std::size_t j = 0; j < rdof.size(); j += 2)
    {
      auto dmap = dofmap.cell_dofs(rdof[j]);
      assert(rdof[j + 1] < (std::size_t)dmap.size());
      const PetscInt local_index = dmap[rdof[j + 1]];
      assert(local_index >= 0);
      assert((Eigen::Index)local_index < local_to_global_map.size());
      send_global_dof_back[i].push_back(local_to_global_map[local_index]);
    }
  }

  std::vector<std::vector<PetscInt>> receive_global_dof_back(num_processes);
  MPI::all_to_all(mpi_comm, send_global_dof_back, receive_global_dof_back);

  // Go through the received data in the same order as when it was
  // sent out as a request, pulling out the global_dof for each vector
  // position

  std::vector<std::size_t> pos(num_processes, 0);
  for (std::size_t i = 0; i != n_vector_vals; ++i)
  {
    const std::size_t src = cell_ownership[i].first;
    assert(src < num_processes);
    const std::vector<PetscInt>& rgdof = receive_global_dof_back[src];
    assert(pos[src] < rgdof.size());
    global_dof[i] = rgdof[pos[src]];
    pos[src]++;
  }

  return global_dof;
}
//-----------------------------------------------------------------------------
std::vector<std::pair<std::size_t, std::size_t>>
HDF5Utility::cell_owners(const mesh::Mesh& mesh,
                         const std::vector<std::size_t>& cells)
{
  // MPI communicator
  const MPI_Comm mpi_comm = mesh.mpi_comm();

  const std::size_t num_processes = MPI::size(mpi_comm);
  const std::size_t num_global_cells
      = mesh.num_entities_global(mesh.topology().dim());
  const std::array<std::int64_t, 2> cell_range
      = MPI::local_range(mpi_comm, num_global_cells);

  // Find the ownership and local index for
  // all cells in MPI::local_range(num_global_cells)
  std::vector<std::pair<std::size_t, std::size_t>> cell_locations;
  cell_owners_in_range(cell_locations, mesh);

  // Requested cells (given in "cells" argument) are now known to be
  // on the "matching" process given by MPI::index_owner
  std::vector<std::vector<std::int64_t>> receive_input_cells(num_processes);
  {
    std::vector<std::vector<std::int64_t>> send_input_cells(num_processes);
    for (auto c = cells.begin(); c != cells.end(); ++c)
    {
      const std::size_t dest = MPI::index_owner(mpi_comm, *c, num_global_cells);
      send_input_cells[dest].push_back(*c);
    }
    MPI::all_to_all(mpi_comm, send_input_cells, receive_input_cells);
  }

  // Reflect back to sending process with cell owner and local index
  std::vector<std::vector<std::int64_t>> send_cells(num_processes);
  for (std::size_t i = 0; i != num_processes; ++i)
  {
    const std::vector<std::int64_t>& rcells = receive_input_cells[i];
    for (std::size_t j = 0; j < rcells.size(); ++j)
    {
      assert(rcells[j] >= cell_range[0]);
      assert(rcells[j] < cell_range[1]);
      std::pair<std::size_t, std::size_t>& loc
          = cell_locations[rcells[j] - cell_range[0]];
      send_cells[i].push_back(loc.first);
      send_cells[i].push_back(loc.second);
    }
  }

  std::vector<std::vector<std::int64_t>> receive_cells(num_processes);
  MPI::all_to_all(mpi_comm, send_cells, receive_cells);

  std::vector<std::pair<std::size_t, std::size_t>> output_cell_locations(
      cells.size());

  // Index to walk through data reflected back
  std::vector<std::size_t> pos(num_processes, 0);
  for (std::size_t i = 0; i < cells.size(); ++i)
  {
    const std::size_t src
        = MPI::index_owner(mpi_comm, cells[i], num_global_cells);
    const std::vector<std::int64_t>& rcell = receive_cells[src];
    assert(pos[src] < rcell.size() / 2);
    output_cell_locations[i].first = rcell[2 * pos[src]];
    output_cell_locations[i].second = rcell[2 * pos[src] + 1];
    pos[src]++;
  }

  return output_cell_locations;
}
//-----------------------------------------------------------------------------
void HDF5Utility::cell_owners_in_range(
    std::vector<std::pair<std::size_t, std::size_t>>& global_owner,
    const mesh::Mesh& mesh)
{
  // MPI communicator
  const MPI_Comm mpi_comm = mesh.mpi_comm();

  const int tdim = mesh.topology().dim();
  const std::size_t n_global_cells = mesh.num_entities_global(tdim);
  const std::size_t num_processes = MPI::size(mpi_comm);

  // Communicate global ownership of cells to matching process
  const std::array<std::int64_t, 2> range
      = MPI::local_range(mpi_comm, n_global_cells);
  global_owner.resize(range[1] - range[0]);

  auto map = mesh.topology().index_map(tdim);
  assert(map);
  const std::vector<std::int64_t> global_indices = map->global_indices(false);

  std::vector<std::vector<std::size_t>> send_owned_global(num_processes);
  for (auto& mesh_cell : mesh::MeshRange(mesh, tdim))
  {
    const std::size_t global_i = global_indices[mesh_cell.index()];
    const std::size_t local_i = mesh_cell.index();
    const std::size_t po_proc
        = MPI::index_owner(mpi_comm, global_i, n_global_cells);
    send_owned_global[po_proc].push_back(global_i);
    send_owned_global[po_proc].push_back(local_i);
  }

  std::vector<std::vector<std::size_t>> owned_global(num_processes);
  MPI::all_to_all(mpi_comm, send_owned_global, owned_global);

  std::int64_t count = 0;
  // Construct mapping from global_index(partial range held) to owning
  // process and remote local_index
  for (std::vector<std::vector<std::size_t>>::iterator owner
       = owned_global.begin();
       owner != owned_global.end(); ++owner)
  {
    for (std::vector<std::size_t>::iterator r = owner->begin();
         r != owner->end(); r += 2)
    {
      const std::size_t proc = owner - owned_global.begin();
      const std::size_t idx = *r - range[0];
      global_owner[idx].first = proc;      // owning process
      global_owner[idx].second = *(r + 1); // local index on owning process
      count++;
    }
  }

  // All cells in range should be accounted for
  assert(count == range[1] - range[0]);
}
//-----------------------------------------------------------------------------
void HDF5Utility::set_local_vector_values(
    const MPI_Comm mpi_comm, la::PETScVector& x, const mesh::Mesh& mesh,
    const std::vector<size_t>& cells,
    const std::vector<std::int64_t>& cell_dofs,
    const std::vector<std::int64_t>& x_cell_dofs,
    const std::vector<PetscScalar>& vector,
    const std::array<std::int64_t, 2> input_vector_range,
    const fem::DofMap& dofmap)
{
  // FIXME: Revise to avoid data copying. Set directly in PETSc Vec.

  // Calculate one (global cell, local_dof_index) to associate with
  // each item in the vector on this process
  const auto [global_cells, remote_local_dofi] = HDF5Utility::map_gdof_to_cell(
      mpi_comm, cells, cell_dofs, x_cell_dofs, input_vector_range);

  // At this point, each process has a set of data, and for each
  // value, a global_cell and local_dof to send it to.  However, it is
  // not known which processes the cells are actually on.

  // Find where the needed cells are held
  std::vector<std::pair<std::size_t, std::size_t>> cell_ownership
      = HDF5Utility::cell_owners(mesh, global_cells);

  // Having found the cell location, the actual global_dof index held
  // by that (cell, local_dof) is needed on the process which holds
  // the data values
  std::vector<std::int64_t> global_dof = HDF5Utility::get_global_dof(
      mpi_comm, cell_ownership, remote_local_dofi, input_vector_range, dofmap);

  const std::size_t num_processes = MPI::size(mpi_comm);

  // Shift to dividing things into the vector range of Function Vector
  const std::array<std::int64_t, 2> vector_range = x.local_range();

  std::vector<std::vector<PetscScalar>> receive_values(num_processes);
  std::vector<std::vector<PetscInt>> receive_indices(num_processes);
  {
    std::vector<std::vector<PetscScalar>> send_values(num_processes);
    std::vector<std::vector<PetscInt>> send_indices(num_processes);
    const std::size_t n_vector_vals
        = input_vector_range[1] - input_vector_range[0];
    std::vector<PetscInt> all_vec_range;

    std::vector<PetscInt> vector_range_second(1, vector_range[1]);
    MPI::gather(mpi_comm, vector_range_second, all_vec_range);
    MPI::broadcast(mpi_comm, all_vec_range);

    for (std::size_t i = 0; i != n_vector_vals; ++i)
    {
      const std::size_t dest
          = std::upper_bound(all_vec_range.begin(), all_vec_range.end(),
                             global_dof[i])
            - all_vec_range.begin();
      assert(dest < num_processes);
      assert(i < vector.size());
      send_indices[dest].push_back(global_dof[i]);
      send_values[dest].push_back(vector[i]);
    }

    MPI::all_to_all(mpi_comm, send_values, receive_values);
    MPI::all_to_all(mpi_comm, send_indices, receive_indices);
  }

  std::vector<PetscScalar> vector_values(vector_range[1] - vector_range[0]);
  for (std::size_t i = 0; i != num_processes; ++i)
  {
    const std::vector<PetscScalar>& rval = receive_values[i];
    const std::vector<PetscInt>& rindex = receive_indices[i];
    assert(rval.size() == rindex.size());
    for (std::size_t j = 0; j != rindex.size(); ++j)
    {
      assert(rindex[j] >= vector_range[0]);
      assert(rindex[j] < vector_range[1]);
      vector_values[rindex[j] - vector_range[0]] = rval[j];
    }
  }

  PetscErrorCode ierr;
  PetscScalar* x_ptr = nullptr;
  ierr = VecGetArray(x.vec(), &x_ptr);
  if (ierr != 0)
    la::petsc_error(ierr, __FILE__, "VecGetArray");
  std::copy(vector_values.begin(), vector_values.end(), x_ptr);
  ierr = VecRestoreArray(x.vec(), &x_ptr);
  if (ierr != 0)
    la::petsc_error(ierr, __FILE__, "VecRestoreArray");
}
//-----------------------------------------------------------------------------
