// Copyright (C) 2016 Patrick E. Farrell and Garth N. Wells
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

#ifdef HAS_PETSC

#include <boost/multi_array.hpp>

#include <dolfin/function/FunctionSpace.h>
#include <dolfin/log/log.h>
#include <dolfin/la/PETScMatrix.h>
#include <dolfin/la/PETScVector.h>
#include <dolfin/geometry/BoundingBoxTree.h>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/common/RangedIndexSet.h>
#include <petscmat.h>
#include "PETScDMCollection.h"

using namespace dolfin;

namespace
{
  // Coordinate comparison operator
  struct lt_coordinate
  {
  lt_coordinate(double tolerance) : TOL(tolerance) {}

    bool operator() (const std::vector<double>& x,
                     const std::vector<double>& y) const
    {
      const std::size_t n = std::max(x.size(), y.size());
      for (std::size_t i = 0; i < n; ++i)
      {
        double xx = 0.0;
        double yy = 0.0;
        if (i < x.size())
          xx = x[i];
        if (i < y.size())
          yy = y[i];

        if (xx < (yy - TOL))
          return true;
        else if (xx > (yy + TOL))
          return false;
      }
      return false;
    }

    // Tolerance
    const double TOL;
  };

  std::map<std::vector<double>, std::vector<std::size_t>, lt_coordinate>
    tabulate_coordinates_to_dofs(const FunctionSpace& V)
  {
    std::map<std::vector<double>, std::vector<std::size_t>, lt_coordinate>
      coords_to_dofs(lt_coordinate(1.0e-12));

    // Extract mesh, dofmap and element
    dolfin_assert(V.dofmap());
    dolfin_assert(V.element());
    dolfin_assert(V.mesh());
    const GenericDofMap& dofmap = *V.dofmap();
    const FiniteElement& element = *V.element();
    const Mesh& mesh = *V.mesh();
    std::vector<std::size_t> local_to_global;
    dofmap.tabulate_local_to_global_dofs(local_to_global);

    // Geometric dimension
    const std::size_t gdim = mesh.geometry().dim();

    // Loop over cells and tabulate dofs
    boost::multi_array<double, 2> coordinates;
    std::vector<double> coordinate_dofs;
    std::vector<double> coors(gdim);

    // Speed up the computations by only visiting (most) dofs once
    const std::size_t local_size = dofmap.ownership_range().second
      - dofmap.ownership_range().first;
    RangedIndexSet already_visited({0, local_size});

    for (CellIterator cell(mesh); !cell.end(); ++cell)
    {
      // Update UFC cell
      cell->get_coordinate_dofs(coordinate_dofs);

      // Get local-to-global map
      const ArrayView<const dolfin::la_index> dofs
        = dofmap.cell_dofs(cell->index());

      // Tabulate dof coordinates on cell
      element.tabulate_dof_coordinates(coordinates, coordinate_dofs, *cell);

      // Map dofs into coords_to_dofs
      for (std::size_t i = 0; i < dofs.size(); ++i)
      {
        const std::size_t dof = dofs[i];
        if (dof < local_size)
        {
          // Skip already checked dofs
          if (!already_visited.insert(dof))
            continue;

          // Put coordinates in coors
          std::copy(coordinates[i].begin(), coordinates[i].end(), coors.begin());

          // Add dof to list at this coord
          const auto ins = coords_to_dofs.insert({coors, {local_to_global[dof]}});

          if (!ins.second)
            ins.first->second.push_back(local_to_global[dof]);
        }
      }
    }
    return coords_to_dofs;
  }
}
//-----------------------------------------------------------------------------
void PETScDMCollection::find_exterior_points(MPI_Comm mpi_comm,
     std::shared_ptr<const BoundingBoxTree> treec,
     int dim, int data_size,
     const std::vector<double>& send_points,
     const std::vector<int>& send_indices,
     std::vector<int>& indices,
     std::vector<std::size_t>& cell_ids,
     std::vector<double>& points)
{
  dolfin_assert(send_indices.size()/data_size == send_points.size()/dim);
  const boost::const_multi_array_ref<int, 2> send_indices_arr(send_indices.data(),
                                                  boost::extents[send_indices.size()/data_size][data_size]);

  unsigned int mpi_rank = MPI::rank(mpi_comm);
  unsigned int mpi_size = MPI::size(mpi_comm);

  // Get all points on all processes
  std::vector<std::vector<double>> recv_points(mpi_size);
  MPI::all_gather(mpi_comm, send_points, recv_points);

  unsigned int num_recv_points = 0;
  for (auto &p : recv_points)
    num_recv_points += p.size();
  num_recv_points /= dim;

  // Save distances and ids of nearest cells on this
  // process
  std::vector<double> send_distance;
  std::vector<unsigned int> ids;

  send_distance.reserve(num_recv_points);
  ids.reserve(num_recv_points);

  for (const auto &p : recv_points)
  {
    unsigned int n_points = p.size()/dim;
    for (unsigned int i = 0; i < n_points; ++i)
    {
      const Point curr_point(dim, &p[i*dim]);
      std::pair<unsigned int, double> find_point
        = treec->compute_closest_entity(curr_point);
      send_distance.push_back(find_point.second);
      ids.push_back(find_point.first);
    }
  }

  // All processes get the same distance information
  std::vector<double> recv_distance(num_recv_points*mpi_size);
  MPI::all_gather(mpi_comm, send_distance, recv_distance);

  // Determine which process has closest cell for each
  // point, and send the global indices to that process
  int ct = 0;
  std::vector<std::vector<unsigned int>> send_global_indices(mpi_size);

  for (unsigned int p = 0; p != mpi_size; ++p)
  {
    unsigned int n_points = recv_points[p].size()/dim;
    boost::multi_array_ref<double, 2>
      point_arr(recv_points[p].data(),
                boost::extents[n_points][dim]);
    for (unsigned int i = 0; i < n_points; ++i)
    {
      unsigned int min_proc = 0;
      double min_val = recv_distance[ct];
      for (unsigned int q = 1; q != mpi_size; ++q)
      {
        const double val
          = recv_distance[q*num_recv_points + ct];
        if (val < min_val)
        {
          min_val = val;
          min_proc = q;
        }
      }

      if (min_proc == mpi_rank)
      {
        // If this process has closest cell,
        // save the information
        points.insert(points.end(),
                      point_arr[i].begin(),
                      point_arr[i].end());
        cell_ids.push_back(ids[ct]);
      }
      if (p == mpi_rank)
      {
        send_global_indices[min_proc]
          .insert(send_global_indices[min_proc].end(),
                  send_indices_arr[i].begin(),
                  send_indices_arr[i].end());
      }
      ++ct;
    }
  }

  // Send out global indices for the points provided by this process
  std::vector<std::vector<unsigned int>> recv_global_indices(mpi_size);
  MPI::all_to_all(mpi_comm, send_global_indices, recv_global_indices);

  for (auto &p : recv_global_indices)
    indices.insert(indices.end(), p.begin(), p.end());

}
//-----------------------------------------------------------------------------
std::shared_ptr<PETScMatrix> PETScDMCollection::create_transfer_matrix
(std::shared_ptr<const FunctionSpace> coarse_space,
 std::shared_ptr<const FunctionSpace> fine_space)
{
  // Get coarse mesh and dimension of the domain
  dolfin_assert(coarse_space->mesh());
  const Mesh meshc = *coarse_space->mesh();
  std::size_t dim = meshc.geometry().dim();

  // MPI communicator, size and rank
  const MPI_Comm mpi_comm = meshc.mpi_comm();
  const unsigned int mpi_size = MPI::size(mpi_comm);

  // Initialise bounding box tree and dofmaps
  std::shared_ptr<BoundingBoxTree> treec = meshc.bounding_box_tree();
  std::shared_ptr<const GenericDofMap> coarsemap = coarse_space->dofmap();
  std::shared_ptr<const GenericDofMap> finemap = fine_space->dofmap();

  // Create map from coordinates to dofs sharing that coordinate
  std::map<std::vector<double>, std::vector<std::size_t>, lt_coordinate>
    coords_to_dofs = tabulate_coordinates_to_dofs(*fine_space);

  // Global dimensions of the dofs and of the transfer matrix (M-by-N, where
  // M is the fine space dimension, N is the coarse space dimension)
  std::size_t M = fine_space->dim();
  std::size_t N = coarse_space->dim();

  // Local dimension of the dofs and of the transfer matrix
  std::size_t m = finemap->dofs().size();
  std::size_t n = coarsemap->dofs().size();

  // Get finite element for the coarse space. This will be needed to evaluate
  // the basis functions for each cell.
  std::shared_ptr<const FiniteElement> el = coarse_space->element();

  // Check that it is the same kind of element on each space.
  {
    std::shared_ptr<const FiniteElement> elf = fine_space->element();
    // Check that function ranks match
    if (el->value_rank() != elf->value_rank())
    {
      dolfin_error("create_transfer_matrix",
                   "Creating interpolation matrix",
                   "Ranks of function spaces do not match: %d, %d.",
                   el->value_rank(), elf->value_rank());
    }

    // Check that function dims match
    for (std::size_t i = 0; i < el->value_rank(); ++i)
    {
      if (el->value_dimension(i) != elf->value_dimension(i))
      {
        dolfin_error("create_transfer_matrix",
                     "Creating interpolation matrix",
                     "Dimension %d of function space (%d) does not match dimension %d of function space (%d)",
                     i, el->value_dimension(i), i, elf->value_dimension(i));
      }
    }
  }
  // number of dofs per cell for the finite element.
  std::size_t eldim = el->space_dimension();
  // Number of dofs associated with each fine point
  unsigned int data_size = 1;
  for (unsigned data_dim = 0; data_dim < el->value_rank(); data_dim++)
    data_size *= el->value_dimension(data_dim);

  // The overall idea is: a fine point can be on a coarse cell in the current processor,
  // on a coarse cell in a different processor, or outside the coarse domain.
  // If the point is found on the processor, evaluate basis functions,
  // if found elsewhere, use the other processor to evaluate basis functions,
  // if not found at all, or if found in multiple processors,
  // use compute_closest_entity on all processors and find
  // which coarse cell is the closest entity to the fine point amongst all processors.

  // vector containing the ranks of the processors which might contain
  // a fine point

  // the next vectors we are defining here contain information relative to the
  // fine points for which a corresponding coarse cell owned by the current
  // processor was found.
  // found_ids[i] contains the coarse cell id relative to each fine point
  std::vector<std::size_t> found_ids;
  found_ids.reserve((std::size_t)M/mpi_size);
  // found_points[dim*i:dim*i + dim] contains the coordinates of the fine point i
  std::vector<double> found_points;
  found_points.reserve((std::size_t)dim*M/mpi_size);
  // global_row_indices[i] contains the global row indices of the fine point i
  // global_row_indices[data_size*i:data_size*i + data_size] are the rows associated with
  // this point
  std::vector<int> global_row_indices;
  global_row_indices.reserve((std::size_t) data_size*M/mpi_size);

  // Collect up any points which lie outside the domain
  std::vector<double> exterior_points;
  std::vector<int> exterior_global_indices;

  // Send out points which are found in remote BBox
  std::vector<std::vector<double>> send_found(mpi_size);
  std::vector<std::vector<int>> send_found_global_row_indices(mpi_size);

  std::vector<int> proc_list;
  std::vector<unsigned int> found_ranks;
  for (const auto &map_it : coords_to_dofs)
  {
    const std::vector<double>& _x = map_it.first;
    Point curr_point(dim, _x.data());

    // Compute which processes' BBoxes contain the fine point
    found_ranks = treec->compute_process_collisions(curr_point);

    if (found_ranks.empty())
    {
      // Point is outside the domain
      exterior_points.insert(exterior_points.end(), _x.begin(), _x.end());
      exterior_global_indices.insert(exterior_global_indices.end(),
                                     map_it.second.begin(),
                                     map_it.second.end());
    }
    else
    {
      // Send points to candidate processes,
      // also recording the processes they are sent to in proc_list
      proc_list.push_back(found_ranks.size());
      for (auto &rp : found_ranks)
      {
        proc_list.push_back(rp);
        send_found[rp].insert(send_found[rp].end(),
                              _x.begin(), _x.end());
        send_found_global_row_indices[rp].insert(
         send_found_global_row_indices[rp].end(),
         map_it.second.begin(), map_it.second.end());
      }
    }
  }
  std::vector<std::vector<double>> recv_found(mpi_size);
  MPI::all_to_all(mpi_comm, send_found, recv_found);

  // On remote process, convert received points to ID of containing cell and send back
  // to originating process
  std::vector<std::vector<unsigned int>> send_ids(mpi_size);
  for (unsigned int p = 0; p < mpi_size; ++p)
  {
    unsigned int n_points = recv_found[p].size()/dim;
    for (unsigned int i = 0; i < n_points; ++i)
    {
      const Point curr_point(dim, &recv_found[p][i*dim]);
      send_ids[p].push_back(treec->compute_first_entity_collision(curr_point));
    }
  }

  std::vector<std::vector<unsigned int>> recv_ids(mpi_size);
  MPI::all_to_all(mpi_comm, send_ids, recv_ids);

  // Revisit original list of sent points in same order
  std::vector<int> count(mpi_size, 0);
  for (auto p = proc_list.begin(); p != proc_list.end(); p += (*p + 1))
  {
    unsigned int nprocs = *p;
    int owner = -1;
    // Find first process which owns a cell containing the point
    for (unsigned int j = 1; j != (nprocs + 1); ++j)
    {
      const int proc = *(p + j);
      const unsigned int id = recv_ids[proc][count[proc]];
      if (id != std::numeric_limits<unsigned int>::max())
      {
        owner = proc;
        break;
      }
    }

    if (owner == -1)
    {
      // Point not found remotely, so add to not_found list
      int proc = *(p + 1);
      exterior_points.insert(exterior_points.end(),
                             &send_found[proc][count[proc]*dim],
                             &send_found[proc][(count[proc] + 1)*dim]);
      exterior_global_indices.insert(exterior_global_indices.end(),
            &send_found_global_row_indices[proc][count[proc]*data_size],
            &send_found_global_row_indices[proc][(count[proc] + 1)*data_size]);
    }
    else if (nprocs > 1)
    {
      // If point is found on multiple processes, send -1 as the index to the
      // remote processes which are not the "owner"
      for (unsigned int j = 1; j != (nprocs + 1); ++j)
      {
        const int proc = *(p + j);
        if (proc != owner)
        {
          for (unsigned int k = 0; k != data_size; ++k)
            send_found_global_row_indices[proc]
              [count[proc]*data_size + k] = -1;
        }
      }
    }

    // Move to next point
    for (unsigned int j = 1; j != (nprocs + 1); ++j)
      ++count[*(p + j)];
  }

  // Finally, send indices
  std::vector<std::vector<int>> recv_found_global_row_indices(mpi_size);
  MPI::all_to_all(mpi_comm, send_found_global_row_indices,
                  recv_found_global_row_indices);

  // Flatten results ready for insertion
  for (unsigned int p = 0; p != mpi_size; ++p)
  {
    const auto& id_p = send_ids[p];
    const unsigned int npoints = id_p.size();
    dolfin_assert(npoints == recv_found[p].size()/dim);
    dolfin_assert(npoints == recv_found_global_row_indices[p].size()/data_size);

    const boost::multi_array_ref<double, 2>
      point_p(recv_found[p].data(), boost::extents[npoints][dim]);
    const boost::multi_array_ref<int, 2>
      global_idx_p(recv_found_global_row_indices[p].data(),
                   boost::extents[npoints][data_size]);

    for (unsigned int i = 0; i < npoints; ++i)
    {
      if (id_p[i] != std::numeric_limits<unsigned int>::max()
          and global_idx_p[i][0] != -1)
      {
        found_ids.push_back(id_p[i]);
        global_row_indices.insert(global_row_indices.end(),
                                  global_idx_p[i].begin(),
                                  global_idx_p[i].end());

        found_points.insert(found_points.end(),
                            point_p[i].begin(), point_p[i].end());
      }
    }
  }

  // Find closest cells for points that lie outside the domain
  // and add them to the lists
  find_exterior_points(mpi_comm, treec, dim, data_size,
                       exterior_points,
                       exterior_global_indices,
                       global_row_indices,
                       found_ids,
                       found_points);

  // Now every processor should have the information needed to
  // assemble its portion of the matrix.  The ids of coarse cell owned
  // by each processor are currently stored in found_ids
  // and their respective global row indices are stored in global_row_indices.
  // One last loop and we are ready to go!

  // m_owned is the number of rows the current processor needs to set
  // note that the processor might not own these rows
  const std::size_t m_owned = global_row_indices.size();

  // Initialise row and column indices and values of the transfer matrix
  std::vector<std::vector<int>> col_indices(m_owned, std::vector<int>(eldim));
  std::vector<std::vector<double>> values(m_owned, std::vector<double>(eldim));
  std::vector<double> temp_values(eldim*data_size);

  // Initialise global sparsity pattern: record on-process and
  // off-process dependencies of fine dofs
  std::vector<std::vector<dolfin::la_index>> send_dnnz(mpi_size);
  std::vector<std::vector<dolfin::la_index>> send_onnz(mpi_size);

  // Initialise local to global dof maps (needed to allocate
  // the entries of the transfer matrix with the correct global indices)
  std::vector<std::size_t> coarse_local_to_global_dofs;
  coarsemap->tabulate_local_to_global_dofs(coarse_local_to_global_dofs);

  std::vector<double> coordinate_dofs; // cell dofs coordinates vector
  ufc::cell ufc_cell; // ufc cell

  // Loop over the found coarse cells
  for (unsigned int i = 0; i < found_ids.size(); ++i)
  {
    // Get coarse cell id and point
    unsigned int id = found_ids[i];
    Point curr_point(dim, &found_points[i*dim]);

    // Create coarse cell
    Cell coarse_cell(meshc, static_cast<std::size_t>(id));
    // Get dofs coordinates of the coarse cell
    coarse_cell.get_coordinate_dofs(coordinate_dofs);
    // Save cell information into the ufc cell
    coarse_cell.get_cell_data(ufc_cell);
    // Evaluate the basis functions of the coarse cells
    // at the fine point and store the values into temp_values
    el->evaluate_basis_all(temp_values.data(),
                           curr_point.coordinates(),
                           coordinate_dofs.data(),
                           ufc_cell.orientation);

    // Get the coarse dofs associated with this cell
    ArrayView<const dolfin::la_index> temp_dofs = coarsemap->cell_dofs(id);

    // Loop over the fine dofs associated with this collision
    for (unsigned k = 0; k < data_size; k++)
    {
      const unsigned int fine_row = i*data_size + k;
      const std::size_t global_fine_dof = global_row_indices[fine_row];
      int p = finemap->index_map()->global_index_owner(global_fine_dof/data_size);

      // Loop over the coarse dofs and stuff their contributions
      for (unsigned j = 0; j < eldim; j++)
      {
        const std::size_t coarse_dof
          = coarse_local_to_global_dofs[temp_dofs[j]];

        // Set the column
        col_indices[fine_row][j] = coarse_dof;
        // Set the value
        values[fine_row][j] = temp_values[data_size*j + k];

        int pc = coarsemap->index_map()->global_index_owner(coarse_dof/data_size);
        if (p == pc)
          send_dnnz[p].push_back(global_fine_dof);
        else
          send_onnz[p].push_back(global_fine_dof);

      }
    }
  }

  // Communicate off-process columns nnz, and flatten to get nnz per row
  // we also keep track of the ownership range
  std::size_t mbegin = finemap->ownership_range().first;
  std::size_t mend = finemap->ownership_range().second;
  std::vector<std::vector<dolfin::la_index>> recv_onnz;
  MPI::all_to_all(mpi_comm, send_onnz, recv_onnz);
  std::vector<int> onnz(m, 0);
  for (const auto &p : recv_onnz)
    for (const auto &q : p)
    {
      dolfin_assert(q >= (dolfin::la_index)mbegin
                    and q < (dolfin::la_index)mend);
      ++onnz[q - mbegin];
    }

  // Communicate on-process columns nnz, and flatten to get nnz per row
  std::vector<std::vector<dolfin::la_index>> recv_dnnz;
  MPI::all_to_all(mpi_comm, send_dnnz, recv_dnnz);
  std::vector<int> dnnz(m, 0);
  for (const auto &p : recv_dnnz)
    for (const auto &q : p)
    {
      dolfin_assert(q >= (dolfin::la_index)mbegin
                    and q < (dolfin::la_index)mend);
      ++dnnz[q - mbegin];
    }

  // Initialise PETSc Mat and error code
  PetscErrorCode ierr;
  Mat I;

  // Create and initialise the transfer matrix as MATMPIAIJ/MATSEQAIJ
  ierr = MatCreate(mpi_comm, &I); CHKERRABORT(PETSC_COMM_WORLD, ierr);
  if (mpi_size > 1)
  {
    ierr = MatSetType(I, MATMPIAIJ); CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = MatSetSizes(I, m, n, M, N); CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = MatMPIAIJSetPreallocation(I, PETSC_DEFAULT, dnnz.data(), PETSC_DEFAULT, onnz.data());
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
  }
  else
  {
    ierr = MatSetType(I, MATSEQAIJ); CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = MatSetSizes(I, m, n, M, N); CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = MatSeqAIJSetPreallocation(I, PETSC_DEFAULT, dnnz.data());
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
  }

  // Setting transfer matrix values row by row
  for (unsigned int fine_row = 0; fine_row < m_owned; ++fine_row)
  {
    PetscInt fine_dof = global_row_indices[fine_row];
    ierr = MatSetValues(I, 1, &fine_dof, eldim, col_indices[fine_row].data(), values[fine_row].data(), INSERT_VALUES);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
  }

  // Assemble the transfer matrix
  ierr = MatAssemblyBegin(I, MAT_FINAL_ASSEMBLY); CHKERRABORT(PETSC_COMM_WORLD, ierr);
  ierr = MatAssemblyEnd(I, MAT_FINAL_ASSEMBLY); CHKERRABORT(PETSC_COMM_WORLD, ierr);

  // create shared pointer and return the pointer to the transfer matrix
  std::shared_ptr<PETScMatrix> ptr = std::make_shared<PETScMatrix>(I);
  ierr = MatDestroy(&I); CHKERRABORT(PETSC_COMM_WORLD, ierr);
  return ptr;
}
//-----------------------------------------------------------------------------
PETScDMCollection::PETScDMCollection(std::vector<std::shared_ptr<const FunctionSpace>> function_spaces)
  : _spaces(function_spaces), _dms(function_spaces.size(), nullptr)
{
  for (std::size_t i = 0; i < _spaces.size(); ++i)
  {
    dolfin_assert(_spaces[i]);

    // Get MPI communicator from Mesh
    dolfin_assert(_spaces[i]->mesh());
    MPI_Comm comm = _spaces[i]->mesh()->mpi_comm();

    // Create DM
    DMShellCreate(comm, &_dms[i]);
    DMShellSetContext(_dms[i], (void*)&_spaces[i]);

    // Suppy function to create global vector on DM
    DMShellSetCreateGlobalVector(_dms[i],
                                 PETScDMCollection::create_global_vector);

    // Supply function to create interpolation matrix (coarse-to-fine
    // interpolation, i.e. level n to level n+1)
    DMShellSetCreateInterpolation(_dms[i],
                                  PETScDMCollection::create_interpolation);
  }

  for (std::size_t i = 0; i < _spaces.size() - 1; i++)
  {
    // Set the fine 'mesh' associated with _dms[i]
    DMSetFineDM(_dms[i], _dms[i + 1]);
    DMShellSetRefine(_dms[i], PETScDMCollection::refine);
  }

  for (std::size_t i = 1; i < _spaces.size(); i++)
  {
    // Set the coarse 'mesh' associated with _dms[i]
    DMSetCoarseDM(_dms[i], _dms[i - 1]);
    DMShellSetCoarsen(_dms[i], PETScDMCollection::coarsen);
  }
}
//-----------------------------------------------------------------------------
PETScDMCollection::~PETScDMCollection()
{
  // Don't destroy all the DMs!
  // Only destroy the finest one.
  // This is highly counter-intuitive, and possibly a bug in PETSc,
  // but it took Garth and Patrick an entire day to figure out.
  if (!_dms.empty())
    DMDestroy(&_dms.back());
}
//-----------------------------------------------------------------------------
DM PETScDMCollection::get_dm(int i)
{
  dolfin_assert(i >= -(int)_dms.size() and i < (int) _dms.size());
  const int base = i < 0 ? _dms.size() : 0;
  return _dms[base + i];
}
//-----------------------------------------------------------------------------
void PETScDMCollection::check_ref_count() const
{
  for (std::size_t i = 0; i < _dms.size(); ++i)
  {
    PetscInt cnt = 0;
    PetscObjectGetReference((PetscObject)_dms[i], &cnt);
    std::cout << "Ref count " << i << ": " << cnt << std::endl;
  }
}
//-----------------------------------------------------------------------------
void PETScDMCollection::reset(int i)
{
  PetscObjectDereference((PetscObject)_dms[i]);
  //PetscObjectDereference((PetscObject)_dms.back());
  //for (std::size_t i = 0; i < _dms.size(); ++i)
  //  PetscObjectDereference((PetscObject)_dms[i]);
}
//-----------------------------------------------------------------------------
PetscErrorCode PETScDMCollection::create_global_vector(DM dm, Vec* vec)
{
  // Get DOLFIN FunctiobSpace from the PETSc DM object
  std::shared_ptr<FunctionSpace> *V;
  DMShellGetContext(dm, (void**)&V);

  // Create Vector
  Function u(*V);
  *vec = u.vector()->down_cast<PETScVector>().vec();

  // FIXME: Does increasing the reference count lead to a memory leak?
  // Increment PETSc reference counter the Vec
  PetscObjectReference((PetscObject)*vec);

  return 0;
}
//-----------------------------------------------------------------------------
PetscErrorCode PETScDMCollection::create_interpolation(DM dmc, DM dmf, Mat *mat,
                                                       Vec *vec)
{
  // Get DOLFIN FunctionSpaces from PETSc DM objects (V0 is coarse
  // space, V1 is fine space)
  std::shared_ptr<FunctionSpace> *V0, *V1;
  DMShellGetContext(dmc, (void**)&V0);
  DMShellGetContext(dmf, (void**)&V1);

  // Build interpolation matrix (V0 to V1)
  std::shared_ptr<PETScMatrix> P = create_transfer_matrix(*V0, *V1);

  // Copy PETSc matrix pointer and inrease reference count
  *mat = P->mat();
  PetscObjectReference((PetscObject)*mat);

  // Set optional vector to NULL
  *vec = NULL;

  return 0;
}
//-----------------------------------------------------------------------------
PetscErrorCode PETScDMCollection::coarsen(DM dmf, MPI_Comm comm, DM* dmc)
{
  // Get the coarse DM from the fine DM
  return DMGetCoarseDM(dmf, dmc);
}
//-----------------------------------------------------------------------------
PetscErrorCode PETScDMCollection::refine(DM dmc, MPI_Comm comm, DM* dmf)
{
  // Get the fine DM from the coarse DM
  return DMGetFineDM(dmc, dmf);
}
//-----------------------------------------------------------------------------
#endif
