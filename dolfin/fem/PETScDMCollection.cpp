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
std::shared_ptr<PETScMatrix> PETScDMCollection::create_transfer_matrix
(std::shared_ptr<const FunctionSpace> coarse_space,
 std::shared_ptr<const FunctionSpace> fine_space)
{
  // Initialise PETSc Mat and error code
  PetscErrorCode ierr;
  Mat I;

  // Get coarse mesh and dimension of the domain
  dolfin_assert(coarse_space->mesh());
  const Mesh meshc = *coarse_space->mesh();
  std::size_t dim = meshc.geometry().dim();

  // MPI commpunicator, size and rank
  const MPI_Comm mpi_comm = meshc.mpi_comm();
  const unsigned int mpi_size = MPI::size(mpi_comm);
  const unsigned int mpi_rank = MPI::rank(mpi_comm); // mpi_rank is the rank of the current processor

  // Create and initialise the transfer matrix as MATMPIAIJ/MATSEQAIJ
  ierr = MatCreate(mpi_comm, &I); CHKERRABORT(PETSC_COMM_WORLD, ierr);
  if (mpi_size > 1)
  {
    ierr = MatSetType(I, MATMPIAIJ); CHKERRABORT(PETSC_COMM_WORLD, ierr);
  }
  else
  {
    ierr = MatSetType(I, MATSEQAIJ); CHKERRABORT(PETSC_COMM_WORLD, ierr);
  }

  // initialise bounding box tree and dofmaps
  std::shared_ptr<BoundingBoxTree> treec = meshc.bounding_box_tree();
  std::shared_ptr<const GenericDofMap> coarsemap = coarse_space->dofmap();
  std::shared_ptr<const GenericDofMap> finemap = fine_space->dofmap();

  // initialise local to global dof maps (these will be needed to allocate
  // the entries of the transfer matrix with the correct global indices)
  std::vector<std::size_t> coarse_local_to_global_dofs;
  coarsemap->tabulate_local_to_global_dofs(coarse_local_to_global_dofs);

  // Create map from coordinates to dofs sharing that coordinate
  std::map<std::vector<double>, std::vector<std::size_t>, lt_coordinate>
    coords_to_dofs = tabulate_coordinates_to_dofs(*fine_space);

  // Global dimensions of the dofs and of the transfer matrix (M-by-N, where
  // M is the fine space dimension, N is the coarse space dimension)
  std::size_t M = fine_space->dim();
  std::size_t N = coarse_space->dim();

  // Local dimension of the dofs and of the transfer matrix
  // we also keep track of the ownership range
  std::size_t mbegin = finemap->ownership_range().first;
  std::size_t mend = finemap->ownership_range().second;
  std::size_t m = finemap->dofs().size();

  std::size_t nbegin = coarsemap->ownership_range().first;
  std::size_t nend = coarsemap->ownership_range().second;
  std::size_t n = coarsemap->dofs().size();

  // we store the ownership range of the fine dofs so that
  // we can communicate it to the other workers.
  // This will be useful to check which dofs are owned by which processor
  std::vector<std::size_t> global_n_range(2,0);
  global_n_range[0] = nbegin;
  global_n_range[1] = nend;

  // Set transfer matrix sizes
  ierr = MatSetSizes(I, m, n, M, N); CHKERRABORT(PETSC_COMM_WORLD, ierr);

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

  // Miscellaneous initialisations, these will all be needed later
  Point curr_point; // point variable
  Cell curr_cell; // cell variable
  std::vector<double> coordinate_dofs; // cell dofs coordinates vector
  ufc::cell ufc_cell; // ufc cell
  unsigned int id = 0; // cell id

  // The overall idea is: a fine point can be on a coarse cell in the current processor,
  // on a coarse cell in a different processor, or outside the coarse domain.
  // If the point is found on the processor, evaluate basis functions,
  // if found elsewhere, use the other processor to evaluate basis functions,
  // if not found at all, or if found in multiple processors,
  // use compute_closest_entity on all processors and find
  // which coarse cell is the closest entity to the fine point amongst all processors.

  std::vector<double> _x(dim); // vector with point coordinates
  // vector containing the ranks of the processors which might contain a fine point
  std::vector<unsigned int> found_ranks;

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
  // found_points_senders[i] holds the rank of the process that owns the fine point
  // this is not strictly needed, but it is useful
  std::vector<std::size_t> found_points_senders;
  found_points_senders.reserve((std::size_t)M/mpi_size);

  // similar stuff for the not found points
  std::vector<unsigned int> not_found_points_senders;
  not_found_points_senders.reserve((std::size_t)M/mpi_size);
  std::vector<double> not_found_points;
  std::vector<int> not_found_global_row_indices;

  // same for the found elsewhere points
  std::vector<double> found_elsewhere;
  std::vector<int> found_elsewhere_global_row_indices;
  // which_processor contains the rank of the processor
  // that owns the coarse cell where the fine point was found
  std::vector<unsigned int> which_processor;

  // Loop over fine points owned by the current processor,
  // and find out who owns the coarse cell where the fine point lies.
  for (const auto &map_it : coords_to_dofs)
  {
    // Copy coordinates into buffer.
    std::copy(map_it.first.begin(), map_it.first.end(), _x.begin());

    // get the fine point into a Point variable
    if (dim == 3)
      curr_point = Point(_x[0], _x[1], _x[2]);
    else if (dim == 2)
      curr_point = Point(_x[0], _x[1]);
    else
      curr_point = Point(_x[0]);

    // compute which processors share ownership of the coarse cell
    // that contains the fine point
    found_ranks = treec->compute_process_collisions(curr_point);

    // if the fine point is not in the domain or if more than one
    // processors share it, mark it as not found
    // (not found points will be searched by all the processors and
    // the processor that owns closest coarse cell to that point will be found,
    // so that even if multiple processes share the cell, we find the one that
    // actually owns it)
    if (found_ranks.empty() || found_ranks.size() > 1)
    {
      // we store fine point coordinates, global row indices and the senders
      // this information will be sent to all the processors
      not_found_points.insert(not_found_points.end(), _x.begin(), _x.end());
      not_found_global_row_indices.insert(not_found_global_row_indices.end(), map_it.second.begin(), map_it.second.end());
      not_found_points_senders.push_back(mpi_rank);
    }
    // if the fine point collides with a coarse cell owned by the current processor,
    // find the coarse cell the fine point lives in
    else if (found_ranks[0] == mpi_rank)
    {
      // find the coarse cell where the fine point lies
      id = treec->compute_first_entity_collision(curr_point);

      // Safety control: if no cell is found on the current processor
      // mark the point as not_found
      if (id == std::numeric_limits<unsigned int>::max())
      {
        not_found_points.insert(not_found_points.end(), _x.begin(), _x.end());
        not_found_global_row_indices.insert(not_found_global_row_indices.end(), map_it.second.begin(), map_it.second.end());
        not_found_points_senders.push_back(mpi_rank);
      }
      else
      {
        // if a cell is found on the current processor, add the point
        // and relative information to the various vectors
        found_ids.push_back(id);
        found_points.insert(found_points.end(), _x.begin(), _x.end());
        global_row_indices.insert(global_row_indices.end(), map_it.second.begin(), map_it.second.end());
        found_points_senders.push_back(mpi_rank);
      }
    }
    // if found elsewhere, store the process where it was found
    else
    {
      found_elsewhere.insert(found_elsewhere.end(),_x.begin(), _x.end());
      which_processor.push_back(found_ranks[0]);
      found_elsewhere_global_row_indices.insert(found_elsewhere_global_row_indices.end(), map_it.second.begin(), map_it.second.end());
    }
  } // end for loop

    // We now need to communicate various information to all the processors:
    // processor column and row ownership
  std::vector<std::vector<std::size_t>> global_n_range_recv(mpi_size, std::vector<std::size_t>(2));
  std::vector<std::size_t> global_m(mpi_size);
  std::vector<std::size_t> global_row_offset(mpi_size);
  MPI::all_gather(mpi_comm, global_n_range, global_n_range_recv);
  MPI::all_gather(mpi_comm, m, global_m);
  MPI::all_gather(mpi_comm, mbegin, global_row_offset);

  // Ok, now we need to handle the points which have been found elsewhere
  // We need to communicate these points to the other workers
  // as well as the relative information
  std::vector<std::vector<double>> found_elsewhere_recv(mpi_size);
  std::vector<std::vector<int>> found_elsewhere_global_row_indices_recv(mpi_size);
  std::vector<std::vector<unsigned int>> which_processor_recv(mpi_size);
  MPI::all_gather(mpi_comm, found_elsewhere, found_elsewhere_recv);
  MPI::all_gather(mpi_comm, found_elsewhere_global_row_indices, found_elsewhere_global_row_indices_recv);
  MPI::all_gather(mpi_comm, which_processor, which_processor_recv);

  // First, handle the points that were found elsewhere
  unsigned int how_many = 0;
  unsigned int receiver = mpi_rank;

  // we loop over the processors that own the fine points that need to be found
  // we call them senders here.
  for (unsigned sender = 0; sender < mpi_size; ++sender)
  {
    // We already searched on the current processor
    if (sender == receiver)
      continue;

    // how many fine points do we need to check?
    how_many = found_elsewhere_recv[sender].size()/dim;
    if (how_many == 0)
      continue;

    // for each fine point, create a Point variable and try to find the
    // coarse cell it lives in. If we cannot, mark the fine point as not found
    // for robustness.
    for (unsigned int i = 0; i < how_many; ++i)
    {
      if (receiver == which_processor_recv[sender][i])
      {
        if (dim == 3)
        {
          _x[0] = found_elsewhere_recv[sender][i*dim];
          _x[1] = found_elsewhere_recv[sender][i*dim + 1];
          _x[2] = found_elsewhere_recv[sender][i*dim + 2];
          curr_point = Point(_x[0], _x[1], _x[2]);
        }
        else if (dim == 2)
        {
          _x[0] = found_elsewhere_recv[sender][i*dim];
          _x[1] = found_elsewhere_recv[sender][i*dim + 1];
          curr_point = Point(_x[0], _x[1]);
        }
        else
        {
          _x[0] = found_elsewhere_recv[sender][i*dim];
          curr_point = Point(_x[0]);
        }

        id = treec->compute_first_entity_collision(curr_point);
        // if the point is not found on the current processor
        // mark it as not found and leave it for later
        if (id == std::numeric_limits<unsigned int>::max())
        {
          not_found_points.insert(not_found_points.end(), _x.begin(), _x.end());
          not_found_global_row_indices.insert(not_found_global_row_indices.end(), &found_elsewhere_global_row_indices_recv[sender][data_size*i], &found_elsewhere_global_row_indices_recv[sender][data_size*i + data_size]);
          not_found_points_senders.push_back(sender);
        }
        else
        {
          // if found, store information
          found_ids.push_back(id);
          found_points.insert(found_points.end(), _x.begin(), _x.end());
          global_row_indices.insert(global_row_indices.end(), &found_elsewhere_global_row_indices_recv[sender][data_size*i],
                                    &found_elsewhere_global_row_indices_recv[sender][data_size*i + data_size]);
          found_points_senders.push_back(sender);
        }
      }
    }
  }

  // communicate the not found list to all the processors
  std::vector<std::vector<double>> not_found_points_recv(mpi_size);
  std::vector<std::vector<int>> not_found_global_row_indices_recv(mpi_size);
  std::vector<std::vector<unsigned int>> not_found_points_senders_recv(mpi_size);
  MPI::all_gather(mpi_comm, not_found_points, not_found_points_recv);
  MPI::all_gather(mpi_comm, not_found_global_row_indices, not_found_global_row_indices_recv);
  MPI::all_gather(mpi_comm, not_found_points_senders, not_found_points_senders_recv);

  // handle not_found points:
  // we need to compute their distances from the closest owned coarse cell
  // and the index/id of that cell.
  std::vector<double> not_found_distances;
  std::vector<unsigned int> not_found_cell_indices;
  // we also need to store the fine point coordinates
  // in case the current processor owns the closest cell
  std::vector<double> found_not_found_points;
  // We need to flatten some vectors for further use
  std::vector<int> not_found_global_row_indices_flattened;
  std::vector<unsigned int> not_found_points_senders_flattened;

  // we loop over all the processors where a fine point was found
  // note that from now on, every processor is doing the same check:
  // compute id and distance of the closest owned coarse cell to the
  // fine point, then send the distances to all the processors, so that
  // each processor can determine which processor owns the closest coarse cell
  for (unsigned int proc = 0; proc < mpi_size; ++proc)
  {
    how_many = not_found_points_recv[proc].size()/dim;

    if (how_many == 0)
      continue;

    // flattening not_found_global_row_indices_recv one step at a time.
    not_found_global_row_indices_flattened.insert(not_found_global_row_indices_flattened.end(),
                                                  not_found_global_row_indices_recv[proc].begin(),
                                                  not_found_global_row_indices_recv[proc].end());
    // updating the std::vector of who owns the fine points
    not_found_points_senders_flattened.insert(not_found_points_senders_flattened.end(),
                                              not_found_points_senders_recv[proc].begin(),
                                              not_found_points_senders_recv[proc].end());

    // reserve memory for speed
    not_found_cell_indices.reserve(not_found_cell_indices.size() + how_many);
    found_not_found_points.reserve(not_found_points.size() + dim*how_many);
    not_found_distances.reserve(not_found_distances.size() + how_many);

    // same trick as before, store the fine point coordinates into a Point
    // variable, then run compute_closest_entity to find the closest owned
    // cell id and distance from the fine point
    for (unsigned int i = 0; i < how_many; ++i)
    {
      if (dim == 3)
      {
        _x[0] = not_found_points_recv[proc][i*dim];
        _x[1] = not_found_points_recv[proc][i*dim + 1];
        _x[2] = not_found_points_recv[proc][i*dim + 2];
        curr_point = Point(_x[0], _x[1], _x[2]);
      }
      else if (dim == 2)
      {
        _x[0] = not_found_points_recv[proc][i*dim];
        _x[1] = not_found_points_recv[proc][i*dim + 1];
        curr_point = Point(_x[0], _x[1]);
      }
      else
      {
        _x[0] = not_found_points_recv[proc][i*dim];
        curr_point = Point(_x[0]);
      }

      std::pair<unsigned int, double> find_point = treec->compute_closest_entity(curr_point);
      not_found_cell_indices.push_back(find_point.first);
      not_found_distances.push_back(find_point.second);
      // store the (now) found, (previously) not found fine point coordinates in a vector
      found_not_found_points.insert(found_not_found_points.end(), _x.begin(), _x.end());
    }
  }

  // communicate all distances to all processor so that each one can tell
  // which processor owns the closest coarse cell to the not found point
  std::vector<std::vector<double>> not_found_distances_recv(mpi_size);
  MPI::all_gather(mpi_comm, not_found_distances, not_found_distances_recv);

  // now need to find which processor has a cell which is closest to the not_found points

  // initialise some variables
  double min_val; // minimum distance
  unsigned min_proc=0; // processor that owns the minimum distance cell
  unsigned int sender; // processor that asked to search for the not found fine point

  how_many = not_found_cell_indices.size();
  for (unsigned i=0; i<how_many; i++)
  {
    // loop over the distances and find the processor who has
    // the point closest to one of its cells
    min_proc = 0;
    min_val = not_found_distances_recv[min_proc][i];
    for (unsigned proc_it = 1; proc_it<mpi_size; proc_it++)
    {
      if (not_found_distances_recv[proc_it][i] < min_val)
      {
        min_val = not_found_distances_recv[proc_it][i];
        min_proc = proc_it;
      }
    }

    // if the current processor is the one which owns the closest cell,
    // add the fine point and closest coarse cell information to the
    // vectors of found points
    if (min_proc == mpi_rank)
    {
      // allocate cell id to current worker if distance is minimum
      id = not_found_cell_indices[i];
      found_ids.push_back(id);
      global_row_indices.insert(global_row_indices.end(), &not_found_global_row_indices_flattened[data_size*i],
                                &not_found_global_row_indices_flattened[data_size*i + data_size]);
      found_points.insert(found_points.end(), found_not_found_points.begin() + dim*i, found_not_found_points.begin() + dim*(i+1));
      sender = not_found_points_senders_flattened[i];
      found_points_senders.push_back(sender);
    }
  }

  // Now every processor should have the information needed to assemble its portion of the matrix.
  // The ids of coarse cell owned by each processor are currently stored in found_ids
  // and their respective global row indices are stored in global_row_indices.
  // The processors that own the matrix rows relative to the fine point are stored in found_points_senders.
  // One last loop and we are ready to go!

  // m_owned is the number of rows the current processor needs to set
  // note that the processor might not own these rows
  std::size_t m_owned = found_ids.size()*data_size;

  // initialise row and column indices and values of the transfer matrix
  // FIXME: replace with boost::multi_array
  std::vector<std::vector<int>> col_indices(m_owned);
  //    std::vector<int> fine_row_indices(m_owned);
  std::vector<std::vector<double>> values(m_owned);
  for(unsigned i = 0; i < m_owned; ++i)
  {
    col_indices[i].resize(eldim);
    values[i].resize(eldim);
  }
  // initialise a single chunk of values (needed for later)
  std::vector<double> temp_values(eldim*data_size);

  // Initialise global sparsity pattern: record on-process and off-process dependencies of fine dofs
  std::vector<std::vector<dolfin::la_index>> send_dnnz(mpi_size);
  std::vector<std::vector<dolfin::la_index>> send_onnz(mpi_size);

  // Get block size of index map (used to calculate dof ownership, below)
  int bs = finemap->index_map()->block_size();

  // loop over the found coarse cells
  for (unsigned int i = 0; i < found_ids.size(); ++i)
  {
    // get coarse cell id
    id = found_ids[i];

    // save fine point coordinates into a Point variable
    if (dim == 3)
    {
      _x[0] = found_points[i*dim];
      _x[1] = found_points[i*dim + 1];
      _x[2] = found_points[i*dim + 2];
      curr_point = Point(_x[0], _x[1], _x[2]);
    }
    else if (dim == 2)
    {
      _x[0] = found_points[i*dim];
      _x[1] = found_points[i*dim + 1];
      curr_point = Point(_x[0], _x[1]);
    }
    else
    {
      _x[0] = found_points[i*dim];
      curr_point = Point(_x[0]);
    }

    // create coarse cell
    curr_cell = Cell(meshc, static_cast<std::size_t>(id));
    // get dofs coordinates of the coarse cell
    curr_cell.get_coordinate_dofs(coordinate_dofs);
    // save cell information into the ufc cell
    curr_cell.get_cell_data(ufc_cell);
    // evaluate the basis functions of the coarse cells
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
      // Loop over the coarse dofs and stuff their contributions
      for (unsigned j = 0; j < eldim; j++)
      {
        unsigned fine_row = i*data_size + k;
        PetscInt global_fine_dof = global_row_indices[fine_row];
        PetscInt coarse_dof = coarse_local_to_global_dofs[temp_dofs[j]];

        // Set the column
        col_indices[fine_row][j] = coarse_dof;
        // Set the value
        values[fine_row][j] = temp_values[data_size*j + k];
        // Record which global row we're actually talking about
        //          fine_row_indices[fine_row] = global_fine_dof;

        // Once we have the global column indices, determine the sparsity pattern.
        // Which columns are owned by the process that owns the fine point?

        // get the fine point owner processor
        sender = found_points_senders[i];
        // get its column ownership range
        PetscInt n_own_begin = global_n_range_recv[sender][0];
        PetscInt n_own_end = global_n_range_recv[sender][1];
        // check and allocate sparsity pattern
        int p = finemap->index_map()->global_index_owner(global_fine_dof/bs);
        if ((n_own_begin <= coarse_dof) && (coarse_dof < n_own_end))
          send_dnnz[p].push_back(global_fine_dof);
        else
          send_onnz[p].push_back(global_fine_dof);

      } // end loop over all coarse dofs in the cell
    } // end loop over fine dofs associated with this collision
  } // end loop over found points

  std::vector<std::vector<dolfin::la_index>> recv_onnz;
  MPI::all_to_all(mpi_comm, send_onnz, recv_onnz);
  std::vector<int> onnz(m, 0);
  for (const auto &p : recv_onnz)
    for (const auto &q : p)
    {
      dolfin_assert(q >= (dolfin::la_index)mbegin and q < (dolfin::la_index)mend);
      ++onnz[q - mbegin];
    }

  std::vector<std::vector<dolfin::la_index>> recv_dnnz;
  MPI::all_to_all(mpi_comm, send_dnnz, recv_dnnz);
  std::vector<int> dnnz(m, 0);
  for (const auto &p : recv_dnnz)
    for (const auto &q : p)
    {
      dolfin_assert(q >= (dolfin::la_index)mbegin and q < (dolfin::la_index)mend);
      ++dnnz[q - mbegin];
    }

  if (mpi_size > 1)
  {
    ierr = MatMPIAIJSetPreallocation(I, PETSC_DEFAULT, dnnz.data(), PETSC_DEFAULT, onnz.data());
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
  }
  else
  {
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
 : _spaces(function_spaces)
{
  for (auto &V : _spaces)
  {
    dolfin_assert(V);

    DM dm;
    dolfin_assert(V->mesh());
    DMShellCreate(V->mesh()->mpi_comm(), &dm);
    DMShellSetContext(dm, (void*)&V);

    DMShellSetCreateGlobalVector(dm, PETScDMCollection::create_global_vector);
    DMShellSetCreateInterpolation(dm, PETScDMCollection::create_interpolation);

    _dms.push_back(dm);
  }

  for (std::size_t i = 0; i < function_spaces.size() - 1; i++)
  {
    //DM dmc = function_spaces[i];
    //DM dmf = function_spaces[i+1];
    //DMSetFineDM(dmc, dmf)
    //DMShellSetRefine(dmc, PETScDMCollection::refine);
    DMSetFineDM(_dms[i], _dms[i + 1]);
    DMShellSetRefine(_dms[i], PETScDMCollection::refine);
  }

  for (std::size_t i = 1; i < function_spaces.size(); i++)
  {
    //DM dmc = function_spaces[i-1];
    //DM dmf = function_spaces[i];
    //DMSetCoarseDM(dmf, dmc);
    //DMShellSetRefine(dmf, PETScDMCollection::coarsen);

    DMSetCoarseDM(_dms[i],_dms[i - 1]);
    DMShellSetCoarsen(_dms[i], PETScDMCollection::coarsen);

  }
}
//-----------------------------------------------------------------------------
PETScDMCollection::~PETScDMCollection()
{
  for (auto dm : _dms)
    DMDestroy(&dm);
}
//-----------------------------------------------------------------------------
PetscErrorCode PETScDMCollection::create_global_vector(DM dm, Vec* vec)
{
  std::shared_ptr<FunctionSpace> *V;
  DMShellGetContext(dm, (void**)&V);

  Function u(*V);
  *vec = u.vector()->down_cast<PETScVector>().vec();
  PetscObjectReference((PetscObject)*vec);
  return 0;
}
//-----------------------------------------------------------------------------
PetscErrorCode PETScDMCollection::create_interpolation(DM dmc, DM dmf, Mat *mat, Vec *vec)
{
  std::shared_ptr<FunctionSpace> *Vc, *Vf;
  DMShellGetContext(dmc, (void**)&Vc);
  DMShellGetContext(dmf, (void**)&Vf);

  std::shared_ptr<PETScMatrix> P = create_transfer_matrix(*Vc, *Vf);

  *mat = P->mat();
  *vec = NULL;

  PetscObjectReference((PetscObject)*mat);

  return 0;
}
//-----------------------------------------------------------------------------
PetscErrorCode PETScDMCollection::coarsen(DM dmf, MPI_Comm comm, DM* dmc)
{
  DMGetCoarseDM(dmf, dmc);
  return 0;
}
//-----------------------------------------------------------------------------
PetscErrorCode PETScDMCollection::refine(DM dmc, MPI_Comm comm, DM* dmf)
{
  DMGetFineDM(dmc, dmf);
  return 0;
}
//-----------------------------------------------------------------------------
#endif
