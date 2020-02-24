// Copyright (C) 2016 Patrick E. Farrell and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "PETScDMCollection.h"
#include <Eigen/Dense>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/function/Function.h>
#include <dolfinx/function/FunctionSpace.h>
#include <dolfinx/geometry/BoundingBoxTree.h>
#include <dolfinx/geometry/utils.h>
#include <dolfinx/la/PETScMatrix.h>
#include <dolfinx/mesh/CoordinateDofs.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/MeshEntity.h>
#include <dolfinx/mesh/MeshIterator.h>
#include <petscdmshell.h>
#include <petscmat.h>

using namespace dolfinx;
using namespace dolfinx::fem;

namespace
{
// Coordinate comparison operator
struct lt_coordinate
{
  lt_coordinate(double tolerance) : TOL(tolerance) {}

  bool operator()(const std::vector<double>& x,
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

std::map<std::vector<double>, std::vector<std::int64_t>, lt_coordinate>
tabulate_coordinates_to_dofs(const function::FunctionSpace& V)
{
  std::map<std::vector<double>, std::vector<std::int64_t>, lt_coordinate>
      coords_to_dofs(lt_coordinate(1.0e-12));

  // Extract mesh, dofmap and element
  assert(V.dofmap());
  assert(V.element());
  assert(V.mesh());
  const fem::DofMap& dofmap = *V.dofmap();
  const fem::FiniteElement& element = *V.element();
  const mesh::Mesh& mesh = *V.mesh();
  Eigen::Array<std::int64_t, Eigen::Dynamic, 1> local_to_global
      = dofmap.index_map->indices(true);

  // Geometric dimension
  const int gdim = mesh.geometry().dim();
  const int tdim = mesh.topology().dim();

  // Get dof coordinates on reference element
  const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& X
      = element.dof_reference_coordinates();

  // Get coordinate mapping
  if (!mesh.geometry().coord_mapping)
  {
    throw std::runtime_error(
        "CoordinateElement has not been attached to mesh.");
  }
  const CoordinateElement& cmap = *mesh.geometry().coord_mapping;

  // Prepare cell geometry
  const graph::AdjacencyList<std::int32_t>& connectivity_g
      = mesh.coordinate_dofs().entity_points();
  const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& pos_g
      = connectivity_g.offsets();
  const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& cell_g
      = connectivity_g.array();
  // FIXME: Add proper interface for num coordinate dofs
  const int num_dofs_g = connectivity_g.num_links(0);
  const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& x_g
      = mesh.geometry().points();

  // Loop over cells and tabulate dofs
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coordinates(element.space_dimension(), gdim);
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coordinate_dofs(num_dofs_g, gdim);
  std::vector<double> coors(gdim);

  // Speed up the computations by only visiting (most) dofs once
  const std::int64_t local_size
      = dofmap.index_map->size_local() * dofmap.index_map->block_size();
  std::vector<bool> already_visited(local_size, false);

  for (auto& cell : mesh::MeshRange(mesh, tdim))
  {
    // Get cell coordinates
    const int cell_index = cell.index();
    for (int i = 0; i < num_dofs_g; ++i)
      for (int j = 0; j < gdim; ++j)
        coordinate_dofs(i, j) = x_g(cell_g[pos_g[cell_index] + i], j);

    // Get local-to-global map
    auto dofs = dofmap.cell_dofs(cell.index());

    // Tabulate dof coordinates on cell
    cmap.push_forward(coordinates, X, coordinate_dofs);

    // Map dofs into coords_to_dofs
    for (Eigen::Index i = 0; i < dofs.size(); ++i)
    {
      const std::int64_t dof = dofs[i];
      if (dof < local_size)
      {
        // Skip already checked dofs
        if (already_visited[dof])
          continue;

        // Put coordinates in coors
        std::copy(coordinates.row(i).data(),
                  coordinates.row(i).data() + coordinates.row(i).size(),
                  coors.begin());

        // Add dof to list at this coord
        const auto ins = coords_to_dofs.insert({coors, {local_to_global[dof]}});
        if (!ins.second)
          ins.first->second.push_back(local_to_global[dof]);

        already_visited[dof] = true;
      }
    }
  }

  return coords_to_dofs;
}
} // namespace

//-----------------------------------------------------------------------------
PETScDMCollection::PETScDMCollection(
    std::vector<std::shared_ptr<const function::FunctionSpace>> function_spaces)
    : _spaces(function_spaces), _dms(function_spaces.size(), nullptr)
{
  for (std::size_t i = 0; i < _spaces.size(); ++i)
  {
    assert(_spaces[i]);
    assert(_spaces[i].get());

    // Get MPI communicator from mesh::Mesh
    assert(_spaces[i]->mesh());
    MPI_Comm comm = _spaces[i]->mesh()->mpi_comm();

    // Create DM
    DMShellCreate(comm, &_dms[i]);
    DMShellSetContext(_dms[i], (void*)_spaces[i].get());

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
  const int base = i < 0 ? _dms.size() : 0;
  return _dms.at(base + i);
}
//-----------------------------------------------------------------------------
void PETScDMCollection::check_ref_count() const
{
  for (std::size_t i = 0; i < _dms.size(); ++i)
  {
    PetscInt cnt = 0;
    PetscObjectGetReference((PetscObject)_dms[i], &cnt);
  }
}
//-----------------------------------------------------------------------------
void PETScDMCollection::reset(int i)
{
  PetscObjectDereference((PetscObject)_dms[i]);
  // PetscObjectDereference((PetscObject)_dms.back());
  // for (std::size_t i = 0; i < _dms.size(); ++i)
  //  PetscObjectDereference((PetscObject)_dms[i]);
}
//-----------------------------------------------------------------------------
la::PETScMatrix PETScDMCollection::create_transfer_matrix(
    const function::FunctionSpace& coarse_space,
    const function::FunctionSpace& fine_space)
{
  // FIXME: refactor and split up

  // Get coarse mesh and dimension of the domain
  assert(coarse_space.mesh());
  const mesh::Mesh& meshc = *coarse_space.mesh();
  const int gdim = meshc.geometry().dim();
  const int tdim = meshc.topology().dim();

  // MPI communicator, size and rank
  const MPI_Comm mpi_comm = meshc.mpi_comm();
  const int mpi_size = MPI::size(mpi_comm);

  // Initialise bounding box tree and dofmaps
  geometry::BoundingBoxTree treec(meshc, meshc.topology().dim());
  std::shared_ptr<const fem::DofMap> coarsemap = coarse_space.dofmap();
  std::shared_ptr<const fem::DofMap> finemap = fine_space.dofmap();

  // Create map from coordinates to dofs sharing that coordinate
  std::map<std::vector<double>, std::vector<std::int64_t>, lt_coordinate>
      coords_to_dofs = tabulate_coordinates_to_dofs(fine_space);

  // Global dimensions of the dofs and of the transfer matrix (M-by-N,
  // where M is the fine space dimension, N is the coarse space
  // dimension)
  std::size_t M = fine_space.dim();
  std::size_t N = coarse_space.dim();

  // Local dimension of the dofs and of the transfer matrix
  std::array<std::int64_t, 2> m = finemap->index_map->local_range();
  std::array<std::int64_t, 2> n = coarsemap->index_map->local_range();
  m[0] *= finemap->index_map->block_size();
  m[1] *= finemap->index_map->block_size();
  n[0] *= coarsemap->index_map->block_size();
  n[1] *= coarsemap->index_map->block_size();

  // Get finite element for the coarse space. This will be needed to
  // evaluate the basis functions for each cell.
  std::shared_ptr<const fem::FiniteElement> el = coarse_space.element();

  // Check that it is the same kind of element on each space.
  {
    std::shared_ptr<const fem::FiniteElement> elf = fine_space.element();
    assert(elf);
    // Check that function ranks match
    if (el->value_rank() != elf->value_rank())
    {
      throw std::runtime_error("Ranks of function spaces do not match:"
                               + std::to_string(el->value_rank()) + ", "
                               + std::to_string(elf->value_rank()));
    }

    // Check that function dims match
    for (int i = 0; i < el->value_rank(); ++i)
    {
      if (el->value_dimension(i) != elf->value_dimension(i))
      {
        throw std::runtime_error("Dimensions of function spaces ("
                                 + std::to_string(i) + ") do not match:"
                                 + std::to_string(el->value_dimension(i)) + ", "
                                 + std::to_string(elf->value_dimension(i)));
      }
    }
  }

  // Number of dofs per cell for the finite element.
  std::size_t eldim = el->space_dimension();

  // Number of dofs associated with each fine point
  int data_size = 1;
  for (int data_dim = 0; data_dim < el->value_rank(); data_dim++)
    data_size *= el->value_dimension(data_dim);

  // The overall idea is: a fine point can be on a coarse cell in the
  // current processor, on a coarse cell in a different processor, or
  // outside the coarse domain.  If the point is found on the
  // processor, evaluate basis functions, if found elsewhere, use the
  // other processor to evaluate basis functions, if not found at all,
  // or if found in multiple processors, use compute_closest_entity on
  // all processors and find which coarse cell is the closest entity
  // to the fine point amongst all processors.

  // found_ids[i] contains the coarse cell id for each fine point
  std::vector<std::size_t> found_ids;
  found_ids.reserve((std::size_t)M / mpi_size);

  // found_points[dim*i:dim*(i + 1)] contain the coordinates of the
  // fine point i
  std::vector<double> found_points;
  found_points.reserve((std::size_t)gdim * M / mpi_size);

  // global_row_indices[data_size*i:data_size*(i + 1)] are the rows associated
  // with
  // this point
  std::vector<int> global_row_indices;
  global_row_indices.reserve((std::size_t)data_size * M / mpi_size);

  // Collect up any points which lie outside the domain
  std::vector<double> exterior_points;
  std::vector<int> exterior_global_indices;

  // 1. Allocate all points on this process to "Bounding Boxes" based
  // on the global BoundingBoxTree, and send them to those
  // processes. Any points which fall outside the global BBTree are
  // collected up separately.

  std::vector<std::vector<double>> send_found(mpi_size);
  std::vector<std::vector<int>> send_found_global_row_indices(mpi_size);

  std::vector<int> proc_list;
  std::vector<int> found_ranks;
  // Iterate through fine points on this process
  for (const auto& map_it : coords_to_dofs)
  {
    const std::vector<double>& _x = map_it.first;
    Eigen::Map<const Eigen::Vector3d> curr_point(_x.data());

    // Compute which processes' BBoxes contain the fine point
    found_ranks = geometry::compute_process_collisions(treec, curr_point);

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
      // Send points to candidate processes, also recording the
      // processes they are sent to in proc_list
      proc_list.push_back(found_ranks.size());
      for (auto& rp : found_ranks)
      {
        proc_list.push_back(rp);
        send_found[rp].insert(send_found[rp].end(), _x.begin(), _x.end());
        // Also save the indices, but don't send yet.
        send_found_global_row_indices[rp].insert(
            send_found_global_row_indices[rp].end(), map_it.second.begin(),
            map_it.second.end());
      }
    }
  }
  std::vector<std::vector<double>> recv_found(mpi_size);
  MPI::all_to_all(mpi_comm, send_found, recv_found);

  // 2. On remote process, find the Cell which the point lies inside,
  // if any.  Send back the result to the originating process. In the
  // case that the point is found inside cells on more than one
  // process, the originating process will arbitrate.
  std::vector<std::vector<int>> send_ids(mpi_size);
  for (int p = 0; p < mpi_size; ++p)
  {
    int n_points = recv_found[p].size() / gdim;
    for (int i = 0; i < n_points; ++i)
    {
      Eigen::Map<const Eigen::Vector3d> curr_point(&recv_found[p][i * gdim]);
      send_ids[p].push_back(
          geometry::compute_first_entity_collision(treec, curr_point, meshc));
    }
  }
  std::vector<std::vector<int>> recv_ids(mpi_size);
  MPI::all_to_all(mpi_comm, send_ids, recv_ids);

  // 3. Revisit original list of sent points in the same order as
  // before. Now we also have the remote cell-id, if any.
  std::vector<int> count(mpi_size, 0);
  for (auto p = proc_list.begin(); p != proc_list.end(); p += (*p + 1))
  {
    int nprocs = *p;
    int owner = -1;
    // Find first process which owns a cell containing the point
    for (int j = 1; j < (nprocs + 1); ++j)
    {
      const int proc = *(p + j);
      const int id = recv_ids[proc][count[proc]];
      if (id >= 0)
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
                             &send_found[proc][count[proc] * gdim],
                             &send_found[proc][(count[proc] + 1) * gdim]);
      exterior_global_indices.insert(
          exterior_global_indices.end(),
          &send_found_global_row_indices[proc][count[proc] * data_size],
          &send_found_global_row_indices[proc][(count[proc] + 1) * data_size]);
    }
    else if (nprocs > 1)
    {
      // If point is found on multiple processes, send -1 as the index
      // to the remote processes which are not the "owner"
      for (int j = 1; j < (nprocs + 1); ++j)
      {
        const int proc = *(p + j);
        if (proc != owner)
        {
          for (int k = 0; k < data_size; ++k)
            send_found_global_row_indices[proc][count[proc] * data_size + k]
                = -1;
        }
      }
    }

    // Move to next point
    for (int j = 1; j < (nprocs + 1); ++j)
      ++count[*(p + j)];
  }

  // Finally, send indices
  std::vector<std::vector<int>> recv_found_global_row_indices(mpi_size);
  MPI::all_to_all(mpi_comm, send_found_global_row_indices,
                  recv_found_global_row_indices);

  // Flatten results ready for insertion
  for (int p = 0; p < mpi_size; ++p)
  {
    const std::vector<int>& id_p = send_ids[p];
    const int npoints = id_p.size();
    assert(npoints == (int)recv_found[p].size() / gdim);
    assert(npoints == (int)recv_found_global_row_indices[p].size() / data_size);

    Eigen::Map<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                  Eigen::RowMajor>>
        point_p(recv_found[p].data(), npoints, gdim);

    Eigen::Map<const Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic,
                                  Eigen::RowMajor>>
        global_idx_p(recv_found_global_row_indices[p].data(), npoints,
                     data_size);
    for (int i = 0; i < npoints; ++i)
    {
      if (id_p[i] >= 0 and global_idx_p(i, 0) != -1)
      {
        found_ids.push_back(id_p[i]);
        global_row_indices.insert(
            global_row_indices.end(), global_idx_p.row(i).data(),
            global_idx_p.row(i).data() + global_idx_p.cols());

        found_points.insert(found_points.end(), point_p.row(i).data(),
                            point_p.row(i).data() + point_p.cols());
      }
    }
  }

  // Find closest cells for points that lie outside the domain and add
  // them to the lists
  find_exterior_points(mpi_comm, meshc, treec, gdim, data_size, exterior_points,
                       exterior_global_indices, global_row_indices, found_ids,
                       found_points);

  // Now every processor should have the information needed to
  // assemble its portion of the matrix.  The ids of coarse cell owned
  // by each processor are currently stored in found_ids and their
  // respective global row indices are stored in global_row_indices.
  // One last loop and we are ready to go!

  // m_owned is the number of rows the current processor needs to set
  // note that the processor might not own these rows
  const int m_owned = global_row_indices.size();

  // Initialise row and column indices and values of the transfer
  // matrix
  Eigen::Array<PetscInt, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      col_indices(m_owned, eldim);
  Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      values(m_owned, eldim);
  Eigen::Tensor<double, 3, Eigen::RowMajor> temp_values(1, eldim, data_size);

  // Initialise global sparsity pattern: record on-process and
  // off-process dependencies of fine dofs
  std::vector<std::vector<PetscInt>> send_dnnz(mpi_size);
  std::vector<std::vector<PetscInt>> send_onnz(mpi_size);

  // Initialise local to global dof maps (needed to allocate the
  // entries of the transfer matrix with the correct global indices)
  Eigen::Array<std::int64_t, Eigen::Dynamic, 1> coarse_local_to_global_dofs
      = coarsemap->index_map->indices(true);

  // Loop over the found coarse cells
  Eigen::Map<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                Eigen::RowMajor>>
      x(found_points.data(), found_ids.size(), gdim);
  const auto cmap = meshc.geometry().coord_mapping;
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> X(1,
                                                                          gdim);
  Eigen::Tensor<double, 3, Eigen::RowMajor> J(1, gdim, tdim);
  Eigen::Array<double, 1, 1> detJ(1);
  Eigen::Tensor<double, 3, Eigen::RowMajor> K(1, tdim, gdim);

  // Prepare cell geometry
  const graph::AdjacencyList<std::int32_t>& connectivity_g
      = meshc.coordinate_dofs().entity_points();
  const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& pos_g
      = connectivity_g.offsets();
  const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& cell_g
      = connectivity_g.array();
  // FIXME: Add proper interface for num coordinate dofs
  const int num_dofs_g = connectivity_g.num_links(0);
  const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& x_g
      = meshc.geometry().points();
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coordinate_dofs(num_dofs_g, gdim);
  ; // cell dofs coordinates vector

  for (std::size_t i = 0; i < found_ids.size(); ++i)
  {
    // Get coarse cell id and point
    std::size_t id = found_ids[i];

    // Create coarse cell
    mesh::MeshEntity coarse_cell(meshc, tdim, static_cast<std::size_t>(id));

    // Get dofs coordinates of the coarse cell
    const int cell_index = coarse_cell.index();
    for (int j = 0; j < num_dofs_g; ++j)
      for (int k = 0; k < gdim; ++k)
        coordinate_dofs(j, k) = x_g(cell_g[pos_g[cell_index] + j], k);

    // Evaluate the basis functions of the coarse cells at the fine
    // point and store the values into temp_values

    cmap->compute_reference_geometry(X, J, detJ, K, x.row(i), coordinate_dofs);
    el->evaluate_reference_basis(temp_values, X);

    // Get the coarse dofs associated with this cell
    auto temp_dofs = coarsemap->cell_dofs(id);

    // Loop over the fine dofs associated with this collision
    for (int k = 0; k < data_size; k++)
    {
      const int fine_row = i * data_size + k;
      const std::size_t global_fine_dof = global_row_indices[fine_row];
      int p = finemap->index_map->owner(global_fine_dof / data_size);

      // Loop over the coarse dofs and stuff their contributions
      for (unsigned j = 0; j < eldim; j++)
      {
        const std::size_t coarse_dof
            = coarse_local_to_global_dofs[temp_dofs[j]];

        // Set the column
        col_indices(fine_row, j) = coarse_dof;
        // Set the value
        values(fine_row, j) = temp_values(0, j, k);

        int pc = coarsemap->index_map->owner(coarse_dof / data_size);
        if (p == pc)
          send_dnnz[p].push_back(global_fine_dof);
        else
          send_onnz[p].push_back(global_fine_dof);
      }
    }
  }

  // Communicate off-process columns nnz, and flatten to get nnz per
  // row we also keep track of the ownership range
  std::size_t mbegin = m[0];
  std::size_t mend = m[1];
  std::vector<PetscInt> recv_onnz;
  MPI::all_to_all(mpi_comm, send_onnz, recv_onnz);

  std::vector<PetscInt> onnz(m[1] - m[0], 0);
  for (const auto& q : recv_onnz)
  {
    assert(q >= (PetscInt)mbegin and q < (PetscInt)mend);
    ++onnz[q - mbegin];
  }

  // Communicate on-process columns nnz, and flatten to get nnz per
  // row
  std::vector<PetscInt> recv_dnnz;
  MPI::all_to_all(mpi_comm, send_dnnz, recv_dnnz);
  std::vector<PetscInt> dnnz(m[1] - m[0], 0);
  for (const auto& q : recv_dnnz)
  {
    assert(q >= (PetscInt)mbegin and q < (PetscInt)mend);
    ++dnnz[q - mbegin];
  }

  // Initialise PETSc Mat and error code
  PetscErrorCode ierr;
  Mat I;

  // Create and initialise the transfer matrix as MATMPIAIJ/MATSEQAIJ
  ierr = MatCreate(mpi_comm, &I);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);

  if (mpi_size > 1)
  {
    ierr = MatSetType(I, MATMPIAIJ);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = MatSetSizes(I, m[1] - m[0], n[1] - n[0], M, N);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = MatMPIAIJSetPreallocation(I, PETSC_DEFAULT, dnnz.data(),
                                     PETSC_DEFAULT, onnz.data());
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
  }
  else
  {
    ierr = MatSetType(I, MATSEQAIJ);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = MatSetSizes(I, m[1] - m[0], n[1] - n[0], M, N);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = MatSeqAIJSetPreallocation(I, PETSC_DEFAULT, dnnz.data());
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
  }

  // Setting transfer matrix values row by row
  for (int fine_row = 0; fine_row < m_owned; ++fine_row)
  {
    PetscInt fine_dof = global_row_indices[fine_row];
    ierr
        = MatSetValues(I, 1, &fine_dof, eldim, col_indices.row(fine_row).data(),
                       values.row(fine_row).data(), INSERT_VALUES);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
  }

  // Assemble the transfer matrix
  ierr = MatAssemblyBegin(I, MAT_FINAL_ASSEMBLY);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
  ierr = MatAssemblyEnd(I, MAT_FINAL_ASSEMBLY);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);

  return la::PETScMatrix(I, false);
}
//-----------------------------------------------------------------------------
void PETScDMCollection::find_exterior_points(
    MPI_Comm mpi_comm, const mesh::Mesh& meshc,
    const geometry::BoundingBoxTree& treec, int dim, int data_size,
    const std::vector<double>& send_points,
    const std::vector<int>& send_indices, std::vector<int>& indices,
    std::vector<std::size_t>& cell_ids, std::vector<double>& points)
{
  assert(send_indices.size() / data_size == send_points.size() / dim);
  Eigen::Map<
      const Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      send_indices_arr(send_indices.data(), send_indices.size() / data_size,
                       data_size);
  int mpi_rank = MPI::rank(mpi_comm);
  int mpi_size = MPI::size(mpi_comm);

  // Get all points on all processes
  std::vector<std::vector<double>> recv_points(mpi_size);
  MPI::all_gather(mpi_comm, send_points, recv_points);

  int num_recv_points = 0;
  for (auto& p : recv_points)
    num_recv_points += p.size();
  num_recv_points /= dim;

  // Save distances and ids of nearest cells on this process
  std::vector<double> send_distance;
  std::vector<int> ids;

  // FIXME: move outside of function
  geometry::BoundingBoxTree treec_midpoint
      = geometry::create_midpoint_tree(meshc);

  send_distance.reserve(num_recv_points);
  ids.reserve(num_recv_points);
  for (const auto& p : recv_points)
  {
    int n_points = p.size() / dim;
    for (int i = 0; i < n_points; ++i)
    {
      Eigen::Map<const Eigen::Vector3d> curr_point(&p[i * dim]);
      std::pair<int, double> find_point = geometry::compute_closest_entity(
          treec, treec_midpoint, curr_point, meshc);
      send_distance.push_back(find_point.second);
      ids.push_back(find_point.first);
    }
  }

  // All processes get the same distance information
  std::vector<double> recv_distance(num_recv_points * mpi_size);
  MPI::all_gather(mpi_comm, send_distance, recv_distance);

  // Determine which process has closest cell for each point, and send
  // the global indices to that process
  int ct = 0;
  std::vector<std::vector<int>> send_global_indices(mpi_size);

  for (int p = 0; p < mpi_size; ++p)
  {
    int n_points = recv_points[p].size() / dim;
    Eigen::Map<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                  Eigen::RowMajor>>
        point_arr(recv_points[p].data(), n_points, dim);
    for (int i = 0; i < n_points; ++i)
    {
      int min_proc = 0;
      double min_val = recv_distance[ct];
      for (int q = 1; q < mpi_size; ++q)
      {
        const double val = recv_distance[q * num_recv_points + ct];
        if (val < min_val)
        {
          min_val = val;
          min_proc = q;
        }
      }

      if (min_proc == mpi_rank)
      {
        // If this process has closest cell, save the information
        points.insert(points.end(), point_arr.row(i).data(),
                      point_arr.row(i).data() + point_arr.cols());
        cell_ids.push_back(ids[ct]);
      }
      if (p == mpi_rank)
      {
        send_global_indices[min_proc].insert(
            send_global_indices[min_proc].end(), send_indices_arr.row(i).data(),
            send_indices_arr.row(i).data() + send_indices_arr.cols());
      }
      ++ct;
    }
  }

  // Send out global indices for the points provided by this process
  std::vector<int> recv_global_indices;
  MPI::all_to_all(mpi_comm, send_global_indices, recv_global_indices);

  indices.insert(indices.end(), recv_global_indices.begin(),
                 recv_global_indices.end());
}
//-----------------------------------------------------------------------------
PetscErrorCode PETScDMCollection::create_global_vector(DM dm, Vec* vec)
{
  // Get DOLFINX FunctionSpace from the PETSc DM object
  std::shared_ptr<function::FunctionSpace>* V;
  DMShellGetContext(dm, (void**)&V);

  // Create Vector
  function::Function u(*V);
  *vec = u.vector().vec();

  // FIXME: Does increasing the reference count lead to a memory leak?
  // Increment PETSc reference counter the Vec
  PetscObjectReference((PetscObject)*vec);

  return 0;
}
//-----------------------------------------------------------------------------
PetscErrorCode PETScDMCollection::create_interpolation(DM dmc, DM dmf, Mat* mat,
                                                       Vec* vec)
{
  // Get DOLFINX function::FunctionSpaces from PETSc DM objects (V0 is
  // coarse space, V1 is fine space)
  function::FunctionSpace *V0(nullptr), *V1(nullptr);
  DMShellGetContext(dmc, (void**)&V0);
  DMShellGetContext(dmf, (void**)&V1);

  // Build interpolation matrix (V0 to V1)
  assert(V0);
  assert(V1);
  auto P = std::make_shared<la::PETScMatrix>(create_transfer_matrix(*V0, *V1));

  // Copy PETSc matrix pointer and increase reference count
  *mat = P->mat();
  PetscObjectReference((PetscObject)*mat);

  // Set optional vector to NULL
  *vec = nullptr;

  return 0;
}
//-----------------------------------------------------------------------------
PetscErrorCode PETScDMCollection::coarsen(DM dmf, MPI_Comm, DM* dmc)
{
  // Get the coarse DM from the fine DM
  return DMGetCoarseDM(dmf, dmc);
}
//-----------------------------------------------------------------------------
PetscErrorCode PETScDMCollection::refine(DM dmc, MPI_Comm, DM* dmf)
{
  // Get the fine DM from the coarse DM
  return DMGetFineDM(dmc, dmf);
}
//-----------------------------------------------------------------------------
