// Copyright (C) 2013-2018 Johan Hake, Jan Blechta and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "utils.h"
#include <dolfin/common/IndexMap.h>
#include <dolfin/common/Timer.h>
#include <dolfin/common/types.h>
#include <dolfin/fem/Form.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/fem/SparsityPatternBuilder.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/la/PETScMatrix.h>
#include <dolfin/la/PETScVector.h>
#include <dolfin/la/SparsityPattern.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshIterator.h>
#include <dolfin/mesh/Vertex.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
la::PETScVector dolfin::fem::init_vector(const Form& L)
{
  if (L.rank() != 1)
    throw std::runtime_error("Cannot initialise vector. Form must be linear.");

  auto dofmap = L.function_space(0)->dofmap();
  auto index_map = dofmap->index_map();
  assert(index_map);

  return la::PETScVector(*index_map);
}
//-----------------------------------------------------------------------------
la::PETScMatrix
fem::init_nest_matrix(std::vector<std::vector<const fem::Form*>> a)
{
  // FIXME: check that a is square

  // Block shape
  const auto shape = boost::extents[a.size()][a[0].size()];

  // Loop over each form and create matrix
  boost::multi_array<std::shared_ptr<la::PETScMatrix>, 2> mats(shape);
  boost::multi_array<Mat, 2> petsc_mats(shape);
  for (std::size_t i = 0; i < a.size(); ++i)
  {
    for (std::size_t j = 0; j < a[i].size(); ++j)
    {
      if (a[i][j])
      {
        mats[i][j] = std::make_shared<la::PETScMatrix>(init_matrix(*a[i][j]) );
        // std::cout << "  init mat" << std::endl;
        // init(*mats[i][j], *a[i][j]);
        // std::cout << "  post init mat" << std::endl;
        petsc_mats[i][j] = mats[i][j]->mat();
      }
      else
        petsc_mats[i][j] = nullptr;
    }
  }

  // Initialise block (MatNest) matrix
  Mat _A;
  MatCreate(a[0][0]->mesh()->mpi_comm(), &_A);
  MatSetType(_A, MATNEST);
  MatNestSetSubMats(_A, petsc_mats.shape()[0], NULL, petsc_mats.shape()[1],
                    NULL, petsc_mats.data());
  MatSetUp(_A);
  return la::PETScMatrix(_A);
}
//-----------------------------------------------------------------------------
la::PETScVector fem::init_nest(std::vector<const fem::Form*> L)
{
  // Loop over each form and create vector
  std::vector<std::shared_ptr<la::PETScVector>> vecs(L.size());
  std::vector<Vec> petsc_vecs(L.size());
  for (std::size_t i = 0; i < L.size(); ++i)
  {
    if (L[i])
    {
      vecs[i] = std::make_shared<la::PETScVector>(init_vector(*L[i]));
      petsc_vecs[i] = vecs[i]->vec();
    }
    else
      petsc_vecs[i] = nullptr;
  }

  // Create nested (VecNest) vector
  Vec y;
  VecCreateNest(vecs[0]->mpi_comm(), petsc_vecs.size(), NULL, petsc_vecs.data(),
                &y);
  return la::PETScVector(y);
}
//-----------------------------------------------------------------------------
la::PETScMatrix fem::init_monolithic_matrix(
                          std::vector<std::vector<const fem::Form*>> a)
{
  // FIXME: handle null blocks

  // std::cout << "Initialising block matrix" << std::endl;

  // Block shape
  // const auto shape = boost::extents[a.size()][a[0].size()];

  std::vector<std::vector<std::unique_ptr<la::SparsityPattern>>> patterns(
      a.size());
  std::vector<std::vector<const la::SparsityPattern*>> p(
      a.size(), std::vector<const la::SparsityPattern*>(a[0].size()));
  for (std::size_t row = 0; row < a.size(); ++row)
  {
    for (std::size_t col = 0; col < a[row].size(); ++col)
    {
      // std::cout << "  Initialising block: " << row << ", " << col <<
      // std::endl;
      auto map0 = a[row][col]->function_space(0)->dofmap()->index_map();
      auto map1 = a[row][col]->function_space(1)->dofmap()->index_map();

      // std::cout << "  Push Initialising block: " << std::endl;
      std::array<std::shared_ptr<const common::IndexMap>, 2> maps
          = {{map0, map1}};

      // Build sparsity pattern
      std::array<const GenericDofMap*, 2> dofmaps
          = {{a[row][col]->function_space(0)->dofmap().get(),
              a[row][col]->function_space(1)->dofmap().get()}};
      const mesh::Mesh& mesh = *a[row][col]->mesh();
      auto sp = std::make_unique<la::SparsityPattern>(
          SparsityPatternBuilder::build(mesh.mpi_comm(), mesh, dofmaps, true,
                                        false, false, false, false));
      patterns[row].push_back(std::move(sp));
      p[row][col] = patterns[row][col].get();
      assert(p[row][col]);
    }
  }

  // Create merged sparsity pattern
  // std::cout << "  Build merged sparsity pattern" << std::endl;
  la::SparsityPattern pattern(a[0][0]->mesh()->mpi_comm(), p);

  // Initialise matrix
  // std::cout << "  Init parent matrix" << std::endl;
  la::PETScMatrix A(a[0][0]->mesh()->mpi_comm(), pattern);
  // std::cout << "  Post init parent matrix" << std::endl;

  // Build list of row and column index maps (over each block)
  std::array<std::vector<const common::IndexMap*>, 2> index_maps;
  for (std::size_t i = 0; i < a.size(); ++i)
  {
    auto map = a[i][0]->function_space(0)->dofmap()->index_map();
    index_maps[0].push_back(map.get());
  }
  for (std::size_t i = 0; i < a[0].size(); ++i)
  {
    auto map = a[0][i]->function_space(1)->dofmap()->index_map();
    index_maps[1].push_back(map.get());
  }

  // Create row and column local-to-global maps to attach to matrix
  std::array<std::vector<PetscInt>, 2> _maps;
  for (std::size_t i = 0; i < a.size(); ++i)
  {
    auto map = a[i][0]->function_space(0)->dofmap()->index_map();
    for (std::size_t k = 0; k < map->size(common::IndexMap::MapSize::ALL); ++k)
    {
      auto index_k = map->local_to_global(k);
      std::size_t index = get_global_index(index_maps[0], i, index_k);
      _maps[0].push_back(index);
    }
  }
  for (std::size_t i = 0; i < a[0].size(); ++i)
  {
    auto map = a[0][i]->function_space(1)->dofmap()->index_map();
    for (std::size_t k = 0; k < map->size(common::IndexMap::MapSize::ALL); ++k)
    {
      auto index_k = map->local_to_global(k);
      std::size_t index = get_global_index(index_maps[1], i, index_k);
      _maps[1].push_back(index);
    }
  }

  // exit(0);

  // Create PETSc local-to-global map/index sets and attach to matrix
  ISLocalToGlobalMapping petsc_local_to_global0, petsc_local_to_global1;
  ISLocalToGlobalMappingCreate(MPI_COMM_SELF, 1, _maps[0].size(),
                               _maps[0].data(), PETSC_COPY_VALUES,
                               &petsc_local_to_global0);
  ISLocalToGlobalMappingCreate(MPI_COMM_SELF, 1, _maps[1].size(),
                               _maps[1].data(), PETSC_COPY_VALUES,
                               &petsc_local_to_global1);
  MatSetLocalToGlobalMapping(A.mat(), petsc_local_to_global0,
                             petsc_local_to_global1);

  // Clean up local-to-global maps
  ISLocalToGlobalMappingDestroy(&petsc_local_to_global0);
  ISLocalToGlobalMappingDestroy(&petsc_local_to_global1);

  return A;
}
//-----------------------------------------------------------------------------
la::PETScVector fem::init_monolithic(std::vector<const fem::Form*> L)
{
  // FIXME: handle null blocks
  // FIXME: handle mixed block sizes

  std::size_t local_size = 0;
  std::vector<const common::IndexMap*> index_maps;
  for (std::size_t i = 0; i < L.size(); ++i)
  {
    assert(L[i]);
    assert(L[i]->rank() == 1);
    auto map = L[i]->function_space(0)->dofmap()->index_map();
    local_size += map->size(common::IndexMap::MapSize::OWNED);
    index_maps.push_back(map.get());
  }

  std::vector<std::size_t> ghosts;
  for (std::size_t i = 0; i < L.size(); ++i)
  {
    const Eigen::Array<la_index_t, Eigen::Dynamic, 1>& field_ghosts
        = index_maps[i]->ghosts();
    for (Eigen::Index j = 0; j < field_ghosts.size(); ++j)
    {
      std::size_t global_index
          = get_global_index(index_maps, i, field_ghosts[j]);
      ghosts.push_back(global_index);
    }
  }

  // Create map for combined problem
  common::IndexMap index_map(L[0]->mesh()->mpi_comm(), local_size, ghosts, 1);
  return la::PETScVector(index_map);
}
//-----------------------------------------------------------------------------
la::PETScMatrix dolfin::fem::init_matrix(const Form& a)
{
  bool keep_diagonal = false;
  if (a.rank() != 2)
  {
    throw std::runtime_error(
        "Cannot initialise matrx. Form is not a bilinear form");
  }

  // Get dof maps
  std::array<const GenericDofMap*, 2> dofmaps
      = {{a.function_space(0)->dofmap().get(),
          a.function_space(1)->dofmap().get()}};

  // Get mesh
  assert(a.mesh());
  const mesh::Mesh& mesh = *(a.mesh());

  common::Timer t0("Build sparsity");

  // Get common::IndexMaps for each dimension
  std::array<std::shared_ptr<const common::IndexMap>, 2> index_maps
      = {{dofmaps[0]->index_map(), dofmaps[1]->index_map()}};

  // Create and build sparsity pattern
  la::SparsityPattern pattern = SparsityPatternBuilder::build(
      mesh.mpi_comm(), mesh, dofmaps, (a.integrals().num_cell_integrals() > 0),
      (a.integrals().num_interior_facet_integrals() > 0),
      (a.integrals().num_exterior_facet_integrals() > 0),
      (a.integrals().num_vertex_integrals() > 0), keep_diagonal);
  t0.stop();

  // Initialize matrix
  common::Timer t1("Init tensor");
  la::PETScMatrix A(a.mesh()->mpi_comm(), pattern);
  t1.stop();

  // Insert zeros to dense rows in increasing order of column index
  // to avoid CSR data reallocation when assembling in random order
  // resulting in quadratic complexity; this has to be done before
  // inserting to diagonal below

  // Tabulate indices of dense rows
  Eigen::Array<std::size_t, Eigen::Dynamic, 1> global_dofs
      = dofmaps[0]->tabulate_global_dofs();
  if (global_dofs.size() > 0)
  {
    // Get local row range
    const common::IndexMap& index_map_0 = *dofmaps[0]->index_map();
    const auto row_range = A.local_range(0);

    assert(index_map_0.block_size() == 1);

    // Set zeros in dense rows in order of increasing column index
    const double block = 0.0;
    dolfin::la_index_t IJ[2];
    for (Eigen::Index i = 0; i < global_dofs.size(); ++i)
    {
      const std::int64_t I = index_map_0.local_to_global(global_dofs[i]);
      if (I >= row_range[0] && I < row_range[1])
      {
        IJ[0] = I;
        for (std::int64_t J = 0; J < A.size()[1]; J++)
        {
          IJ[1] = J;
          A.set(&block, 1, &IJ[0], 1, &IJ[1]);
        }
      }
    }

    // Eventually wait with assembly flush for keep_diagonal
    if (!keep_diagonal)
      A.apply(la::PETScMatrix::AssemblyType::FLUSH);
  }

  // FIXME: Check if there is a PETSc function for this
  // Insert zeros on the diagonal as diagonal entries may be
  // optimised away, e.g. when calling PETScMatrix::apply.
  if (keep_diagonal)
  {
    // Loop over rows and insert 0.0 on the diagonal
    const double block = 0.0;
    const auto row_range = A.local_range(0);
    const std::int64_t range = std::min(row_range[1], A.size()[1]);

    for (std::int64_t i = row_range[0]; i < range; i++)
    {
      const dolfin::la_index_t _i = i;
      A.set(&block, 1, &_i, 1, &_i);
    }

    A.apply(la::PETScMatrix::AssemblyType::FLUSH);
  }

  return A;
}
//-----------------------------------------------------------------------------
std::size_t
dolfin::fem::get_global_index(const std::vector<const common::IndexMap*> maps,
                              const unsigned int field, const unsigned int n)
{
  // Get process that owns global index
  int owner = maps[field]->owner(n);
  // if (MPI::rank(MPI_COMM_WORLD) == 1)
  //   std::cout << "    owning process: " << owner << std::endl;

  // Offset from lower rank processes
  std::size_t offset = 0;
  if (owner > 0)
  {
    for (std::size_t j = 0; j < maps.size(); ++j)
    {
      // if (MPI::rank(MPI_COMM_WORLD) == 1)
      //   std::cout << "   p off: " << maps[j]->_all_ranges[owner] <<
      //   std::endl;
      if (j != field)
      {
        offset += maps[j]->_all_ranges[owner];
      }
    }
  }

  // Local (process) offset
  for (unsigned int i = 0; i < field; ++i)
    offset += (maps[i]->_all_ranges[owner + 1] - maps[i]->_all_ranges[owner]);

  // if (MPI::rank(MPI_COMM_WORLD) == 2)
  //   std::cout << "    proc offeset (2): " << offset << std::endl;

  return n + offset;
}
//-----------------------------------------------------------------------------
