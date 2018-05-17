// Copyright (C) 2013-2018 Johan Hake, Jan Blechta and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "utils.h"
#include <dolfin/common/ArrayView.h>
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
void fem::init_nest(la::PETScMatrix& A,
                    std::vector<std::vector<const fem::Form*>> a)
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
        mats[i][j] = std::make_shared<la::PETScMatrix>(A.mpi_comm());
        init(*mats[i][j], *a[i][j]);
        petsc_mats[i][j] = mats[i][j]->mat();
      }
      else
        petsc_mats[i][j] = nullptr;
    }
  }

  // Initialise block (MatNest) matrix
  MatSetType(A.mat(), MATNEST);
  MatNestSetSubMats(A.mat(), petsc_mats.shape()[0], NULL, petsc_mats.shape()[1],
                    NULL, petsc_mats.data());
  MatSetUp(A.mat());
}
//-----------------------------------------------------------------------------
void fem::init_nest(la::PETScVector& x, std::vector<const fem::Form*> L)
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
  VecCreateNest(x.mpi_comm(), petsc_vecs.size(), NULL, petsc_vecs.data(), &y);
  x.reset(y);
  VecDestroy(&y);

  /*
  VecSetType(x.vec(), VECNEST);
  VecNestSetSubVecs(x.vec(), petsc_vecs.size(), NULL, petsc_vecs.data());
  */
}
//-----------------------------------------------------------------------------
void fem::init_monolithic(la::PETScMatrix& A,
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
  la::SparsityPattern pattern(A.mpi_comm(), p);

  // Initialise matrix
  // std::cout << "  Init parent matrix" << std::endl;
  A.init(pattern);
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
}
//-----------------------------------------------------------------------------
void fem::init_monolithic(la::PETScVector& x, std::vector<const fem::Form*> L)
{
  // if (a.rank() != 1)
  //  throw std::runtime_error(
  //      "Cannot initialise vector. Form is not a linear form");

  if (!x.empty())
    throw std::runtime_error("Cannot initialise non-empty vector");

  // FIXME: handle null blocks
  // FIXME: handle mixed block sizes

  std::size_t local_size = 0;
  std::vector<const common::IndexMap*> index_maps;
  for (std::size_t i = 0; i < L.size(); ++i)
  {
    auto map = L[i]->function_space(0)->dofmap()->index_map();
    local_size += map->size(common::IndexMap::MapSize::OWNED);
    index_maps.push_back(map.get());
    // int block_size = map->block_size();
  }

  // Create map for combined problem
  common::IndexMap index_map(x.mpi_comm(), local_size, {}, 1);

  // std::vector<la_index_t> local_to_global(
  //     map.size(common::IndexMap::MapSize::ALL));
  // for (std::size_t i = 0; i < local_to_global.size(); ++i)
  // {
  //   local_to_global[i] = map.local_to_global(i);
  //   // std::cout << "l2g: " << i << ", " << local_to_global[i] << std::endl;
  // }

  std::vector<la_index_t> local_to_global;
  for (std::size_t i = 0; i < L.size(); ++i)
  {
    assert(L[i]);
    auto map = L[i]->function_space(0)->dofmap()->index_map();
    for (std::size_t k = 0; k < map->size(common::IndexMap::MapSize::ALL); ++k)
    {
      auto index_k = map->local_to_global(k);
      std::size_t index = get_global_index(index_maps, i, index_k);
      local_to_global.push_back(index);
    }
  }

  // Initialize vector. This needs fixing because in this case we have a
  // non-standard l2g map
  throw std::runtime_error("This case needs to be updated");
  // x._init(index_map.local_range(), local_to_global, {}, 1);
}
//-----------------------------------------------------------------------------
void dolfin::fem::init(la::PETScMatrix& A, const Form& a)
{
  bool keep_diagonal = false;
  if (a.rank() != 2)
  {
    throw std::runtime_error(
        "Cannot initialise matrx. Form is not a bilinear form");
  }

  if (!A.empty())
    throw std::runtime_error("Cannot initialise layout of non-empty matrix");

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
      A.mpi_comm(), mesh, dofmaps, (a.integrals().num_cell_integrals() > 0),
      (a.integrals().num_interior_facet_integrals() > 0),
      (a.integrals().num_exterior_facet_integrals() > 0),
      (a.integrals().num_vertex_integrals() > 0), keep_diagonal);
  t0.stop();

  // Initialize matrix
  common::Timer t1("Init tensor");
  A.init(pattern);
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
        for (std::int64_t J = 0; J < A.size(1); J++)
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
    const std::int64_t range = std::min(row_range[1], A.size(1));

    for (std::int64_t i = row_range[0]; i < range; i++)
    {
      const dolfin::la_index_t _i = i;
      A.set(&block, 1, &_i, 1, &_i);
    }

    A.apply(la::PETScMatrix::AssemblyType::FLUSH);
  }
}
//-----------------------------------------------------------------------------
std::vector<std::size_t>
dolfin::fem::dof_to_vertex_map(const function::FunctionSpace& space)
{
  // Get vertex_to_dof_map and invert it
  const std::vector<dolfin::la_index_t> vertex_map = vertex_to_dof_map(space);
  std::vector<std::size_t> return_map(vertex_map.size());
  for (std::size_t i = 0; i < vertex_map.size(); i++)
    return_map[vertex_map[i]] = i;
  return return_map;
}
//-----------------------------------------------------------------------------
std::vector<dolfin::la_index_t>
dolfin::fem::vertex_to_dof_map(const function::FunctionSpace& space)
{
  // Get the mesh
  assert(space.mesh());
  assert(space.dofmap());
  const mesh::Mesh& mesh = *space.mesh();
  const GenericDofMap& dofmap = *space.dofmap();

  if (dofmap.is_view())
    std::runtime_error("Cannot tabulate vertex_to_dof_map for a subspace");

  // Initialize vertex to cell connections
  const std::size_t top_dim = mesh.topology().dim();
  mesh.init(0, top_dim);

  // Num dofs per vertex
  const std::size_t dofs_per_vertex = dofmap.num_entity_dofs(0);
  const std::size_t vert_per_cell
      = mesh.topology().connectivity(top_dim, 0).size(0);
  if (vert_per_cell * dofs_per_vertex != dofmap.max_element_dofs())
    std::runtime_error("Can only tabulate dofs on vertices");

  // Allocate data for tabulating local to local map
  std::vector<int> local_to_local_map(dofs_per_vertex);

  // Create return data structure
  std::vector<dolfin::la_index_t> return_map(dofs_per_vertex
                                             * mesh.num_entities(0));

  // Iterate over all vertices (including ghosts)
  std::size_t local_vertex_ind = 0;
  const auto v_begin = mesh::MeshIterator<mesh::Vertex>(mesh, 0);
  const auto v_end
      = mesh::MeshIterator<mesh::Vertex>(mesh, mesh.num_entities(0));
  for (auto vertex = v_begin; vertex != v_end; ++vertex)
  {
    // Get the first cell connected to the vertex
    const mesh::Cell cell(mesh, vertex->entities(top_dim)[0]);

// Find local vertex number
#ifdef DEBUG
    bool vertex_found = false;
#endif
    for (std::size_t i = 0; i < cell.num_entities(0); i++)
    {
      if (cell.entities(0)[i] == vertex->index())
      {
        local_vertex_ind = i;
#ifdef DEBUG
        vertex_found = true;
#endif
        break;
      }
    }
    assert(vertex_found);

    // Get all cell dofs
    auto cell_dofs = dofmap.cell_dofs(cell.index());

    // Tabulate local to local map of dofs on local vertex
    dofmap.tabulate_entity_dofs(local_to_local_map, 0, local_vertex_ind);

    // Fill local dofs for the vertex
    for (std::size_t local_dof = 0; local_dof < dofs_per_vertex; local_dof++)
    {
      const dolfin::la_index_t global_dof
          = cell_dofs[local_to_local_map[local_dof]];
      return_map[dofs_per_vertex * vertex->index() + local_dof] = global_dof;
    }
  }

  // Return the map
  return return_map;
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
