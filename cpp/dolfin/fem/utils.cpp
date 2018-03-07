// Copyright (C) 2013-2018 Johan Hake, Jan Blechta and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "utils.h"
#include <dolfin/common/ArrayView.h>
#include <dolfin/common/IndexMap.h>
#include <dolfin/common/Timer.h>
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

namespace
{
void _check_coordinates(const mesh::MeshGeometry& geometry,
                        const function::Function& position)
{
  dolfin_assert(position.function_space());
  dolfin_assert(position.function_space()->mesh());
  dolfin_assert(position.function_space()->dofmap());
  dolfin_assert(position.function_space()->element());
  dolfin_assert(position.function_space()->element()->ufc_element());

  if (position.function_space()->element()->ufc_element()->family()
      != std::string("Lagrange"))

  {
    log::dolfin_error(
        "fem_utils.cpp", "set/get mesh geometry coordinates from/to function",
        "expecting 'Lagrange' finite element family rather than '%s'",
        position.function_space()->element()->ufc_element()->family());
  }

  if (position.value_rank() != 1)
  {
    log::dolfin_error(
        "fem_utils.cpp", "set/get mesh geometry coordinates from/to function",
        "function has incorrect value rank %d, need 1", position.value_rank());
  }

  if (position.value_dimension(0) != geometry.dim())
  {
    log::dolfin_error("fem_utils.cpp",
                      "set/get mesh geometry coordinates from/to function",
                      "function value dimension %d and geometry dimension %d "
                      "do not match",
                      position.value_dimension(0), geometry.dim());
  }

  if (position.function_space()->element()->ufc_element()->degree()
      != geometry.degree())
  {
    log::dolfin_error(
        "fem_utils.cpp", "set/get mesh geometry coordinates from/to function",
        "function degree %d and geometry degree %d do not match",
        position.function_space()->element()->ufc_element()->degree(),
        geometry.degree());
  }
}

// This helper function sets geometry from position (if setting) or
// stores geometry into position (otherwise)
void _get_set_coordinates(mesh::MeshGeometry& geometry,
                          function::Function& position, const bool setting)
{
  auto& x = geometry.x();
  auto& v = *position.vector();
  const auto& dofmap = *position.function_space()->dofmap();
  const auto& mesh = *position.function_space()->mesh();
  const auto tdim = mesh.topology().dim();
  const auto gdim = mesh.geometry().dim();

  std::vector<std::size_t> num_local_entities(tdim + 1);
  std::vector<std::size_t> coords_per_entity(tdim + 1);
  std::vector<std::vector<std::vector<std::size_t>>> local_to_local(tdim + 1);
  std::vector<std::vector<std::size_t>> offsets(tdim + 1);

  for (std::size_t dim = 0; dim <= tdim; ++dim)
  {
    // Get number local entities
    num_local_entities[dim] = mesh.type().num_entities(dim);

    // Get local-to-local mapping of dofs
    local_to_local[dim].resize(num_local_entities[dim]);
    for (std::size_t local_ind = 0; local_ind != num_local_entities[dim];
         ++local_ind)
      dofmap.tabulate_entity_dofs(local_to_local[dim][local_ind], dim,
                                  local_ind);

    // Get entity offsets; could be retrieved directly from geometry
    coords_per_entity[dim] = geometry.num_entity_coordinates(dim);
    for (std::size_t coord_ind = 0; coord_ind != coords_per_entity[dim];
         ++coord_ind)
    {
      const auto offset = geometry.get_entity_index(dim, coord_ind, 0);
      offsets[dim].push_back(offset);
    }
  }

  // Initialize needed connectivities
  for (std::size_t dim = 0; dim <= tdim; ++dim)
  {
    if (coords_per_entity[dim] > 0)
      mesh.init(tdim, dim);
  }

  std::vector<double> values;
  const std::int32_t* global_entities;
  std::size_t xi, vi;

  // Get/set cell-by-cell
  for (auto& c : mesh::MeshRange<mesh::Cell>(mesh))
  {
    // Get/prepare values and dofs on cell
    auto cell_dofs = dofmap.cell_dofs(c.index());
    values.resize(cell_dofs.size());
    if (setting)
      v.get_local(values.data(), cell_dofs.size(), cell_dofs.data());

    // Iterate over all entities on cell
    for (std::size_t dim = 0; dim <= tdim; ++dim)
    {
      // Get local-to-global entity mapping
      if (!coords_per_entity[dim])
        continue;
      global_entities = c.entities(dim);

      for (std::size_t local_entity = 0;
           local_entity != num_local_entities[dim]; ++local_entity)
      {
        for (std::size_t local_dof = 0; local_dof != coords_per_entity[dim];
             ++local_dof)
        {
          for (std::size_t component = 0; component != gdim; ++component)
          {
            // Compute indices
            xi = gdim
                     * (offsets[dim][local_dof] + global_entities[local_entity])
                 + component;
            vi = local_to_local[dim][local_entity]
                               [gdim * local_dof + component];

            // Set one or other
            if (setting)
              x[xi] = values[vi];
            else
              values[vi] = x[xi];
          }
        }
      }
    }

    // Store cell contribution to dof vector (if getting)
    if (!setting)
      v.set_local(values.data(), cell_dofs.size(), cell_dofs.data());
  }

  if (!setting)
    v.apply();
}
}

//-----------------------------------------------------------------------------
void dolfin::fem::init(la::PETScVector& x, const Form& a)
{
  if (a.rank() != 1)
    throw std::runtime_error(
        "Cannot initialise vector. Form is not a linear form");

  if (!x.empty())
    throw std::runtime_error("Cannot initialise layout of non-empty matrix");

  // Get dof map
  auto dofmap = a.function_space(0)->dofmap();

  // Get dimensions and mapping across processes for each dimension
  auto index_map = dofmap->index_map();

  // FIXME: Do we need to sort out ghosts here
  // Build ghost
  // std::vector<dolfin::la_index_t> ghosts;

  // Build local-to-global index map
  int block_size = index_map->block_size();
  std::vector<la_index_t> local_to_global(
      index_map->size(common::IndexMap::MapSize::ALL));
  for (std::size_t i = 0; i < local_to_global.size(); ++i)
    local_to_global[i] = index_map->local_to_global(i);

  // Initialize vector
  x.init(index_map->local_range(), local_to_global, {}, block_size);
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
        petsc_mats[i][j] = mats[i][j]->mat();
        init(*mats[i][j], *a[i][j]);
      }
      else
        petsc_mats[i][j] = nullptr;
    }
  }

  // Initialise block (MatNest) matrix
  MatSetType(A.mat(), MATNEST);
  MatNestSetSubMats(A.mat(), petsc_mats.shape()[0], NULL, petsc_mats.shape()[1],
                    NULL, petsc_mats.data());
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
      vecs[i] = std::make_shared<la::PETScVector>(x.mpi_comm());
      petsc_vecs[i] = vecs[i]->vec();
      init(*vecs[i], *L[i]);
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

  std::cout << "Initialising block matrix" << std::endl;

  // Block shape
  const auto shape = boost::extents[a.size()][a[0].size()];

  boost::multi_array<std::shared_ptr<la::SparsityPattern>, 2> patterns(shape);
  std::vector<std::vector<const la::SparsityPattern*>> p(
      a.size(), std::vector<const la::SparsityPattern*>(a[0].size()));
  for (std::size_t row = 0; row < a.size(); ++row)
  {
    for (std::size_t col = 0; col < a[row].size(); ++col)
    {
      std::cout << "  Initialising block: " << row << ", " << col << std::endl;
      auto map0 = a[row][col]->function_space(0)->dofmap()->index_map();
      auto map1 = a[row][col]->function_space(1)->dofmap()->index_map();

      std::cout << "  Push Initialising block: " << std::endl;
      std::array<std::shared_ptr<const common::IndexMap>, 2> maps
          = {{map0, map1}};
      patterns[row][col]
          = std::make_shared<la::SparsityPattern>(A.mpi_comm(), maps, 0);

      // Build sparsity pattern
      std::cout << "  Build sparsity pattern " << std::endl;
      std::array<const GenericDofMap*, 2> dofmaps
          = {{a[row][col]->function_space(0)->dofmap().get(),
              a[row][col]->function_space(1)->dofmap().get()}};
      SparsityPatternBuilder::build(*patterns[row][col], *a[row][col]->mesh(),
                                    dofmaps, true, false, false, false, false);
      std::cout << "  End Build sparsity pattern " << std::endl;
      p[row][col] = patterns[row][col].get();
      std::cout << "  End push back sparsity pattern pointer " << std::endl;
    }
  }

  // Create merged sparsity pattern
  std::cout << "  Build merged sparsity pattern" << std::endl;
  la::SparsityPattern pattern(A.mpi_comm(), p);

  // Initialise matrix
  std::cout << "  Init parent matrix" << std::endl;
  A.init(pattern);
  std::cout << "  Post init parent matrix" << std::endl;
}
//-----------------------------------------------------------------------------
void fem::init_monolithic(la::PETScVector& x, std::vector<const fem::Form*> L)
{
  // if (a.rank() != 1)
  //  throw std::runtime_error(
  //      "Cannot initialise vector. Form is not a linear form");

  if (!x.empty())
    throw std::runtime_error("Cannot initialise layout of non-empty matrix");

  // FIXME: handle null blocks
  // FIXME: handle mixed block sizes

  std::size_t local_size = 0;
  for (std::size_t i = 0; i < L.size(); ++i)
  {
    auto map = L[i]->function_space(0)->dofmap()->index_map();
    local_size += map->size(common::IndexMap::MapSize::OWNED);
    // int block_size = map->block_size();
  }

  // Create map for combined problem
  common::IndexMap map(x.mpi_comm(), local_size, 1);

  std::vector<la_index_t> local_to_global(
      map.size(common::IndexMap::MapSize::ALL));
  for (std::size_t i = 0; i < local_to_global.size(); ++i)
  {
    local_to_global[i] = map.local_to_global(i);
    std::cout << "l2g: " << i << ", " << local_to_global[i] << std::endl;
  }

  // Initialize vector
  x.init(map.local_range(), local_to_global, {}, 1);
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
  dolfin_assert(a.mesh());
  const mesh::Mesh& mesh = *(a.mesh());

  common::Timer t0("Build sparsity");

  // Get common::IndexMaps for each dimension
  std::array<std::shared_ptr<const common::IndexMap>, 2> index_maps
      = {{dofmaps[0]->index_map(), dofmaps[1]->index_map()}};

  // Create and build sparsity pattern
  la::SparsityPattern pattern(A.mpi_comm(), index_maps, 0);
  SparsityPatternBuilder::build(
      pattern, mesh, dofmaps, (a.integrals().num_cell_integrals() > 0),
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
  const std::size_t primary_dim = pattern.primary_dim();
  std::vector<std::size_t> global_dofs;
  dofmaps[primary_dim]->tabulate_global_dofs(global_dofs);
  if (global_dofs.size() > 0)
  {
    // Get local row range
    const std::size_t primary_codim = primary_dim == 0 ? 1 : 0;
    const common::IndexMap& index_map_0 = *dofmaps[primary_dim]->index_map();
    const auto row_range = A.local_range(primary_dim);

    // Set zeros in dense rows in order of increasing column index
    const double block = 0.0;
    dolfin::la_index_t IJ[2];
    for (std::size_t i : global_dofs)
    {
      const std::int64_t I = index_map_0.local_to_global_index(i);
      if (I >= row_range[0] && I < row_range[1])
      {
        IJ[primary_dim] = I;
        for (std::int64_t J = 0; J < A.size(primary_codim); J++)
        {
          IJ[primary_codim] = J;
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
  dolfin_assert(space.mesh());
  dolfin_assert(space.dofmap());
  const mesh::Mesh& mesh = *space.mesh();
  const GenericDofMap& dofmap = *space.dofmap();

  if (dofmap.is_view())
  {
    log::dolfin_error("fem_utils.cpp", "tabulate vertex to dof map",
                      "Cannot tabulate vertex_to_dof_map for a subspace");
  }

  // Initialize vertex to cell connections
  const std::size_t top_dim = mesh.topology().dim();
  mesh.init(0, top_dim);

  // Num dofs per vertex
  const std::size_t dofs_per_vertex = dofmap.num_entity_dofs(0);
  const std::size_t vert_per_cell = mesh.topology()(top_dim, 0).size(0);
  if (vert_per_cell * dofs_per_vertex != dofmap.max_element_dofs())
  {
    log::dolfin_error("DofMap.cpp", "tabulate dof to vertex map",
                      "Can only tabulate dofs on vertices");
  }

  // Allocate data for tabulating local to local map
  std::vector<std::size_t> local_to_local_map(dofs_per_vertex);

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
    dolfin_assert(vertex_found);

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
void dolfin::fem::set_coordinates(mesh::MeshGeometry& geometry,
                                  const function::Function& position)
{
  _check_coordinates(geometry, position);
  _get_set_coordinates(geometry, const_cast<function::Function&>(position),
                       true);
}
//-----------------------------------------------------------------------------
void dolfin::fem::get_coordinates(function::Function& position,
                                  const mesh::MeshGeometry& geometry)
{
  _check_coordinates(geometry, position);
  _get_set_coordinates(const_cast<mesh::MeshGeometry&>(geometry), position,
                       false);
}
//-----------------------------------------------------------------------------
mesh::Mesh dolfin::fem::create_mesh(function::Function& coordinates)
{
  // FIXME: This function is a mess

  dolfin_assert(coordinates.function_space());
  dolfin_assert(coordinates.function_space()->element());
  dolfin_assert(coordinates.function_space()->element()->ufc_element());

  // Fetch old mesh and create new mesh
  const mesh::Mesh& mesh0 = *(coordinates.function_space()->mesh());
  mesh::Mesh mesh1(mesh0.mpi_comm());

  // FIXME: Share this code with Mesh assignment operaror; a need
  //        to duplicate its code here is not maintainable
  // Assign all data except geometry

  mesh1._topology = mesh0._topology;
  if (mesh0._cell_type)
    mesh1._cell_type.reset(
        mesh::CellType::create(mesh0._cell_type->cell_type()));
  else
    mesh1._cell_type.reset();
  mesh1._ordered = mesh0._ordered;
  mesh1._ghost_mode = mesh0._ghost_mode;

  // Rename
  mesh1.rename(mesh0.name(), mesh0.label());

  // Prepare a new geometry
  mesh1.geometry().init(
      mesh0.geometry().dim(),
      coordinates.function_space()->element()->ufc_element()->degree());
  std::vector<std::size_t> num_entities(mesh0.topology().dim() + 1);
  for (std::size_t dim = 0; dim <= mesh0.topology().dim(); ++dim)
    num_entities[dim] = mesh0.topology().size(dim);
  mesh1.geometry().init_entities(num_entities);

  // Assign coordinates
  set_coordinates(mesh1.geometry(), coordinates);

  return mesh1;
}
//-----------------------------------------------------------------------------
