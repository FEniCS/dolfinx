// Copyright (C) 2013-2019 Johan Hake, Jan Blechta and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "utils.h"
#include <Eigen/Dense>
#include <array>
#include <dolfin/common/IndexMap.h>
#include <dolfin/common/Timer.h>
#include <dolfin/common/types.h>
#include <dolfin/fem/DofMap.h>
#include <dolfin/fem/DofMapBuilder.h>
#include <dolfin/fem/Form.h>
#include <dolfin/fem/SparsityPatternBuilder.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/la/PETScMatrix.h>
#include <dolfin/la/PETScVector.h>
#include <dolfin/la/SparsityPattern.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshEntity.h>
#include <dolfin/mesh/MeshIterator.h>
#include <memory>
#include <ufc.h>

using namespace dolfin;

namespace
{
// Try to figure out block size. FIXME - replace elsewhere
int analyse_block_structure(
    const std::vector<std::shared_ptr<const fem::ElementDofLayout>> sub_dofmaps)
{
  // Must be at least two subdofmaps
  if (sub_dofmaps.size() < 2)
    return 1;

  for (auto dmap : sub_dofmaps)
  {
    assert(dmap);

    // If any subdofmaps have subdofmaps themselves, ignore any
    // potential block structure
    if (dmap->num_sub_dofmaps() > 0)
      return 1;

    // Check number of dofs are the same for all subdofmaps
    for (int d = 0; d < 4; ++d)
    {
      if (sub_dofmaps[0]->num_entity_dofs(d) != dmap->num_entity_dofs(d))
        return 1;
    }
  }

  // All subdofmaps are simple, and have the same number of dofs
  return sub_dofmaps.size();
}
} // namespace

//-----------------------------------------------------------------------------
std::vector<std::vector<std::shared_ptr<const common::IndexMap>>>
fem::blocked_index_sets(const std::vector<std::vector<const fem::Form*>> a)
{
  std::vector<std::vector<std::shared_ptr<const common::IndexMap>>> maps(2);
  maps[0].resize(a.size());
  maps[1].resize(a[0].size());

  // Loop over rows and columns
  for (std::size_t i = 0; i < a.size(); ++i)
  {
    for (std::size_t j = 0; j < a[i].size(); ++j)
    {
      if (a[i][j])
      {
        assert(a[i][j]->rank() == 2);
        auto m0 = a[i][j]->function_space(0)->dofmap->index_map;
        auto m1 = a[i][j]->function_space(1)->dofmap->index_map;
        if (!maps[0][i])
          maps[0][i] = m0;
        else
        {
          // TODO: Check that maps are the same
        }

        if (!maps[1][j])
          maps[1][j] = m1;
        else
        {
          // TODO: Check that maps are the same
        }
      }
    }
  }

  for (std::size_t i = 0; i < maps.size(); ++i)
  {
    for (std::size_t j = 0; j < maps[i].size(); ++j)
    {
      if (!maps[i][j])
        throw std::runtime_error("Could not deduce all block index maps.");
    }
  }

  return maps;
}
//-----------------------------------------------------------------------------
la::PETScMatrix dolfin::fem::create_matrix(const Form& a)
{
  bool keep_diagonal = false;
  if (a.rank() != 2)
  {
    throw std::runtime_error(
        "Cannot initialise matrx. Form is not a bilinear form");
  }

  // Get dof maps
  std::array<const DofMap*, 2> dofmaps = {
      {a.function_space(0)->dofmap.get(), a.function_space(1)->dofmap.get()}};

  // Get mesh
  assert(a.mesh());
  const mesh::Mesh& mesh = *(a.mesh());

  common::Timer t0("Build sparsity");

  // Get common::IndexMaps for each dimension
  std::array<std::shared_ptr<const common::IndexMap>, 2> index_maps
      = {{dofmaps[0]->index_map, dofmaps[1]->index_map}};

  // Create and build sparsity pattern
  la::SparsityPattern pattern(mesh.mpi_comm(), index_maps);
  if (a.integrals().num_integrals(fem::FormIntegrals::Type::cell) > 0)
    SparsityPatternBuilder::cells(pattern, mesh, {{dofmaps[0], dofmaps[1]}});
  if (a.integrals().num_integrals(fem::FormIntegrals::Type::interior_facet) > 0)
    SparsityPatternBuilder::interior_facets(pattern, mesh,
                                            {{dofmaps[0], dofmaps[1]}});
  if (a.integrals().num_integrals(fem::FormIntegrals::Type::exterior_facet) > 0)
    SparsityPatternBuilder::exterior_facets(pattern, mesh,
                                            {{dofmaps[0], dofmaps[1]}});
  pattern.assemble();
  t0.stop();

  // Initialize matrix
  common::Timer t1("Init tensor");
  la::PETScMatrix A(a.mesh()->mpi_comm(), pattern);
  t1.stop();

  // FIXME: Check if there is a PETSc function for this
  // Insert zeros on the diagonal as diagonal entries may be
  // optimised away, e.g. when calling PETScMatrix::apply.
  if (keep_diagonal)
  {
    // Loop over rows and insert 0.0 on the diagonal
    const PetscScalar block = 0.0;
    std::array<PetscInt, 2> row_range;
    MatGetOwnershipRange(A.mat(), &row_range[0], &row_range[1]);
    const std::int64_t range = std::min(row_range[1], (PetscInt)A.size()[1]);

    for (std::int64_t i = row_range[0]; i < range; i++)
    {
      const PetscInt _i = i;
      A.set(&block, 1, &_i, 1, &_i);
    }

    A.apply(la::PETScMatrix::AssemblyType::FLUSH);
  }

  return A;
}
//-----------------------------------------------------------------------------
la::PETScMatrix
fem::create_matrix_block(std::vector<std::vector<const fem::Form*>> a)
{
  // FIXME: this assumes that a00 is not null
  const mesh::Mesh& mesh = *a[0][0]->mesh();

  // FIXME: assume no null block in first row or column
  // Extract and check row/column ranges
  // std::vector<std::shared_ptr<const common::IndexMap>> rmaps, cmaps;
  // for (std::size_t row = 0; row < a.size(); ++row)
  //   rmaps.push_back(a[row][0]->function_space(0)->dofmap->index_map;
  // for (std::size_t col = 0; col < a[0].size(); ++col)
  //   cmaps.push_back(a[0][col]->function_space(1)->dofmap->index_map;

  std::vector<std::vector<std::shared_ptr<const common::IndexMap>>> maps
      = blocked_index_sets(a);
  std::vector<std::shared_ptr<const common::IndexMap>> rmaps = maps[0];
  std::vector<std::shared_ptr<const common::IndexMap>> cmaps = maps[1];

  // Build sparsity pattern for each block
  std::vector<std::vector<std::unique_ptr<la::SparsityPattern>>> patterns(
      a.size());
  std::vector<std::vector<const la::SparsityPattern*>> p(
      a.size(), std::vector<const la::SparsityPattern*>(a[0].size()));
  for (std::size_t row = 0; row < a.size(); ++row)
  {
    for (std::size_t col = 0; col < a[row].size(); ++col)
    {
      if (a[row][col])
      {
        // Build sparsity pattern for block
        std::array<const DofMap*, 2> dofmaps
            = {{a[row][col]->function_space(0)->dofmap.get(),
                a[row][col]->function_space(1)->dofmap.get()}};
        // auto sp = std::make_unique<la::SparsityPattern>(
        //     SparsityPatternBuilder::build(mesh.mpi_comm(), mesh, dofmaps,
        //     true,
        //                                   false, false));

        std::array<std::shared_ptr<const common::IndexMap>, 2> index_maps
            = {{dofmaps[0]->index_map, dofmaps[1]->index_map}};
        auto sp = std::make_unique<la::SparsityPattern>(mesh.mpi_comm(),
                                                        index_maps);
        if (a[row][col]->integrals().num_integrals(
                fem::FormIntegrals::Type::cell)
            > 0)
          SparsityPatternBuilder::cells(*sp, mesh, dofmaps);
        if (a[row][col]->integrals().num_integrals(
                fem::FormIntegrals::Type::interior_facet)
            > 0)
        {
          SparsityPatternBuilder::interior_facets(*sp, mesh, dofmaps);
        }
        if (a[row][col]->integrals().num_integrals(
                fem::FormIntegrals::Type::exterior_facet)
            > 0)
        {
          SparsityPatternBuilder::exterior_facets(*sp, mesh, dofmaps);
        }
        sp->assemble();
        patterns[row].push_back(std::move(sp));
      }
      else
      {
        // FIXME: create sparsity pattern that has just a row/col range
        const std::array<std::shared_ptr<const common::IndexMap>, 2> maps
            = {{rmaps[row], cmaps[col]}};
        auto sp = std::make_unique<la::SparsityPattern>(mesh.mpi_comm(), maps);
        patterns[row].push_back(std::move(sp));
      }

      p[row][col] = patterns[row][col].get();
      assert(p[row][col]);
    }
  }

  // Create merged sparsity pattern
  la::SparsityPattern pattern(mesh.mpi_comm(), p);

  // Initialise matrix
  la::PETScMatrix A(mesh.mpi_comm(), pattern);

  // Build list of row and column index maps (over each block)
  std::array<std::vector<const common::IndexMap*>, 2> index_maps;
  for (auto& m : maps[0])
    index_maps[0].push_back(m.get());
  for (auto& m : maps[1])
    index_maps[1].push_back(m.get());

  // Create row and column local-to-global maps to attach to matrix
  std::array<std::vector<PetscInt>, 2> _maps;
  for (std::size_t i = 0; i < maps[0].size(); ++i)
  {
    auto map = maps[0][i];
    std::size_t size = map->size_local() + map->num_ghosts();
    const int bs0 = map->block_size;
    for (std::size_t k = 0; k < size; ++k)
    {
      std::size_t index_k = map->local_to_global(k);
      for (int block = 0; block < bs0; ++block)
      {
        std::size_t index
            = get_global_index(index_maps[0], i, index_k * bs0 + block);
        _maps[0].push_back(index);
      }
    }
  }

  for (std::size_t i = 0; i < maps[1].size(); ++i)
  {
    auto map = maps[1][i];
    std::size_t size = map->size_local() + map->num_ghosts();
    const int bs1 = map->block_size;
    for (std::size_t k = 0; k < size; ++k)
    {
      std::size_t index_k = map->local_to_global(k);
      for (int block = 0; block < bs1; ++block)
      {
        std::size_t index
            = get_global_index(index_maps[1], i, index_k * bs1 + block);
        _maps[1].push_back(index);
      }
    }
  }

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
la::PETScMatrix
fem::create_matrix_nest(std::vector<std::vector<const fem::Form*>> a)
{
  // Check that array of forms is not empty and is square
  if (a.empty())
    throw std::runtime_error("Cannot created nested matrix without forms.");
  for (const auto& a_row : a)
  {
    if (a_row.size() != a[0].size())
    {
      throw std::runtime_error(
          "Array for forms must be rectangular to initialised nested matrix.");
    }
  }

  // Loop over each form and create matrix
  Eigen::Array<std::shared_ptr<la::PETScMatrix>, Eigen::Dynamic, Eigen::Dynamic,
               Eigen::RowMajor>
      mats(a.size(), a[0].size());
  Eigen::Array<Mat, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> petsc_mats(
      a.size(), a[0].size());
  for (std::size_t i = 0; i < a.size(); ++i)
  {
    for (std::size_t j = 0; j < a[i].size(); ++j)
    {
      if (a[i][j])
      {
        mats(i, j) = std::make_shared<la::PETScMatrix>(create_matrix(*a[i][j]));
        petsc_mats(i, j) = mats(i, j)->mat();
        // mats[i][j] =
        // std::make_shared<la::PETScMatrix>(create_matrix(*a[i][j]));
        // petsc_mats[i][j] = mats[i][j]->mat();
      }
      else
        petsc_mats(i, j) = nullptr;
    }
  }

  // Initialise block (MatNest) matrix
  Mat _A;
  MatCreate(a[0][0]->mesh()->mpi_comm(), &_A);
  MatSetType(_A, MATNEST);
  MatNestSetSubMats(_A, petsc_mats.rows(), nullptr, petsc_mats.cols(), nullptr,
                    petsc_mats.data());
  MatSetUp(_A);

  return la::PETScMatrix(_A);
}
//-----------------------------------------------------------------------------
la::PETScVector fem::create_vector_block(std::vector<const fem::Form*> L)
{
  // FIXME: handle null blocks?

  // FIXME: handle consatnt block size > 1

  // std::size_t local_size = 0;
  std::vector<const common::IndexMap*> index_maps;
  for (const fem::Form* form : L)
  {
    assert(form);
    assert(form->rank() == 1);
    auto map = form->function_space(0)->dofmap->index_map;
    index_maps.push_back(map.get());
  }

  std::size_t local_size = 0;
  std::vector<std::int64_t> ghosts;
  for (std::size_t i = 0; i < L.size(); ++i)
  {
    const common::IndexMap* map = index_maps[i];
    const int bs = index_maps[i]->block_size;
    local_size += map->size_local() * bs;

    const Eigen::Array<PetscInt, Eigen::Dynamic, 1>& field_ghosts
        = map->ghosts();
    for (Eigen::Index j = 0; j < field_ghosts.size(); ++j)
    {
      for (int k = 0; k < bs; ++k)
      {
        std::int64_t global_index
            = get_global_index(index_maps, i, bs * field_ghosts[j] + k);
        ghosts.push_back(global_index);
      }
    }
  }

  // Create map for combined problem
  common::IndexMap index_map(L[0]->mesh()->mpi_comm(), local_size, ghosts, 1);
  return la::PETScVector(index_map);
}
//-----------------------------------------------------------------------------
la::PETScVector fem::create_vector_nest(std::vector<const fem::Form*> L)
{
  // Loop over each form and create vector
  std::vector<std::shared_ptr<la::PETScVector>> vecs(L.size());
  std::vector<Vec> petsc_vecs(L.size());
  for (std::size_t i = 0; i < L.size(); ++i)
  {
    if (L[i])
    {
      const common::IndexMap& index_map
          = *L[i]->function_space(0)->dofmap->index_map;
      vecs[i] = std::make_shared<la::PETScVector>(index_map);
      petsc_vecs[i] = vecs[i]->vec();
    }
    else
      petsc_vecs[i] = nullptr;
  }

  // Create nested (VecNest) vector
  Vec y;
  VecCreateNest(vecs[0]->mpi_comm(), petsc_vecs.size(), nullptr,
                petsc_vecs.data(), &y);
  return la::PETScVector(y);
}
//-----------------------------------------------------------------------------
std::size_t
dolfin::fem::get_global_index(const std::vector<const common::IndexMap*> maps,
                              const unsigned int field,
                              const unsigned int index)
{
  // FIXME: handle/check block size > 1

  // Get process that owns global index
  const int bs = maps[field]->block_size;
  int owner = maps[field]->owner(index / bs);

  // Offset from lower rank processes
  std::size_t offset = 0;
  if (owner > 0)
  {
    for (std::size_t j = 0; j < maps.size(); ++j)
    {
      if (j != field)
        offset += maps[j]->_all_ranges[owner] * maps[j]->block_size;
    }
  }

  // Local (process) offset
  for (unsigned int i = 0; i < field; ++i)
  {
    offset += (maps[i]->_all_ranges[owner + 1] - maps[i]->_all_ranges[owner])
              * maps[i]->block_size;
  }

  return index + offset;
}
//-----------------------------------------------------------------------------
fem::ElementDofLayout
fem::create_element_dof_layout(const ufc_dofmap& dofmap,
                               const mesh::CellType cell_type,
                               const std::vector<int>& parent_map)
{
  // Copy over number of dofs per entity type
  std::array<int, 4> num_entity_dofs;
  std::copy(dofmap.num_entity_dofs, dofmap.num_entity_dofs + 4,
            num_entity_dofs.data());

  // Fill entity dof indices
  const int tdim = mesh::cell_dim(cell_type);
  std::vector<std::vector<std::set<int>>> entity_dofs(tdim + 1);
  std::vector<int> work_array;
  for (int dim = 0; dim <= tdim; ++dim)
  {
    const int num_entities = mesh::cell_num_entities(cell_type, dim);
    entity_dofs[dim].resize(num_entities);
    for (int i = 0; i < num_entities; ++i)
    {
      work_array.resize(num_entity_dofs[dim]);
      dofmap.tabulate_entity_dofs(work_array.data(), dim, i);
      entity_dofs[dim][i] = std::set<int>(work_array.begin(), work_array.end());
    }
  }

  // TODO: UFC dofmaps just use simple offset for each field but this
  // could be different for custom dofmaps This data should come
  // directly from the UFC interface in place of the the implicit
  // assumption

  // Create UFC subdofmaps and compute offset
  std::vector<std::shared_ptr<ufc_dofmap>> ufc_sub_dofmaps;
  std::vector<int> offsets(1, 0);
  for (int i = 0; i < dofmap.num_sub_dofmaps; ++i)
  {
    auto ufc_sub_dofmap
        = std::shared_ptr<ufc_dofmap>(dofmap.create_sub_dofmap(i));
    ufc_sub_dofmaps.push_back(ufc_sub_dofmap);
    const int num_dofs = ufc_sub_dofmap->num_element_support_dofs;
    offsets.push_back(offsets.back() + num_dofs);
  }

  std::vector<std::shared_ptr<const fem::ElementDofLayout>> sub_dofmaps;
  for (std::size_t i = 0; i < ufc_sub_dofmaps.size(); ++i)
  {
    auto ufc_sub_dofmap = ufc_sub_dofmaps[i];
    assert(ufc_sub_dofmap);
    std::vector<int> parent_map_sub(ufc_sub_dofmap->num_element_support_dofs);
    std::iota(parent_map_sub.begin(), parent_map_sub.end(), offsets[i]);
    sub_dofmaps.push_back(
        std::make_shared<fem::ElementDofLayout>(create_element_dof_layout(
            *ufc_sub_dofmaps[i], cell_type, parent_map_sub)));
  }

  // Check for "block structure". This should ultimately be replaced,
  // but keep for now to mimic existing code
  const int block_size = analyse_block_structure(sub_dofmaps);

  return fem::ElementDofLayout(block_size, entity_dofs, parent_map, sub_dofmaps,
                               cell_type);
}
//-----------------------------------------------------------------------------
fem::DofMap fem::create_dofmap(const ufc_dofmap& ufc_dofmap,
                               const mesh::Mesh& mesh)
{
  return DofMapBuilder::build(
      mesh, std::make_shared<ElementDofLayout>(
                create_element_dof_layout(ufc_dofmap, mesh.cell_type)));
}
//-----------------------------------------------------------------------------
std::vector<std::tuple<int, std::string, std::shared_ptr<function::Function>>>
fem::get_coeffs_from_ufc_form(const ufc_form& ufc_form)
{
  std::vector<std::tuple<int, std::string, std::shared_ptr<function::Function>>>
      coeffs;
  const char** names = ufc_form.coefficient_name_map();
  for (int i = 0; i < ufc_form.num_coefficients; ++i)
  {
    coeffs.push_back(
        std::make_tuple<int, std::string, std::shared_ptr<function::Function>>(
            ufc_form.original_coefficient_position(i), names[i], nullptr));
  }
  return coeffs;
}
//-----------------------------------------------------------------------------
std::vector<std::pair<std::string, std::shared_ptr<function::Constant>>>
fem::get_constants_from_ufc_form(const ufc_form& ufc_form)
{
  std::vector<std::pair<std::string, std::shared_ptr<function::Constant>>>
      constants;
  const char** names = ufc_form.constant_name_map();
  for (int i = 0; i < ufc_form.num_constants; ++i)
  {
    constants.push_back(
        std::make_pair<std::string, std::shared_ptr<function::Constant>>(
            names[i], nullptr));
  }
  return constants;
}
//-----------------------------------------------------------------------------
fem::Form fem::create_form(
    const ufc_form& ufc_form,
    const std::vector<std::shared_ptr<const function::FunctionSpace>>& spaces)
{
  assert(ufc_form.rank == (int)spaces.size());

  // Check argument function spaces
  for (std::size_t i = 0; i < spaces.size(); ++i)
  {
    assert(spaces[i]->element);
    std::unique_ptr<ufc_finite_element, decltype(free)*> ufc_element(
        ufc_form.create_finite_element(i), free);

    assert(ufc_element);
    if (std::string(ufc_element->signature) != spaces[i]->element->signature())
    {
      throw std::runtime_error(
          "Cannot create form. Wrong type of function space for argument.");
    }
  }

  // Get list of integral IDs, and load tabulate tensor into memory for each
  FormIntegrals integrals;
  std::vector<int> cell_integral_ids(ufc_form.num_cell_integrals);
  ufc_form.get_cell_integral_ids(cell_integral_ids.data());
  for (int id : cell_integral_ids)
  {
    ufc_integral* cell_integral = ufc_form.create_cell_integral(id);
    assert(cell_integral);
    integrals.register_tabulate_tensor(FormIntegrals::Type::cell, id,
                                       cell_integral->tabulate_tensor);
    std::free(cell_integral);
  }

  std::vector<int> exterior_facet_integral_ids(
      ufc_form.num_exterior_facet_integrals);
  ufc_form.get_exterior_facet_integral_ids(exterior_facet_integral_ids.data());
  for (int id : exterior_facet_integral_ids)
  {
    ufc_integral* exterior_facet_integral
        = ufc_form.create_exterior_facet_integral(id);
    assert(exterior_facet_integral);
    integrals.register_tabulate_tensor(
        FormIntegrals::Type::exterior_facet, id,
        exterior_facet_integral->tabulate_tensor);
    std::free(exterior_facet_integral);
  }

  std::vector<int> interior_facet_integral_ids(
      ufc_form.num_interior_facet_integrals);
  ufc_form.get_interior_facet_integral_ids(interior_facet_integral_ids.data());
  for (int id : interior_facet_integral_ids)
  {
    ufc_integral* interior_facet_integral
        = ufc_form.create_interior_facet_integral(id);
    assert(interior_facet_integral);
    integrals.register_tabulate_tensor(
        FormIntegrals::Type::interior_facet, id,
        interior_facet_integral->tabulate_tensor);
    std::free(interior_facet_integral);
  }

  // Not currently working
  std::vector<int> vertex_integral_ids(ufc_form.num_vertex_integrals);
  ufc_form.get_vertex_integral_ids(vertex_integral_ids.data());
  if (vertex_integral_ids.size() > 0)
  {
    throw std::runtime_error(
        "Vertex integrals not supported. Under development.");
  }

  // Create CoordinateMapping
  ufc_coordinate_mapping* cmap = ufc_form.create_coordinate_mapping();
  std::shared_ptr<const fem::CoordinateMapping> coord_mapping
      = fem::get_cmap_from_ufc_cmap(*cmap);
  std::free(cmap);

  return fem::Form(spaces, integrals,
                   FormCoefficients(fem::get_coeffs_from_ufc_form(ufc_form)),
                   fem::get_constants_from_ufc_form(ufc_form), coord_mapping);
}
//-----------------------------------------------------------------------------
std::shared_ptr<const fem::CoordinateMapping>
fem::get_cmap_from_ufc_cmap(const ufc_coordinate_mapping& ufc_cmap)
{
  static const std::map<ufc_shape, mesh::CellType> ufc_to_cell
      = {{vertex, mesh::CellType::point},
         {interval, mesh::CellType::interval},
         {triangle, mesh::CellType::triangle},
         {tetrahedron, mesh::CellType::tetrahedron},
         {quadrilateral, mesh::CellType::quadrilateral},
         {hexahedron, mesh::CellType::hexahedron}};
  const auto it = ufc_to_cell.find(ufc_cmap.cell_shape);
  assert(it != ufc_to_cell.end());

  mesh::CellType cell_type = it->second;
  assert(ufc_cmap.topological_dimension == mesh::cell_dim(cell_type));

  return std::make_shared<fem::CoordinateMapping>(
      cell_type, ufc_cmap.topological_dimension, ufc_cmap.geometric_dimension,
      ufc_cmap.signature, ufc_cmap.compute_physical_coordinates,
      ufc_cmap.compute_reference_geometry);
}
//-----------------------------------------------------------------------------
