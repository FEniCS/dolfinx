// Copyright (C) 2013-2019 Johan Hake, Jan Blechta and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "utils.h"
#include <Eigen/Dense>
#include <array>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/log.h>
#include <dolfinx/common/types.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/DofMapBuilder.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/SparsityPatternBuilder.h>
#include <dolfinx/function/Constant.h>
#include <dolfinx/function/Function.h>
#include <dolfinx/function/FunctionSpace.h>
#include <dolfinx/la/PETScMatrix.h>
#include <dolfinx/la/PETScVector.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <dolfinx/mesh/TopologyComputation.h>
#include <memory>
#include <petscsys.h>
#include <string>
#include <ufc.h>

using namespace dolfinx;

namespace
{
//-----------------------------------------------------------------------------
int get_num_permutations(const mesh::CellType cell_type)
{
  // In general, this will return num_edges + 2*num_faces + 4*num_volumes
  switch (cell_type)
  {
  case (mesh::CellType::point):
    return 0;
  case (mesh::CellType::interval):
    return 0;
  case (mesh::CellType::triangle):
    return 3;
  case (mesh::CellType::tetrahedron):
    return 14;
  case (mesh::CellType::quadrilateral):
    return 4;
  case (mesh::CellType::hexahedron):
    return 24;
  default:
    LOG(WARNING) << "Dof permutations are not defined for this cell type. High "
                    "order elements may be incorrect.";
    return 0;
  }
}
// Try to figure out block size. FIXME - replace elsewhere
int analyse_block_structure(
    const std::vector<std::shared_ptr<const fem::ElementDofLayout>>&
        sub_dofmaps)
{
  // Must be at least two subdofmaps
  if (sub_dofmaps.size() < 2)
    return 1;

  for (const auto& dmap : sub_dofmaps)
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
std::array<std::vector<std::shared_ptr<const function::FunctionSpace>>, 2>
fem::block_function_spaces(
    const Eigen::Ref<const Eigen::Array<const fem::Form*, Eigen::Dynamic,
                                        Eigen::Dynamic, Eigen::RowMajor>>& a)
{
  std::array<std::vector<std::shared_ptr<const function::FunctionSpace>>, 2> V;
  V[0] = std::vector<std::shared_ptr<const function::FunctionSpace>>(a.rows(),
                                                                     nullptr);
  V[1] = std::vector<std::shared_ptr<const function::FunctionSpace>>(a.cols(),
                                                                     nullptr);

  // Loop over rows
  for (int i = 0; i < a.rows(); ++i)
  {
    // Loop over columns
    for (int j = 0; j < a.cols(); ++j)
    {
      if (a(i, j))
      {
        assert(a(i, j)->rank() == 2);

        if (!V[0][i])
          V[0][i] = a(i, j)->function_space(0);
        else
        {
          if (V[0][i] != a(i, j)->function_space(0))
            throw std::runtime_error("Mismatched test space for row.");
        }

        if (!V[1][j])
          V[1][j] = a(i, j)->function_space(1);
        else
        {
          if (V[1][j] != a(i, j)->function_space(1))
            throw std::runtime_error("Mismatched trial space for column.");
        }
      }
    }
  }

  // Check there are no null entries
  for (std::size_t i = 0; i < V.size(); ++i)
    for (std::size_t j = 0; j < V[i].size(); ++j)
      if (!V[i][j])
        throw std::runtime_error("Could not deduce all block spaces.");

  return V;
}
//-----------------------------------------------------------------------------
la::SparsityPattern dolfinx::fem::create_sparsity_pattern(const Form& a)
{
  if (a.rank() != 2)
  {
    throw std::runtime_error(
        "Cannot create sparsity pattern. Form is not a bilinear form");
  }

  // Get dof maps
  std::array<const DofMap*, 2> dofmaps
      = {{a.function_space(0)->dofmap().get(),
          a.function_space(1)->dofmap().get()}};

  // Get mesh
  std::shared_ptr<const mesh::Mesh> mesh = a.mesh();
  assert(mesh);
  const int tdim = mesh->topology().dim();

  common::Timer t0("Build sparsity");

  // Get common::IndexMaps for each dimension
  std::array<std::shared_ptr<const common::IndexMap>, 2> index_maps
      = {{dofmaps[0]->index_map, dofmaps[1]->index_map}};

  // Create and build sparsity pattern
  la::SparsityPattern pattern(mesh->mpi_comm(), index_maps);
  if (a.integrals().num_integrals(fem::FormIntegrals::Type::cell) > 0)
  {
    SparsityPatternBuilder::cells(pattern, mesh->topology(),
                                  {{dofmaps[0], dofmaps[1]}});
  }

  if (a.integrals().num_integrals(fem::FormIntegrals::Type::interior_facet) > 0)
  {
    // FIXME: cleanup these calls? Some of the happen internally again.
    mesh->topology_mutable().create_entities(tdim - 1);
    mesh->topology_mutable().create_connectivity(tdim - 1, tdim);
    SparsityPatternBuilder::interior_facets(pattern, mesh->topology(),
                                            {{dofmaps[0], dofmaps[1]}});
  }

  if (a.integrals().num_integrals(fem::FormIntegrals::Type::exterior_facet) > 0)
  {
    // FIXME: cleanup these calls? Some of the happen internally again.
    mesh->topology_mutable().create_entities(tdim - 1);
    mesh->topology_mutable().create_connectivity(tdim - 1, tdim);
    SparsityPatternBuilder::exterior_facets(pattern, mesh->topology(),
                                            {{dofmaps[0], dofmaps[1]}});
  }
  t0.stop();

  return pattern;
}
//-----------------------------------------------------------------------------
la::PETScMatrix dolfinx::fem::create_matrix(const Form& a)
{
  // Build sparsitypattern
  la::SparsityPattern pattern = fem::create_sparsity_pattern(a);

  // Finalise communication
  pattern.assemble();

  // Initialize matrix
  common::Timer t1("Init tensor");
  la::PETScMatrix A(a.mesh()->mpi_comm(), pattern);
  t1.stop();

  return A;
}
//-----------------------------------------------------------------------------
la::PETScMatrix fem::create_matrix_block(
    const Eigen::Ref<const Eigen::Array<const fem::Form*, Eigen::Dynamic,
                                        Eigen::Dynamic, Eigen::RowMajor>>& a)
{
  // Extract and check row/column ranges
  std::array<std::vector<std::shared_ptr<const function::FunctionSpace>>, 2> V
      = block_function_spaces(a);

  std::shared_ptr<const mesh::Mesh> mesh = V[0][0]->mesh();
  assert(mesh);
  const int tdim = mesh->topology().dim();

  // Build sparsity pattern for each block
  std::vector<std::vector<std::unique_ptr<la::SparsityPattern>>> patterns(
      V[0].size());
  for (std::size_t row = 0; row < V[0].size(); ++row)
  {
    for (std::size_t col = 0; col < V[1].size(); ++col)
    {
      const std::array<std::shared_ptr<const common::IndexMap>, 2> index_maps
          = {{V[0][row]->dofmap()->index_map, V[1][col]->dofmap()->index_map}};
      if (a(row, col))
      {
        // Create sparsity pattern for block
        patterns[row].push_back(std::make_unique<la::SparsityPattern>(
            mesh->mpi_comm(), index_maps));

        // Build sparsity pattern for block
        std::array<const DofMap*, 2> dofmaps
            = {{V[0][row]->dofmap().get(), V[1][col]->dofmap().get()}};
        assert(patterns[row].back());
        auto& sp = patterns[row].back();
        assert(sp);
        const FormIntegrals& integrals = a(row, col)->integrals();
        if (integrals.num_integrals(FormIntegrals::Type::cell) > 0)
          SparsityPatternBuilder::cells(*sp, mesh->topology(), dofmaps);
        if (integrals.num_integrals(FormIntegrals::Type::interior_facet) > 0)
        {
          mesh->topology_mutable().create_entities(tdim - 1);
          SparsityPatternBuilder::interior_facets(*sp, mesh->topology(),
                                                  dofmaps);
        }
        if (integrals.num_integrals(FormIntegrals::Type::exterior_facet) > 0)
        {
          mesh->topology_mutable().create_entities(tdim - 1);
          SparsityPatternBuilder::exterior_facets(*sp, mesh->topology(),
                                                  dofmaps);
        }
      }
      else
        patterns[row].push_back(nullptr);
    }
  }

  // Compute offsets for the fields
  std::array<std::vector<std::reference_wrapper<const common::IndexMap>>, 2>
      maps;
  for (std::size_t d = 0; d < 2; ++d)
  {
    for (auto space : V[d])
      maps[d].push_back(*space->dofmap()->index_map.get());
  }


  // FIXME: This is computed again inside the SparsityPattern
  // constructor, but we also need to outside to build the PETSc
  // local-to-global map. Compute outside and pass into SparsityPattern
  // constructor.
  auto [rank_offset, local_offset, ghosts] = common::stack_index_maps(maps[0]);

  // Create merged sparsity pattern
  std::vector<std::vector<const la::SparsityPattern*>> p(V[0].size());
  for (std::size_t row = 0; row < V[0].size(); ++row)
    for (std::size_t col = 0; col < V[1].size(); ++col)
      p[row].push_back(patterns[row][col].get());
  la::SparsityPattern pattern(mesh->mpi_comm(), p, maps);
  pattern.assemble();


  // FIXME: Add option to pass customised local-to-global map to PETSc
  // Mat constructor.

  // Initialise matrix
  la::PETScMatrix A(mesh->mpi_comm(), pattern);

  // Create row and column local-to-global maps (field0, field1, field2,
  // etc), i.e. ghosts of field0 appear before owned indices of field1
  std::array<std::vector<PetscInt>, 2> _maps;
  for (int d = 0; d < 2; ++d)
  {
    for (std::size_t f = 0; f < maps[d].size(); ++f)
    {
      const common::IndexMap& map = maps[d][f].get();
      const int bs = map.block_size();
      const std::int32_t size_local = bs * map.size_local();
      const std::vector<std::int64_t> global = map.global_indices(false);
      for (std::int32_t i = 0; i < size_local; ++i)
        _maps[d].push_back(i + rank_offset + local_offset[f]);
      for (std::size_t i = size_local; i < global.size(); ++i)
        _maps[d].push_back(ghosts[f][i - size_local]);
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
la::PETScMatrix fem::create_matrix_nest(
    const Eigen::Ref<const Eigen::Array<const fem::Form*, Eigen::Dynamic,
                                        Eigen::Dynamic, Eigen::RowMajor>>& a)
{
  // Extract and check row/column ranges
  std::array<std::vector<std::shared_ptr<const function::FunctionSpace>>, 2> V
      = block_function_spaces(a);

  // Loop over each form and create matrix
  Eigen::Array<std::shared_ptr<la::PETScMatrix>, Eigen::Dynamic, Eigen::Dynamic,
               Eigen::RowMajor>
      mats(a.rows(), a.cols());
  Eigen::Array<Mat, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> petsc_mats(
      a.rows(), a.cols());
  for (int i = 0; i < a.rows(); ++i)
  {
    for (int j = 0; j < a.cols(); ++j)
    {
      if (a(i, j))
      {
        mats(i, j) = std::make_shared<la::PETScMatrix>(create_matrix(*a(i, j)));
        petsc_mats(i, j) = mats(i, j)->mat();
      }
      else
        petsc_mats(i, j) = nullptr;
    }
  }

  // Initialise block (MatNest) matrix
  Mat _A;
  MatCreate(V[0][0]->mesh()->mpi_comm(), &_A);
  MatSetType(_A, MATNEST);
  MatNestSetSubMats(_A, petsc_mats.rows(), nullptr, petsc_mats.cols(), nullptr,
                    petsc_mats.data());
  MatSetUp(_A);

  return la::PETScMatrix(_A);
}
//-----------------------------------------------------------------------------
la::PETScVector fem::create_vector_block(
    const std::vector<std::reference_wrapper<const common::IndexMap>>& maps)
{
  // FIXME: handle constant block size > 1

  auto [rank_offset, local_offset, ghosts_new] = common::stack_index_maps(maps);
  std::int32_t local_size = local_offset.back();
  std::vector<std::int64_t> ghosts;
  for (auto& sub_ghost : ghosts_new)
    ghosts.insert(ghosts.end(), sub_ghost.begin(), sub_ghost.end());

  // Create map for combined problem, and create vector
  Eigen::Map<const Eigen::Array<std::int64_t, Eigen::Dynamic, 1>> _ghosts(
      ghosts.data(), ghosts.size());
  common::IndexMap index_map(maps[0].get().mpi_comm(), local_size, _ghosts, 1);

  return la::PETScVector(index_map);
}
//-----------------------------------------------------------------------------
la::PETScVector
fem::create_vector_nest(const std::vector<const common::IndexMap*>& maps)
{
  assert(!maps.empty());

  // Loop over each form and create vector
  std::vector<std::shared_ptr<la::PETScVector>> vecs(maps.size());
  std::vector<Vec> petsc_vecs(maps.size());
  for (std::size_t i = 0; i < maps.size(); ++i)
  {
    if (!maps[i])
    {
      throw std::runtime_error(
          "Cannot construct nested PETSc vectors with null blocks.");
    }
    vecs[i] = std::make_shared<la::PETScVector>(*maps[i]);
    petsc_vecs[i] = vecs[i]->vec();
  }

  // Create nested (VecNest) vector
  Vec y;
  VecCreateNest(vecs[0]->mpi_comm(), petsc_vecs.size(), nullptr,
                petsc_vecs.data(), &y);
  return la::PETScVector(y, false);
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

  int dof_count = 0;

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
      dof_count += num_entity_dofs[dim];
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
        = std::shared_ptr<ufc_dofmap>(dofmap.create_sub_dofmap(i), std::free);
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

  const int num_base_permutations = get_num_permutations(cell_type);
  const Eigen::Map<
      const Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      base_permutations(dofmap.base_permutations, num_base_permutations,
                        dof_count);
  return fem::ElementDofLayout(block_size, entity_dofs, parent_map, sub_dofmaps,
                               cell_type, base_permutations);
}
//-----------------------------------------------------------------------------
fem::DofMap fem::create_dofmap(MPI_Comm comm, const ufc_dofmap& ufc_dofmap,
                               mesh::Topology& topology)
{
  auto element_dof_layout = std::make_shared<ElementDofLayout>(
      create_element_dof_layout(ufc_dofmap, topology.cell_type()));
  assert(element_dof_layout);

  // Create required mesh entities
  const int D = topology.dim();
  for (int d = 0; d < D; ++d)
  {
    if (element_dof_layout->num_entity_dofs(d) > 0)
    {
      // Create local entities
      const auto [cell_entity, entity_vertex, index_map]
          = mesh::TopologyComputation::compute_entities(comm, topology, d);
      if (cell_entity)
        topology.set_connectivity(cell_entity, topology.dim(), d);
      if (entity_vertex)
        topology.set_connectivity(entity_vertex, d, 0);
      if (index_map)
        topology.set_index_map(d, index_map);
    }
  }

  auto [dof_layout, index_map, dofmap]
      = DofMapBuilder::build(comm, topology, element_dof_layout);
  return DofMap(dof_layout, index_map, std::move(dofmap));
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
    coeffs.emplace_back(ufc_form.original_coefficient_position(i), names[i],
                        nullptr);
  }
  return coeffs;
}
//-----------------------------------------------------------------------------
std::vector<std::pair<std::string, std::shared_ptr<const function::Constant>>>
fem::get_constants_from_ufc_form(const ufc_form& ufc_form)
{
  std::vector<std::pair<std::string, std::shared_ptr<const function::Constant>>>
      constants;
  const char** names = ufc_form.constant_name_map();
  for (int i = 0; i < ufc_form.num_constants; ++i)
    constants.emplace_back(names[i], nullptr);
  return constants;
}
//-----------------------------------------------------------------------------
std::shared_ptr<fem::Form> fem::create_form(
    ufc_form* (*fptr)(),
    const std::vector<std::shared_ptr<const function::FunctionSpace>>& spaces)
{
  ufc_form* form = fptr();
  auto L
      = std::make_shared<fem::Form>(dolfinx::fem::create_form(*form, spaces));
  std::free(form);

  return L;
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
    assert(spaces[i]->element());
    std::unique_ptr<ufc_finite_element, decltype(free)*> ufc_element(
        ufc_form.create_finite_element(i), free);
    assert(ufc_element);
    if (std::string(ufc_element->signature)
        != spaces[i]->element()->signature())
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
    integrals.set_tabulate_tensor(FormIntegrals::Type::cell, id,
                                  cell_integral->tabulate_tensor);
    std::free(cell_integral);
  }

  // FIXME: Can this be handled better?
  // FIXME: Handle forms with no space
  if (ufc_form.num_exterior_facet_integrals > 0
      or ufc_form.num_interior_facet_integrals > 0)
  {
    if (!spaces.empty())
    {
      auto mesh = spaces[0]->mesh();
      const int tdim = mesh->topology().dim();
      spaces[0]->mesh()->topology_mutable().create_entities(tdim - 1);
    }
  }

  std::vector<int> exterior_facet_integral_ids(
      ufc_form.num_exterior_facet_integrals);
  ufc_form.get_exterior_facet_integral_ids(exterior_facet_integral_ids.data());
  for (int id : exterior_facet_integral_ids)
  {
    ufc_integral* exterior_facet_integral
        = ufc_form.create_exterior_facet_integral(id);
    assert(exterior_facet_integral);
    integrals.set_tabulate_tensor(FormIntegrals::Type::exterior_facet, id,
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
    integrals.set_tabulate_tensor(FormIntegrals::Type::interior_facet, id,
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

  return fem::Form(spaces, integrals,
                   FormCoefficients(fem::get_coeffs_from_ufc_form(ufc_form)),
                   fem::get_constants_from_ufc_form(ufc_form));
}
//-----------------------------------------------------------------------------
fem::CoordinateElement
fem::create_coordinate_map(const ufc_coordinate_mapping& ufc_cmap)
{
  static const std::map<ufc_shape, mesh::CellType> ufc_to_cell
      = {{vertex, mesh::CellType::point},
         {interval, mesh::CellType::interval},
         {triangle, mesh::CellType::triangle},
         {tetrahedron, mesh::CellType::tetrahedron},
         {quadrilateral, mesh::CellType::quadrilateral},
         {hexahedron, mesh::CellType::hexahedron}};

  // Get cell type
  const mesh::CellType cell_type = ufc_to_cell.at(ufc_cmap.cell_shape);
  assert(ufc_cmap.topological_dimension == mesh::cell_dim(cell_type));

  // Get scalar dof layout for geometry
  ufc_dofmap* dmap = ufc_cmap.create_scalar_dofmap();
  assert(dmap);
  ElementDofLayout dof_layout = create_element_dof_layout(*dmap, cell_type);
  std::free(dmap);

  return fem::CoordinateElement(
      cell_type, ufc_cmap.topological_dimension, ufc_cmap.geometric_dimension,
      ufc_cmap.signature, dof_layout, ufc_cmap.compute_physical_coordinates,
      ufc_cmap.compute_reference_geometry);
}
//-----------------------------------------------------------------------------
fem::CoordinateElement
fem::create_coordinate_map(ufc_coordinate_mapping* (*fptr)())
{
  ufc_coordinate_mapping* cmap = fptr();
  fem::CoordinateElement element = create_coordinate_map(*cmap);
  std::free(cmap);
  return element;
}
//-----------------------------------------------------------------------------
std::shared_ptr<function::FunctionSpace>
fem::create_functionspace(ufc_function_space* (*fptr)(const char*),
                          const std::string function_name,
                          std::shared_ptr<mesh::Mesh> mesh)
{
  ufc_function_space* space = fptr(function_name.c_str());
  ufc_dofmap* ufc_map = space->create_dofmap();
  ufc_finite_element* ufc_element = space->create_element();
  std::shared_ptr<function::FunctionSpace> V
      = std::make_shared<function::FunctionSpace>(
          mesh, std::make_shared<fem::FiniteElement>(*ufc_element),
          std::make_shared<fem::DofMap>(fem::create_dofmap(
              mesh->mpi_comm(), *ufc_map, mesh->topology())));
  std::free(ufc_element);
  std::free(ufc_map);
  std::free(space);
  return V;
}
//-----------------------------------------------------------------------------
Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
fem::pack_coefficients(const fem::Form& form)
{
  // Get form coefficient offsets amd dofmaps
  const fem::FormCoefficients& coefficients = form.coefficients();
  const std::vector<int>& offsets = coefficients.offsets();
  std::vector<const fem::DofMap*> dofmaps(coefficients.size());
  for (int i = 0; i < coefficients.size(); ++i)
    dofmaps[i] = coefficients.get(i)->function_space()->dofmap().get();

  // Get mesh
  std::shared_ptr<const mesh::Mesh> mesh = form.mesh();
  assert(mesh);
  const int tdim = mesh->topology().dim();

  // Unwrap PETSc vectors
  std::vector<const PetscScalar*> v(coefficients.size(), nullptr);
  std::vector<Vec> x(coefficients.size(), nullptr),
      x_local(coefficients.size(), nullptr);
  for (std::size_t i = 0; i < v.size(); ++i)
  {
    x[i] = coefficients.get(i)->vector().vec();
    VecGhostGetLocalForm(x[i], &x_local[i]);
    VecGetArrayRead(x_local[i], &v[i]);
  }

  const int num_cells = mesh->topology().index_map(tdim)->size_local()
                        + mesh->topology().index_map(tdim)->num_ghosts();

  // Copy data into coefficient array
  Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> c(
      num_cells, offsets.back());
  if (coefficients.size() > 0)
  {
    for (int cell = 0; cell < num_cells; ++cell)
    {
      for (std::size_t coeff = 0; coeff < dofmaps.size(); ++coeff)
      {
        auto dofs = dofmaps[coeff]->cell_dofs(cell);
        const PetscScalar* _v = v[coeff];
        for (Eigen::Index k = 0; k < dofs.size(); ++k)
          c(cell, k + offsets[coeff]) = _v[dofs[k]];
      }
    }
  }

  // Restore PETSc vectors
  for (std::size_t i = 0; i < v.size(); ++i)
  {
    VecRestoreArrayRead(x_local[i], &v[i]);
    VecGhostRestoreLocalForm(x[i], &x_local[i]);
  }

  return c;
}
//-----------------------------------------------------------------------------
Eigen::Array<PetscScalar, Eigen::Dynamic, 1>
fem::pack_constants(const fem::Form& form)
{
  const std::vector<
      std::pair<std::string, std::shared_ptr<const function::Constant>>>
      constants = form.constants();
  std::vector<PetscScalar> constant_values;
  for (const auto& constant : constants)
  {
    const std::vector<PetscScalar>& array = constant.second->value;
    constant_values.insert(constant_values.end(), array.data(),
                           array.data() + array.size());
  }

  return Eigen::Map<const Eigen::Array<PetscScalar, Eigen::Dynamic, 1>>(
      constant_values.data(), constant_values.size(), 1);
}
//-----------------------------------------------------------------------------
