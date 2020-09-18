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
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/SparsityPatternBuilder.h>
#include <dolfinx/function/Constant.h>
#include <dolfinx/function/Function.h>
#include <dolfinx/function/FunctionSpace.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <dolfinx/mesh/TopologyComputation.h>
#include <memory>
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
} // namespace

//-----------------------------------------------------------------------------
la::SparsityPattern
fem::create_sparsity_pattern(const mesh::Topology& topology,
                             const std::array<const DofMap*, 2>& dofmaps,
                             const std::set<IntegralType>& integrals)
{
  common::Timer t0("Build sparsity");

  // Get common::IndexMaps for each dimension
  std::array index_maps{dofmaps[0]->index_map, dofmaps[1]->index_map};

  // Create and build sparsity pattern
  assert(dofmaps[0]);
  assert(dofmaps[0]->index_map);
  la::SparsityPattern pattern(dofmaps[0]->index_map->comm(), index_maps);
  for (auto type : integrals)
  {
    if (type == fem::IntegralType::cell)
    {
      SparsityPatternBuilder::cells(pattern, topology,
                                    {{dofmaps[0], dofmaps[1]}});
    }
    else if (type == fem::IntegralType::interior_facet)
    {
      SparsityPatternBuilder::interior_facets(pattern, topology,
                                              {{dofmaps[0], dofmaps[1]}});
    }
    else if (type == fem::IntegralType::exterior_facet)
    {
      SparsityPatternBuilder::exterior_facets(pattern, topology,
                                              {{dofmaps[0], dofmaps[1]}});
    }
  }

  t0.stop();

  return pattern;
}
//-----------------------------------------------------------------------------
fem::ElementDofLayout
fem::create_element_dof_layout(const ufc_dofmap& dofmap,
                               const mesh::CellType cell_type,
                               const std::vector<int>& parent_map)
{
  const int element_block_size = dofmap.block_size;

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
    if (element_block_size == 1)
    {
      offsets.push_back(offsets.back()
                        + ufc_sub_dofmap->num_element_support_dofs
                              * ufc_sub_dofmap->block_size);
    }
    else
      offsets.push_back(offsets.back() + 1);
  }

  std::vector<std::shared_ptr<const fem::ElementDofLayout>> sub_dofmaps;
  for (std::size_t i = 0; i < ufc_sub_dofmaps.size(); ++i)
  {
    auto ufc_sub_dofmap = ufc_sub_dofmaps[i];
    assert(ufc_sub_dofmap);
    std::vector<int> parent_map_sub(ufc_sub_dofmap->num_element_support_dofs);
    for (std::size_t j = 0; j < parent_map_sub.size(); ++j)
      parent_map_sub[j] = offsets[i] + j * element_block_size;
    sub_dofmaps.push_back(
        std::make_shared<fem::ElementDofLayout>(create_element_dof_layout(
            *ufc_sub_dofmaps[i], cell_type, parent_map_sub)));
  }

  // Check for "block structure". This should ultimately be replaced,
  // but keep for now to mimic existing code
  const int num_base_permutations = get_num_permutations(cell_type);
  const Eigen::Map<
      const Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      base_permutations(dofmap.base_permutations, num_base_permutations,
                        dof_count);
  return fem::ElementDofLayout(element_block_size, entity_dofs, parent_map,
                               sub_dofmaps, cell_type, base_permutations);
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

  auto [index_map, dofmap]
      = DofMapBuilder::build(comm, topology, *element_dof_layout);
  return DofMap(element_dof_layout, index_map, std::move(dofmap));
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
      ufc_cmap.signature, dof_layout, ufc_cmap.is_affine,
      ufc_cmap.evaluate_basis_derivatives);
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
  auto V = std::make_shared<function::FunctionSpace>(
      mesh, std::make_shared<fem::FiniteElement>(*ufc_element),
      std::make_shared<fem::DofMap>(
          fem::create_dofmap(mesh->mpi_comm(), *ufc_map, mesh->topology())));
  std::free(ufc_element);
  std::free(ufc_map);
  std::free(space);
  return V;
}
//-----------------------------------------------------------------------------
