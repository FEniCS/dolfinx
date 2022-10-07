// Copyright (C) 2013-2019 Johan Hake, Jan Blechta and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "utils.h"
#include "Constant.h"
#include "DofMap.h"
#include "FiniteElement.h"
#include "Form.h"
#include "Function.h"
#include "FunctionSpace.h"
#include "dofmapbuilder.h"
#include <array>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/log.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <dolfinx/mesh/topologycomputation.h>
#include <memory>
#include <string>
#include <ufcx.h>

using namespace dolfinx;

//-----------------------------------------------------------------------------
la::SparsityPattern fem::create_sparsity_pattern(
    const mesh::Topology& topology,
    const std::array<std::reference_wrapper<const DofMap>, 2>& dofmaps,
    const std::set<IntegralType>& integrals)
{
  common::Timer t0("Build sparsity");

  // Get common::IndexMaps for each dimension
  const std::array index_maps{dofmaps[0].get().index_map,
                              dofmaps[1].get().index_map};
  const std::array bs
      = {dofmaps[0].get().index_map_bs(), dofmaps[1].get().index_map_bs()};

  // Create and build sparsity pattern
  assert(dofmaps[0].get().index_map);
  la::SparsityPattern pattern(dofmaps[0].get().index_map->comm(), index_maps,
                              bs);
  for (auto type : integrals)
  {
    switch (type)
    {
    case IntegralType::cell:
      sparsitybuild::cells(pattern, topology, {{dofmaps[0], dofmaps[1]}});
      break;
    case IntegralType::interior_facet:
      sparsitybuild::interior_facets(pattern, topology,
                                     {{dofmaps[0], dofmaps[1]}});
      break;
    case IntegralType::exterior_facet:
      sparsitybuild::exterior_facets(pattern, topology,
                                     {{dofmaps[0], dofmaps[1]}});
      break;
    default:
      throw std::runtime_error("Unsupported integral type");
    }
  }

  t0.stop();

  return pattern;
}
//-----------------------------------------------------------------------------
fem::ElementDofLayout
fem::create_element_dof_layout(const ufcx_dofmap& dofmap,
                               const mesh::CellType cell_type,
                               const std::vector<int>& parent_map)
{
  const int element_block_size = dofmap.block_size;

  // Copy over number of dofs per entity type
  std::array<int, 4> num_entity_dofs, num_entity_closure_dofs;
  std::copy_n(dofmap.num_entity_dofs, 4, num_entity_dofs.begin());
  std::copy_n(dofmap.num_entity_closure_dofs, 4,
              num_entity_closure_dofs.begin());

  // Fill entity dof indices
  const int tdim = mesh::cell_dim(cell_type);
  std::vector<std::vector<std::vector<int>>> entity_dofs(tdim + 1);
  std::vector<std::vector<std::vector<int>>> entity_closure_dofs(tdim + 1);
  for (int dim = 0; dim <= tdim; ++dim)
  {
    const int num_entities = mesh::cell_num_entities(cell_type, dim);
    entity_dofs[dim].resize(num_entities);
    entity_closure_dofs[dim].resize(num_entities);
    for (int i = 0; i < num_entities; ++i)
    {
      entity_dofs[dim][i].resize(num_entity_dofs[dim]);
      dofmap.tabulate_entity_dofs(entity_dofs[dim][i].data(), dim, i);

      entity_closure_dofs[dim][i].resize(num_entity_closure_dofs[dim]);
      dofmap.tabulate_entity_closure_dofs(entity_closure_dofs[dim][i].data(),
                                          dim, i);
    }
  }

  // TODO: UFC dofmaps just use simple offset for each field but this
  // could be different for custom dofmaps. This data should come
  // directly from the UFC interface in place of the implicit
  // assumption.

  // Create UFC subdofmaps and compute offset
  std::vector<int> offsets(1, 0);
  std::vector<ElementDofLayout> sub_doflayout;
  for (int i = 0; i < dofmap.num_sub_dofmaps; ++i)
  {
    ufcx_dofmap* ufcx_sub_dofmap = dofmap.sub_dofmaps[i];
    if (element_block_size == 1)
    {
      offsets.push_back(offsets.back()
                        + ufcx_sub_dofmap->num_element_support_dofs
                              * ufcx_sub_dofmap->block_size);
    }
    else
      offsets.push_back(offsets.back() + 1);

    std::vector<int> parent_map_sub(ufcx_sub_dofmap->num_element_support_dofs
                                    * ufcx_sub_dofmap->block_size);
    for (std::size_t j = 0; j < parent_map_sub.size(); ++j)
      parent_map_sub[j] = offsets[i] + element_block_size * j;
    sub_doflayout.push_back(
        create_element_dof_layout(*ufcx_sub_dofmap, cell_type, parent_map_sub));
  }

  // Check for "block structure". This should ultimately be replaced,
  // but keep for now to mimic existing code
  return ElementDofLayout(element_block_size, entity_dofs, entity_closure_dofs,
                          parent_map, sub_doflayout);
}
//-----------------------------------------------------------------------------
fem::DofMap
fem::create_dofmap(MPI_Comm comm, const ElementDofLayout& layout,
                   mesh::Topology& topology,
                   const std::function<std::vector<int>(
                       const graph::AdjacencyList<std::int32_t>&)>& reorder_fn,
                   const FiniteElement& element)
{
  // Create required mesh entities
  const int D = topology.dim();
  for (int d = 0; d < D; ++d)
  {
    if (layout.num_entity_dofs(d) > 0)
      topology.create_entities(d);
  }

  auto [_index_map, bs, dofmap]
      = build_dofmap_data(comm, topology, layout, reorder_fn);
  auto index_map = std::make_shared<common::IndexMap>(std::move(_index_map));

  // If the element's DOF transformations are permutations, permute the
  // DOF numbering on each cell
  if (element.needs_dof_permutations())
  {
    const int D = topology.dim();
    const int num_cells = topology.connectivity(D, 0)->num_nodes();
    topology.create_entity_permutations();
    const std::vector<std::uint32_t>& cell_info
        = topology.get_cell_permutation_info();

    const std::function<void(const std::span<std::int32_t>&, std::uint32_t)>
        unpermute_dofs = element.get_dof_permutation_function(true, true);
    for (std::int32_t cell = 0; cell < num_cells; ++cell)
      unpermute_dofs(dofmap.links(cell), cell_info[cell]);
  }

  return DofMap(layout, index_map, bs, std::move(dofmap), bs);
}
//-----------------------------------------------------------------------------
std::vector<std::string> fem::get_coefficient_names(const ufcx_form& ufcx_form)
{
  std::vector<std::string> coefficients;
  const char** names = ufcx_form.coefficient_name_map();
  for (int i = 0; i < ufcx_form.num_coefficients; ++i)
    coefficients.push_back(names[i]);
  return coefficients;
}
//-----------------------------------------------------------------------------
std::vector<std::string> fem::get_constant_names(const ufcx_form& ufcx_form)
{
  std::vector<std::string> constants;
  const char** names = ufcx_form.constant_name_map();
  for (int i = 0; i < ufcx_form.num_constants; ++i)
    constants.push_back(names[i]);
  return constants;
}
//-----------------------------------------------------------------------------
fem::FunctionSpace fem::create_functionspace(
    std::shared_ptr<mesh::Mesh> mesh, const basix::FiniteElement& e, int bs,
    const std::function<std::vector<int>(
        const graph::AdjacencyList<std::int32_t>&)>& reorder_fn)
{
  assert(mesh);

  // Create a DOLFINx element
  auto _e = std::make_shared<FiniteElement>(e, bs);

  // Create UFC subdofmaps and compute offset
  assert(_e);
  const int num_sub_elements = _e->num_sub_elements();
  std::vector<ElementDofLayout> sub_doflayout;
  sub_doflayout.reserve(num_sub_elements);
  for (int i = 0; i < num_sub_elements; ++i)
  {
    auto sub_element = _e->extract_sub_element({i});
    std::vector<int> parent_map_sub(sub_element->space_dimension());
    for (std::size_t j = 0; j < parent_map_sub.size(); ++j)
      parent_map_sub[j] = i + bs * j;
    sub_doflayout.emplace_back(1, e.entity_dofs(), e.entity_closure_dofs(),
                               parent_map_sub, std::vector<ElementDofLayout>());
  }

  // Create a dofmap
  ElementDofLayout layout(bs, e.entity_dofs(), e.entity_closure_dofs(), {},
                          sub_doflayout);
  auto dofmap = std::make_shared<const DofMap>(
      create_dofmap(mesh->comm(), layout, mesh->topology(), reorder_fn, *_e));

  return FunctionSpace(mesh, _e, dofmap);
}
//-----------------------------------------------------------------------------
fem::FunctionSpace fem::create_functionspace(
    ufcx_function_space* (*fptr)(const char*), const std::string& function_name,
    std::shared_ptr<mesh::Mesh> mesh,
    const std::function<std::vector<int>(
        const graph::AdjacencyList<std::int32_t>&)>& reorder_fn)
{
  ufcx_function_space* space = fptr(function_name.c_str());
  if (!space)
  {
    throw std::runtime_error(
        "Could not create UFC function space with function name "
        + function_name);
  }

  ufcx_finite_element* ufcx_element = space->finite_element;
  assert(ufcx_element);

  if (space->geometry_degree != mesh->geometry().cmap().degree()
      or static_cast<basix::cell::type>(space->geometry_basix_cell)
             != mesh::cell_type_to_basix_type(
                 mesh->geometry().cmap().cell_shape())
      or static_cast<basix::element::lagrange_variant>(
             space->geometry_basix_variant)
             != mesh->geometry().cmap().variant())
  {
    throw std::runtime_error("UFL mesh and CoordinateElement do not match.");
  }

  auto element = std::make_shared<FiniteElement>(*ufcx_element);
  assert(element);
  ufcx_dofmap* ufcx_map = space->dofmap;
  assert(ufcx_map);
  ElementDofLayout layout
      = create_element_dof_layout(*ufcx_map, mesh->topology().cell_type());
  return FunctionSpace(
      mesh, element,
      std::make_shared<DofMap>(create_dofmap(
          mesh->comm(), layout, mesh->topology(), reorder_fn, *element)));
}
//-----------------------------------------------------------------------------
