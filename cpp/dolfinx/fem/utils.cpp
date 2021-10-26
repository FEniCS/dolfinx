// Copyright (C) 2013-2019 Johan Hake, Jan Blechta and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "utils.h"
#include <array>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/log.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/dofmapbuilder.h>
#include <dolfinx/fem/sparsitybuild.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <dolfinx/mesh/topologycomputation.h>
#include <memory>
#include <string>
#include <ufc.h>

using namespace dolfinx;

//-----------------------------------------------------------------------------
la::SparsityPattern fem::create_sparsity_pattern(
    const mesh::Topology& topology,
    const std::array<std::reference_wrapper<const fem::DofMap>, 2>& dofmaps,
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
    case fem::IntegralType::cell:
      sparsitybuild::cells(pattern, topology, {{dofmaps[0], dofmaps[1]}});
      break;
    case fem::IntegralType::interior_facet:
      sparsitybuild::interior_facets(pattern, topology,
                                     {{dofmaps[0], dofmaps[1]}});
      break;
    case fem::IntegralType::exterior_facet:
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
fem::create_element_dof_layout(const ufc_dofmap& dofmap,
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
  // could be different for custom dofmaps This data should come
  // directly from the UFC interface in place of the the implicit
  // assumption

  // Create UFC subdofmaps and compute offset
  std::vector<int> offsets(1, 0);
  std::vector<std::shared_ptr<const fem::ElementDofLayout>> sub_dofmaps;
  for (int i = 0; i < dofmap.num_sub_dofmaps; ++i)
  {
    ufc_dofmap* ufc_sub_dofmap = dofmap.sub_dofmaps[i];
    if (element_block_size == 1)
    {
      offsets.push_back(offsets.back()
                        + ufc_sub_dofmap->num_element_support_dofs
                              * ufc_sub_dofmap->block_size);
    }
    else
      offsets.push_back(offsets.back() + 1);

    std::vector<int> parent_map_sub(ufc_sub_dofmap->num_element_support_dofs
                                    * ufc_sub_dofmap->block_size);
    for (std::size_t j = 0; j < parent_map_sub.size(); ++j)
      parent_map_sub[j] = offsets[i] + element_block_size * j;
    sub_dofmaps.push_back(std::make_shared<fem::ElementDofLayout>(
        create_element_dof_layout(*ufc_sub_dofmap, cell_type, parent_map_sub)));
  }

  // Check for "block structure". This should ultimately be replaced,
  // but keep for now to mimic existing code
  return fem::ElementDofLayout(element_block_size, entity_dofs,
                               entity_closure_dofs, parent_map, sub_dofmaps);
}
//-----------------------------------------------------------------------------
fem::DofMap
fem::create_dofmap(MPI_Comm comm, const ufc_dofmap& ufc_dofmap,
                   mesh::Topology& topology,
                   const std::function<std::vector<int>(
                       const graph::AdjacencyList<std::int32_t>&)>& reorder_fn,
                   std::shared_ptr<const dolfinx::fem::FiniteElement> element)
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
          = mesh::compute_entities(comm, topology, d);
      if (cell_entity)
        topology.set_connectivity(cell_entity, topology.dim(), d);
      if (entity_vertex)
        topology.set_connectivity(entity_vertex, d, 0);
      if (index_map)
        topology.set_index_map(d, index_map);
    }
  }

  auto [_index_map, bs, dofmap]
      = fem::build_dofmap_data(comm, topology, *element_dof_layout, reorder_fn);
  auto index_map = std::make_shared<common::IndexMap>(std::move(_index_map));

  // If the element's DOF transformations are permutations, permute the
  // DOF numbering on each cell
  if (element->needs_dof_permutations())
  {
    const int D = topology.dim();
    const int num_cells = topology.connectivity(D, 0)->num_nodes();
    topology.create_entity_permutations();
    const std::vector<std::uint32_t>& cell_info
        = topology.get_cell_permutation_info();

    const std::function<void(const xtl::span<std::int32_t>&, std::uint32_t)>
        unpermute_dofs = element->get_dof_permutation_function(true, true);
    for (std::int32_t cell = 0; cell < num_cells; ++cell)
      unpermute_dofs(dofmap.links(cell), cell_info[cell]);
  }

  return DofMap(element_dof_layout, index_map, bs, std::move(dofmap), bs);
}
//-----------------------------------------------------------------------------
std::vector<std::string> fem::get_coefficient_names(const ufc_form& ufc_form)
{
  std::vector<std::string> coefficients;
  const char** names = ufc_form.coefficient_name_map();
  for (int i = 0; i < ufc_form.num_coefficients; ++i)
    coefficients.push_back(names[i]);
  return coefficients;
}
//-----------------------------------------------------------------------------
std::vector<std::string> fem::get_constant_names(const ufc_form& ufc_form)
{
  std::vector<std::string> constants;
  const char** names = ufc_form.constant_name_map();
  for (int i = 0; i < ufc_form.num_constants; ++i)
    constants.push_back(names[i]);
  return constants;
}
//-----------------------------------------------------------------------------
fem::FunctionSpace fem::create_functionspace(
    ufc_function_space* (*fptr)(const char*), const std::string& function_name,
    std::shared_ptr<mesh::Mesh> mesh,
    const std::function<std::vector<int>(
        const graph::AdjacencyList<std::int32_t>&)>& reorder_fn)
{
  ufc_function_space* space = fptr(function_name.c_str());
  if (!space)
  {
    throw std::runtime_error(
        "Could not create UFC function space with function name "
        + function_name);
  }

  ufc_finite_element* ufc_element = space->finite_element;
  assert(ufc_element);
  std::shared_ptr<const fem::FiniteElement> element
      = std::make_shared<fem::FiniteElement>(*ufc_element);

  ufc_dofmap* ufc_map = space->dofmap;
  assert(ufc_map);
  return fem::FunctionSpace(
      mesh, element,
      std::make_shared<fem::DofMap>(fem::create_dofmap(
          mesh->mpi_comm(), *ufc_map, mesh->topology(), reorder_fn, element)));
}
//-----------------------------------------------------------------------------
