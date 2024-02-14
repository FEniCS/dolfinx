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
#include <algorithm>
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
fem::ElementDofLayout
fem::create_element_dof_layout(const ufcx_dofmap& dofmap,
                               const mesh::CellType cell_type,
                               const std::vector<int>& parent_map)
{
  const int element_block_size = dofmap.block_size;

  // Fill entity dof indices
  const int tdim = mesh::cell_dim(cell_type);
  std::vector<std::vector<std::vector<int>>> entity_dofs(tdim + 1);
  std::vector<std::vector<std::vector<int>>> entity_closure_dofs(tdim + 1);
  {
    int* offset0 = dofmap.entity_dof_offsets;
    int* offset1 = dofmap.entity_closure_dof_offsets;
    for (int d = 0; d <= tdim; ++d)
    {
      int num_entities = mesh::cell_num_entities(cell_type, d);
      entity_dofs[d].resize(num_entities);
      entity_closure_dofs[d].resize(num_entities);
      for (int i = 0; i < num_entities; ++i)
      {
        std::copy(dofmap.entity_dofs + *offset0,
                  dofmap.entity_dofs + *(offset0 + 1),
                  std::back_inserter(entity_dofs[d][i]));
        std::copy(dofmap.entity_closure_dofs + *offset1,
                  dofmap.entity_closure_dofs + *(offset1 + 1),
                  std::back_inserter(entity_closure_dofs[d][i]));
        ++offset0;
        ++offset1;
      }
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
  // but keep for now to mimic existing code.
  return ElementDofLayout(element_block_size, entity_dofs, entity_closure_dofs,
                          parent_map, sub_doflayout);
}
//-----------------------------------------------------------------------------
fem::DofMap fem::create_dofmap(
    MPI_Comm comm, const ElementDofLayout& layout, mesh::Topology& topology,
    std::function<void(const std::span<std::int32_t>&, std::uint32_t)>
        unpermute_dofs,
    std::function<std::vector<int>(const graph::AdjacencyList<std::int32_t>&)>
        reorder_fn)
{
  // Create required mesh entities
  const int D = topology.dim();
  for (int d = 0; d < D; ++d)
  {
    if (layout.num_entity_dofs(d) > 0)
      topology.create_entities(d);
  }

  auto [_index_map, bs, dofmaps]
      = build_dofmap_data(comm, topology, {layout}, reorder_fn);
  auto index_map = std::make_shared<common::IndexMap>(std::move(_index_map));

  // If the element's DOF transformations are permutations, permute the
  // DOF numbering on each cell
  if (unpermute_dofs)
  {
    const int D = topology.dim();
    const int num_cells = topology.connectivity(D, 0)->num_nodes();
    topology.create_entity_permutations();
    const std::vector<std::uint32_t>& cell_info
        = topology.get_cell_permutation_info();
    int dim = layout.num_dofs();
    for (std::int32_t cell = 0; cell < num_cells; ++cell)
    {
      std::span<std::int32_t> dofs(dofmaps.front().data() + cell * dim, dim);
      unpermute_dofs(dofs, cell_info[cell]);
    }
  }

  return DofMap(layout, index_map, bs, std::move(dofmaps.front()), bs);
}
//-----------------------------------------------------------------------------
std::vector<fem::DofMap> fem::create_dofmaps(
    MPI_Comm comm, const std::vector<ElementDofLayout>& layouts,
    mesh::Topology& topology,
    std::function<void(const std::span<std::int32_t>&, std::uint32_t)>
        unpermute_dofs,
    std::function<std::vector<int>(const graph::AdjacencyList<std::int32_t>&)>
        reorder_fn)
{
  // Create required mesh entities
  const int D = topology.dim();
  for (int d = 0; d < D; ++d)
  {
    if (layouts[0].num_entity_dofs(d) > 0)
      topology.create_entities(d);
  }

  auto [_index_map, bs, dofmaps]
      = build_dofmap_data(comm, topology, layouts, reorder_fn);
  auto index_map = std::make_shared<common::IndexMap>(std::move(_index_map));

  // If the element's DOF transformations are permutations, permute the
  // DOF numbering on each cell
  if (unpermute_dofs)
  {
    const int D = topology.dim();
    const int num_cells = topology.connectivity(D, 0)->num_nodes();
    topology.create_entity_permutations();
    const std::vector<std::uint32_t>& cell_info
        = topology.get_cell_permutation_info();
    int dim = layouts[0].num_dofs();
    for (std::int32_t cell = 0; cell < num_cells; ++cell)
    {
      std::span<std::int32_t> dofs(dofmaps[0].data() + cell * dim, dim);
      unpermute_dofs(dofs, cell_info[cell]);
    }
  }

  std::vector<DofMap> dms;
  for (std::size_t i = 0; i < dofmaps.size(); ++i)
  {
    dms.push_back(DofMap(layouts[i], index_map, bs, std::move(dofmaps[i]), bs));
  }

  return dms;
}
//-----------------------------------------------------------------------------
std::vector<std::string> fem::get_coefficient_names(const ufcx_form& ufcx_form)
{
  return std::vector<std::string>(ufcx_form.coefficient_name_map,
                                  ufcx_form.coefficient_name_map
                                      + ufcx_form.num_coefficients);
}
//-----------------------------------------------------------------------------
std::vector<std::string> fem::get_constant_names(const ufcx_form& ufcx_form)
{
  return std::vector<std::string>(ufcx_form.constant_name_map,
                                  ufcx_form.constant_name_map
                                      + ufcx_form.num_constants);
}
//-----------------------------------------------------------------------------
std::vector<std::pair<int, std::vector<std::int32_t>>>
fem::compute_integration_domains(fem::IntegralType integral_type,
                                 const mesh::Topology& topology,
                                 std::span<const std::int32_t> entities,
                                 int dim, std::span<const int> values)
{
  const int tdim = topology.dim();
  if ((integral_type == IntegralType::cell ? tdim : tdim - 1) != dim)
  {
    throw std::runtime_error("Invalid MeshTags dimension: "
                             + std::to_string(dim));
  }

  {
    assert(topology.index_map(dim));
    auto it0 = entities.begin();
    auto it1 = std::lower_bound(it0, entities.end(),
                                topology.index_map(dim)->size_local());
    entities = entities.first(std::distance(it0, it1));
    values = values.first(std::distance(it0, it1));
  }

  std::vector<std::int32_t> entity_data;
  std::vector<int> values1;
  switch (integral_type)
  {
  case IntegralType::cell:
    entity_data.insert(entity_data.begin(), entities.begin(), entities.end());
    values1.insert(values1.begin(), values.begin(), values.end());
    break;
  default:
    auto f_to_c = topology.connectivity(tdim - 1, tdim);
    if (!f_to_c)
    {
      throw std::runtime_error(
          "Topology facet-to-cell connectivity has not been computed.");
    }
    auto c_to_f = topology.connectivity(tdim, tdim - 1);
    if (!c_to_f)
    {
      throw std::runtime_error(
          "Topology cell-to-facet connectivity has not been computed.");
    }

    switch (integral_type)
    {
    case IntegralType::exterior_facet:
    {
      // Create list of tagged boundary facets
      const std::vector bfacets = mesh::exterior_facet_indices(topology);
      std::vector<std::int32_t> facets;
      std::set_intersection(entities.begin(), entities.end(), bfacets.begin(),
                            bfacets.end(), std::back_inserter(facets));
      for (auto f : facets)
      {
        auto index_it = std::lower_bound(entities.begin(), entities.end(), f);
        assert(index_it != entities.end() and *index_it == f);
        std::size_t pos = std::distance(entities.begin(), index_it);
        auto facet
            = impl::get_cell_facet_pairs<1>(f, f_to_c->links(f), *c_to_f);
        entity_data.insert(entity_data.end(), facet.begin(), facet.end());
        values1.push_back(values[pos]);
      }
    }
    break;
    case IntegralType::interior_facet:
    {
      for (std::size_t j = 0; j < entities.size(); ++j)
      {
        const std::int32_t f = entities[j];
        if (f_to_c->num_links(f) == 2)
        {
          // Get the facet as a pair of (cell, local facet) pairs, one
          // for each cell
          auto facets
              = impl::get_cell_facet_pairs<2>(f, f_to_c->links(f), *c_to_f);
          entity_data.insert(entity_data.end(), facets.begin(), facets.end());
          values1.push_back(values[j]);
        }
      }
    }
    break;
    default:
      throw std::runtime_error(
          "Cannot compute integration domains. Integral type not supported.");
    }
  }

  // Build permutation that sorts by meshtag value
  std::vector<std::int32_t> perm(values1.size());
  std::iota(perm.begin(), perm.end(), 0);
  std::stable_sort(perm.begin(), perm.end(),
                   [&values1](auto p0, auto p1)
                   { return values1[p0] < values1[p1]; });

  std::size_t shape = 1;
  if (integral_type == IntegralType::exterior_facet)
    shape = 2;
  else if (integral_type == IntegralType::interior_facet)
    shape = 4;
  std::vector<std::pair<int, std::vector<std::int32_t>>> integrals;
  {
    // Iterator to mark the start of the group
    auto p0 = perm.begin();
    while (p0 != perm.end())
    {
      auto id0 = values1[*p0];
      auto p1 = std::find_if_not(p0, perm.end(),
                                 [id0, &values1](auto idx)
                                 { return id0 == values1[idx]; });

      std::vector<std::int32_t> data;
      data.reserve(shape * std::distance(p0, p1));
      for (auto it = p0; it != p1; ++it)
      {
        auto e_it0 = std::next(entity_data.begin(), (*it) * shape);
        auto e_it1 = std::next(e_it0, shape);
        data.insert(data.end(), e_it0, e_it1);
      }

      integrals.push_back({id0, std::move(data)});
      p0 = p1;
    }
  }

  return integrals;
}
//-----------------------------------------------------------------------------
