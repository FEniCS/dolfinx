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
fem::DofMap fem::create_dofmap(
    MPI_Comm comm, const ElementDofLayout& layout, mesh::Topology& topology,
    std::function<void(std::span<std::int32_t>, std::uint32_t)> permute_inv,
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
  if (permute_inv)
  {
    const int num_cells = topology.connectivity(D, 0)->num_nodes();
    topology.create_entity_permutations();
    const std::vector<std::uint32_t>& cell_info
        = topology.get_cell_permutation_info();
    int dim = layout.num_dofs();
    for (std::int32_t cell = 0; cell < num_cells; ++cell)
    {
      std::span<std::int32_t> dofs(dofmaps.front().data() + cell * dim, dim);
      permute_inv(dofs, cell_info[cell]);
    }
  }

  return DofMap(layout, index_map, bs, std::move(dofmaps.front()), bs);
}
//-----------------------------------------------------------------------------
std::vector<fem::DofMap> fem::create_dofmaps(
    MPI_Comm comm, const std::vector<ElementDofLayout>& layouts,
    mesh::Topology& topology,
    std::function<void(std::span<std::int32_t>, std::uint32_t)> permute_inv,
    std::function<std::vector<int>(const graph::AdjacencyList<std::int32_t>&)>
        reorder_fn)
{
  std::int32_t D = topology.dim();
  assert(layouts.size() == topology.entity_types(D).size());

  // Create required mesh entities
  for (std::int32_t d = 0; d < D; ++d)
  {
    if (layouts.front().num_entity_dofs(d) > 0)
      topology.create_entities(d);
  }

  auto [_index_map, bs, dofmaps]
      = build_dofmap_data(comm, topology, layouts, reorder_fn);
  auto index_map = std::make_shared<common::IndexMap>(std::move(_index_map));

  // If the element's DOF transformations are permutations, permute the
  // DOF numbering on each cell
  if (permute_inv)
  {
    if (layouts.size() != 1)
    {
      throw std::runtime_error(
          "DOF transformations not yet supported in mixed topology.");
    }
    std::int32_t num_cells = topology.connectivity(D, 0)->num_nodes();
    topology.create_entity_permutations();
    const std::vector<std::uint32_t>& cell_info
        = topology.get_cell_permutation_info();
    std::int32_t dim = layouts.front().num_dofs();
    for (std::int32_t cell = 0; cell < num_cells; ++cell)
    {
      std::span<std::int32_t> dofs(dofmaps.front().data() + cell * dim, dim);
      permute_inv(dofs, cell_info[cell]);
    }
  }

  std::vector<DofMap> dms;
  for (std::size_t i = 0; i < dofmaps.size(); ++i)
    dms.emplace_back(layouts[i], index_map, bs, std::move(dofmaps[i]), bs);

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
std::vector<std::int32_t> fem::compute_integration_domains(
    fem::IntegralType integral_type, const mesh::Topology& topology,
    std::span<const std::int32_t> entities, int dim)
{
  const int tdim = topology.dim();
  if ((integral_type == IntegralType::cell ? tdim : tdim - 1) != dim)
  {
    throw std::runtime_error("Invalid MeshTags dimension: "
                             + std::to_string(dim));
  }

  {
    assert(topology.index_map(dim));
    auto it1 = std::ranges::lower_bound(entities,
                                        topology.index_map(dim)->size_local());
    entities = entities.first(std::distance(entities.begin(), it1));
  }

  std::vector<std::int32_t> entity_data;
  switch (integral_type)
  {
  case IntegralType::cell:
    entity_data.insert(entity_data.begin(), entities.begin(), entities.end());
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
      std::ranges::set_intersection(entities, bfacets,
                                    std::back_inserter(facets));
      for (auto f : facets)
      {
        // Get the facet as a pair of (cell, local facet)
        auto facet
            = impl::get_cell_facet_pairs<1>(f, f_to_c->links(f), *c_to_f);
        entity_data.insert(entity_data.end(), facet.begin(), facet.end());
      }
    }
    break;
    case IntegralType::interior_facet:
    {
      // Create indicator for interprocess facets
      const std::vector<std::int32_t>& interprocess_facets
          = topology.interprocess_facets();
      std::int32_t num_facets = topology.index_map(tdim - 1)->size_local()
                                + topology.index_map(tdim - 1)->num_ghosts();
      std::vector<std::int8_t> interprocess_marker(num_facets);
      assert(topology.index_map(tdim - 1));

      std::ranges::for_each(interprocess_facets,
                            [&interprocess_marker](std::int32_t f)
                            { interprocess_marker[f] = 1; });

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
        }
        else if (interprocess_marker[f])
        {
          throw std::runtime_error(
              "Cannot compute interior facet integral over interprocess "
              "facet. Please use ghost mode shared facet when creating the "
              "mesh.");
        }
      }
    }
    break;
    default:
      throw std::runtime_error(
          "Cannot compute integration domains. Integral type not supported.");
    }
  }
  return entity_data;
}
//-----------------------------------------------------------------------------
