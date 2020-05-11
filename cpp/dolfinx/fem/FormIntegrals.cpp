// Copyright (C) 2018-2020 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "FormIntegrals.h"
#include <algorithm>
#include <cstdlib>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/types.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/MeshTags.h>

using namespace dolfinx;
using namespace dolfinx::fem;

//-----------------------------------------------------------------------------
FormIntegrals::FormIntegrals()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
const std::function<void(PetscScalar*, const PetscScalar*, const PetscScalar*,
                         const double*, const int*, const std::uint8_t*,
                         const std::uint32_t)>&
FormIntegrals::get_tabulate_tensor(FormIntegrals::Type type, int i) const
{
  int type_index = static_cast<int>(type);
  const std::vector<struct FormIntegrals::Integral>& integrals
      = _integrals.at(type_index);
  return integrals.at(i).tabulate;
}
//-----------------------------------------------------------------------------
void FormIntegrals::set_tabulate_tensor(
    FormIntegrals::Type type, int i,
    std::function<void(PetscScalar*, const PetscScalar*, const PetscScalar*,
                       const double*, const int*, const std::uint8_t*,
                       const std::uint32_t)>
        fn)

{
  const int type_index = static_cast<int>(type);
  std::vector<struct FormIntegrals::Integral>& integrals
      = _integrals.at(type_index);

  // Find insertion point
  int pos = 0;
  for (const auto& q : integrals)
  {
    if (q.id == i)
    {
      throw std::runtime_error("Integral with ID " + std::to_string(i)
                               + " already exists");
    }
    else if (q.id > i)
      break;
    ++pos;
  }

  // Create new Integral and insert
  struct FormIntegrals::Integral new_integral
      = {fn, i, std::vector<std::int32_t>()};

  integrals.insert(integrals.begin() + pos, new_integral);
}
//-----------------------------------------------------------------------------
int FormIntegrals::num_integrals(FormIntegrals::Type type) const
{
  return _integrals.at(static_cast<int>(type)).size();
}
//-----------------------------------------------------------------------------
std::vector<int> FormIntegrals::integral_ids(FormIntegrals::Type type) const
{
  std::vector<int> ids;
  int type_index = static_cast<int>(type);
  for (const auto& integral : _integrals[type_index])
    ids.push_back(integral.id);

  return ids;
}
//-----------------------------------------------------------------------------
const std::vector<std::int32_t>&
FormIntegrals::integral_domains(FormIntegrals::Type type, int i) const
{
  int type_index = static_cast<int>(type);
  return _integrals.at(type_index).at(i).active_entities;
}
//-----------------------------------------------------------------------------
void FormIntegrals::set_domains(FormIntegrals::Type type,
                                const mesh::MeshTags<int>& marker)
{
  int type_index = static_cast<int>(type);
  std::vector<struct FormIntegrals::Integral>& integrals
      = _integrals.at(type_index);
  if (integrals.size() == 0)
    return;

  std::shared_ptr<const mesh::Mesh> mesh = marker.mesh();
  const mesh::Topology& topology = mesh->topology();
  const int tdim = topology.dim();
  int dim = tdim;
  if (type == Type::exterior_facet or type == Type::interior_facet)
  {
    mesh->topology_mutable().create_connectivity(tdim - 1, tdim);
    dim = tdim - 1;
  }
  else if (type == Type::vertex)
    dim = 0;

  if (dim != marker.dim())
  {
    throw std::runtime_error("Invalid MeshTags dimension:"
                             + std::to_string(marker.dim()));
  }

  // Create a reverse map
  std::map<int, int> id_to_integral;
  for (std::size_t i = 0; i < integrals.size(); ++i)
  {
    if (integrals[i].id != -1)
    {
      integrals[i].active_entities.clear();
      id_to_integral.insert({integrals[i].id, i});
    }
  }

  // Get mesh tag data
  const std::vector<int>& values = marker.values();
  const std::vector<std::int32_t>& tagged_entities = marker.indices();
  assert(topology.index_map(dim));
  const auto entity_end
      = std::lower_bound(tagged_entities.begin(), tagged_entities.end(),
                         topology.index_map(dim)->size_local());

  if (type == Type::exterior_facet)
  {
    auto f_to_c = topology.connectivity(tdim - 1, tdim);
    assert(f_to_c);

    if (type == Type::exterior_facet)
    {
      // Only need to consider shared facets when there are no ghost cells
      assert(topology.index_map(tdim));
      std::set<std::int32_t> fwd_shared;
      if (topology.index_map(tdim)->num_ghosts() == 0)
      {
        fwd_shared.insert(
            topology.index_map(tdim - 1)->forward_indices().begin(),
            topology.index_map(tdim - 1)->forward_indices().end());
      }

      for (auto f = tagged_entities.begin(); f != entity_end; ++f)
      {
        const std::size_t i = std::distance(tagged_entities.cbegin(), f);

        // All "owned" facets connected to one cell, that are not shared,
        // should be external.
        if (f_to_c->num_links(*f) == 1
            and fwd_shared.find(*f) == fwd_shared.end())
        {
          const auto it = id_to_integral.find(values[i]);
          if (it != id_to_integral.end())
            integrals[it->second].active_entities.push_back(*f);
        }
      }
    }
    else if (type == Type::interior_facet)
    {
      for (auto f = tagged_entities.begin(); f != entity_end; ++f)
      {
        if (f_to_c->num_links(*f) == 2)
        {
          const std::size_t i = std::distance(tagged_entities.cbegin(), f);
          const auto it = id_to_integral.find(values[i]);
          if (it != id_to_integral.end())
            integrals[it->second].active_entities.push_back(*f);
        }
      }
    }
  }
  else
  {
    // For cell and vertex integrals use all markers (but not on ghost
    // entities)
    for (auto e = tagged_entities.begin(); e != entity_end; ++e)
    {
      const std::size_t i = std::distance(tagged_entities.cbegin(), e);
      const auto it = id_to_integral.find(values[i]);
      if (it != id_to_integral.end())
        integrals[it->second].active_entities.push_back(*e);
    }
  }
}
//-----------------------------------------------------------------------------
void FormIntegrals::set_default_domains(const mesh::Mesh& mesh)
{
  const mesh::Topology& topology = mesh.topology();
  const int tdim = topology.dim();
  std::vector<struct FormIntegrals::Integral>& cell_integrals
      = _integrals[static_cast<int>(FormIntegrals::Type::cell)];

  // Cells. If there is a default integral, define it on all owned cells
  if (cell_integrals.size() > 0 and cell_integrals[0].id == -1)
  {
    const int num_cells = topology.index_map(tdim)->size_local();
    cell_integrals[0].active_entities.resize(num_cells);
    std::iota(cell_integrals[0].active_entities.begin(),
              cell_integrals[0].active_entities.end(), 0);
  }

  // Exterior facets. If there is a default integral, define it only on
  // owned surface facets.
  std::vector<struct FormIntegrals::Integral>& exf_integrals
      = _integrals[static_cast<int>(FormIntegrals::Type::exterior_facet)];
  if (exf_integrals.size() > 0 and exf_integrals[0].id == -1)
  {
    // If there is a default integral, define it only on surface facets
    exf_integrals[0].active_entities.clear();

    // Get number of facets owned by this process
    mesh.topology_mutable().create_connectivity(tdim - 1, tdim);
    auto f_to_c = topology.connectivity(tdim - 1, tdim);
    assert(topology.index_map(tdim - 1));
    std::set<std::int32_t> fwd_shared_facets;

    // Only need to consider shared facets when there are no ghost cells
    if (topology.index_map(tdim)->num_ghosts() == 0)
    {
      fwd_shared_facets.insert(
          topology.index_map(tdim - 1)->forward_indices().begin(),
          topology.index_map(tdim - 1)->forward_indices().end());
    }

    const int num_facets = topology.index_map(tdim - 1)->size_local();
    for (int f = 0; f < num_facets; ++f)
    {
      if (f_to_c->num_links(f) == 1
          and fwd_shared_facets.find(f) == fwd_shared_facets.end())
        exf_integrals[0].active_entities.push_back(f);
    }
  }

  // Interior facets. If there is a default integral, define it only on
  // owned interior facets.
  std::vector<struct FormIntegrals::Integral>& inf_integrals
      = _integrals[static_cast<int>(FormIntegrals::Type::interior_facet)];
  if (inf_integrals.size() > 0 and inf_integrals[0].id == -1)
  {
    // If there is a default integral, define it only on interior facets
    inf_integrals[0].active_entities.clear();

    // Get number of facets owned by this process
    mesh.topology_mutable().create_connectivity(tdim - 1, tdim);
    assert(topology.index_map(tdim - 1));

    const int num_facets = topology.index_map(tdim - 1)->size_local();
    auto f_to_c = topology.connectivity(tdim - 1, tdim);
    inf_integrals[0].active_entities.reserve(num_facets);
    for (int f = 0; f < num_facets; ++f)
    {
      if (f_to_c->num_links(f) == 2)
        inf_integrals[0].active_entities.push_back(f);
    }
  }
}
//-----------------------------------------------------------------------------
