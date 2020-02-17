// Copyright (C) 2018 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "FormIntegrals.h"
#include <cstdlib>
#include <dolfinx/common/types.h>
#include <dolfinx/mesh/MeshFunction.h>

using namespace dolfinx;
using namespace dolfinx::fem;

//-----------------------------------------------------------------------------
FormIntegrals::FormIntegrals()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
const std::function<void(PetscScalar*, const PetscScalar*, const PetscScalar*,
                         const double*, const int*, const int*)>&
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
                       const double*, const int*, const int*)>
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
  return _integrals[static_cast<int>(type)].size();
}
//-----------------------------------------------------------------------------
std::vector<int> FormIntegrals::integral_ids(FormIntegrals::Type type) const
{
  std::vector<int> ids;
  int type_index = static_cast<int>(type);
  for (auto& integral : _integrals[type_index])
    ids.push_back(integral.id);

  return ids;
}
//-----------------------------------------------------------------------------
const std::vector<std::int32_t>&
FormIntegrals::integral_domains(FormIntegrals::Type type, int i) const
{
  int type_index = static_cast<int>(type);
  return _integrals[type_index].at(i).active_entities;
}
//-----------------------------------------------------------------------------
void FormIntegrals::set_domains(FormIntegrals::Type type,
                                const mesh::MeshFunction<std::size_t>& marker)
{
  int type_index = static_cast<int>(type);
  std::vector<struct FormIntegrals::Integral>& integrals
      = _integrals[type_index];

  if (integrals.size() == 0)
    return;

  std::shared_ptr<const mesh::Mesh> mesh = marker.mesh();

  const mesh::Topology& topology = mesh->topology();
  const int tdim = topology.dim();

  int dim = tdim;
  if (type == Type::exterior_facet or type == Type::interior_facet)
    dim = tdim - 1;
  else if (type == Type::vertex)
    dim = 0;

  if (dim != marker.dim())
  {
    throw std::runtime_error("Invalid MeshFunction dimension:"
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

  // Get  mesh function data array
  const Eigen::Array<std::size_t, Eigen::Dynamic, 1>& mf_values
      = marker.values();

  // Get number of mesh entities of dimension d owned by this process
  assert(topology.index_map(dim));
  const int num_entities = topology.index_map(dim)->size_local();

  if (type == Type::exterior_facet)
  {
    mesh->create_connectivity(tdim - 1, tdim);
    const std::vector<bool>& interior_facets = topology.interior_facets();
    for (Eigen::Index i = 0; i < num_entities; ++i)
    {
      // Check that facet is an exterior facet (and not just on a
      // process boundary)
      if (!interior_facets[i])
      {
        auto it = id_to_integral.find(mf_values[i]);
        if (it != id_to_integral.end())
          integrals[it->second].active_entities.push_back(i);
      }
    }
  }
  else if (type == Type::interior_facet)
  {
    mesh->create_connectivity(tdim - 1, tdim);
    std::shared_ptr<const graph::AdjacencyList<std::int32_t>> connectivity
        = topology.connectivity(tdim - 1, tdim);
    assert(connectivity);
    for (Eigen::Index i = 0; i < num_entities; ++i)
    {
      if (connectivity->num_links(i) == 2)
      {
        auto it = id_to_integral.find(mf_values[i]);
        if (it != id_to_integral.end())
          integrals[it->second].active_entities.push_back(i);
      }
    }
  }
  else
  {
    // For cell and vertex integrals use all markers (but not on ghost
    // entities)
    for (Eigen::Index i = 0; i < num_entities; ++i)
    {
      auto it = id_to_integral.find(mf_values[i]);
      if (it != id_to_integral.end())
        integrals[it->second].active_entities.push_back(i);
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
    mesh.create_connectivity(tdim - 1, tdim);
    assert(topology.index_map(tdim - 1));
    const int num_facets = topology.index_map(tdim - 1)->size_local();
    const std::vector<bool>& interior_facets = topology.interior_facets();
    for (int f = 0; f < num_facets; ++f)
    {
      if (!interior_facets[f])
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
    inf_integrals[0].active_entities.reserve(mesh.num_entities(tdim - 1));

    // Get number of facets owned by this process
    mesh.create_connectivity(tdim - 1, tdim);
    assert(topology.index_map(tdim - 1));
    const int num_facets = topology.index_map(tdim - 1)->size_local();
    const std::vector<bool>& interior_facets = topology.interior_facets();

    // Loop over owned facets
    for (int f = 0; f < num_facets; ++f)
    {
      if (interior_facets[f])
        inf_integrals[0].active_entities.push_back(f);
    }
  }
}
//-----------------------------------------------------------------------------
