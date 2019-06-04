// Copyright (C) 2018 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "FormIntegrals.h"
#include <cstdlib>
#include <dolfin/common/types.h>
#include <dolfin/mesh/Facet.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/MeshIterator.h>

using namespace dolfin;
using namespace dolfin::fem;

//-----------------------------------------------------------------------------
FormIntegrals::FormIntegrals()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
const std::function<void(PetscScalar*, const PetscScalar*, const double*, int)>&
FormIntegrals::get_tabulate_tensor_fn_cell(unsigned int i) const
{
  if (i > _tabulate_tensor_cell.size())
    throw std::runtime_error("Invalid integral index");

  return _tabulate_tensor_cell[i];
}
//-----------------------------------------------------------------------------
const std::function<void(PetscScalar*, const PetscScalar*, const double*,
                         const int*, const int*)>&
FormIntegrals::get_tabulate_tensor_fn_exterior_facet(unsigned int i) const
{
  if (i > _tabulate_tensor_exterior_facet.size())
    throw std::runtime_error("Invalid integral index");

  return _tabulate_tensor_exterior_facet[i];
}
//-----------------------------------------------------------------------------
const std::function<void(PetscScalar*, const PetscScalar* w, const double*,
                         const int*, const int*)>&
FormIntegrals::get_tabulate_tensor_fn_interior_facet(unsigned int i) const
{
  if (i > _tabulate_tensor_interior_facet.size())
    throw std::runtime_error("Invalid integral index");

  return _tabulate_tensor_interior_facet[i];
}
//-----------------------------------------------------------------------------
void FormIntegrals::register_tabulate_tensor_cell(
    int i, void (*fn)(PetscScalar*, const PetscScalar*, const double*, int))
{
  if (std::find(_cell_integral_ids.begin(), _cell_integral_ids.end(), i)
      != _cell_integral_ids.end())
  {
    throw std::runtime_error("Integral with ID " + std::to_string(i)
                             + " already exists");
  }

  // Find insertion position
  int pos = std::distance(_cell_integral_ids.begin(),
                          std::upper_bound(_cell_integral_ids.begin(),
                                           _cell_integral_ids.end(), i));

  _cell_integral_ids.insert(_cell_integral_ids.begin() + pos, i);
  _tabulate_tensor_cell.insert(_tabulate_tensor_cell.begin() + pos, fn);
  _cell_integral_domains.insert(_cell_integral_domains.begin() + pos,
                                std::vector<std::int32_t>());
}
//-----------------------------------------------------------------------------
void FormIntegrals::register_tabulate_tensor_exterior_facet(
    int i, void (*fn)(PetscScalar*, const PetscScalar*, const double*,
                      const int*, const int*))
{

  if (std::find(_exterior_facet_integral_ids.begin(),
                _exterior_facet_integral_ids.end(), i)
      != _exterior_facet_integral_ids.end())
  {
    throw std::runtime_error("Integral with ID " + std::to_string(i)
                             + " already exists");
  }

  // Find insertion position
  int pos
      = std::distance(_exterior_facet_integral_ids.begin(),
                      std::upper_bound(_exterior_facet_integral_ids.begin(),
                                       _exterior_facet_integral_ids.end(), i));

  _exterior_facet_integral_ids.insert(
      _exterior_facet_integral_ids.begin() + pos, i);
  _tabulate_tensor_exterior_facet.insert(
      _tabulate_tensor_exterior_facet.begin() + pos, fn);
  _exterior_facet_integral_domains.insert(
      _exterior_facet_integral_domains.begin() + pos,
      std::vector<std::int32_t>());
}
//-----------------------------------------------------------------------------
void FormIntegrals::register_tabulate_tensor_interior_facet(
    int i, void (*fn)(PetscScalar*, const PetscScalar* w, const double*,
                      const int*, const int*))
{
  // At the moment, only accept one integral with index -1
  if (i != -1)
  {
    throw std::runtime_error(
        "Interior facet integral subdomain not supported. Under development.");
  }

  if (std::find(_interior_facet_integral_ids.begin(),
                _interior_facet_integral_ids.end(), i)
      != _interior_facet_integral_ids.end())
  {
    throw std::runtime_error("Integral with ID " + std::to_string(i)
                             + " already exists");
  }

  // Find insertion position
  int pos
      = std::distance(_interior_facet_integral_ids.begin(),
                      std::upper_bound(_interior_facet_integral_ids.begin(),
                                       _interior_facet_integral_ids.end(), i));

  _interior_facet_integral_ids.insert(
      _interior_facet_integral_ids.begin() + pos, i);
  _tabulate_tensor_interior_facet.insert(
      _tabulate_tensor_interior_facet.begin() + pos, fn);
  _interior_facet_integral_domains.insert(
      _interior_facet_integral_domains.begin() + pos,
      std::vector<std::int32_t>());
}
//-----------------------------------------------------------------------------
int FormIntegrals::num_integrals(FormIntegrals::Type type) const
{
  switch (type)
  {
  case Type::cell:
    return _tabulate_tensor_cell.size();
  case Type::exterior_facet:
    return _tabulate_tensor_exterior_facet.size();
  case Type::interior_facet:
    return _tabulate_tensor_interior_facet.size();
  default:
    throw std::runtime_error("FormIntegral type not supported.");
  }

  return 0;
}
//-----------------------------------------------------------------------------
const std::vector<int>&
FormIntegrals::integral_ids(FormIntegrals::Type type) const
{
  switch (type)
  {
  case Type::cell:
    return _cell_integral_ids;
  case Type::exterior_facet:
    return _exterior_facet_integral_ids;
  case Type::interior_facet:
    return _interior_facet_integral_ids;
  default:
    throw std::runtime_error("FormIntegral type not supported.");
  }

  return _cell_integral_ids;
}
//-----------------------------------------------------------------------------
const std::vector<std::int32_t>&
FormIntegrals::integral_domains(FormIntegrals::Type type, unsigned int i) const
{
  switch (type)
  {
  case Type::cell:
    if (i >= _cell_integral_domains.size())
      throw std::runtime_error("Invalid cell integral:" + std::to_string(i));
    return _cell_integral_domains[i];
  case Type::exterior_facet:
    return _exterior_facet_integral_domains[i];
  case Type::interior_facet:
    return _interior_facet_integral_domains[i];
  default:
    throw std::runtime_error("FormIntegral type not supported.");
  }

  return _cell_integral_domains[i];
}
//-----------------------------------------------------------------------------
void FormIntegrals::set_domains(FormIntegrals::Type type,
                                const mesh::MeshFunction<std::size_t>& marker)
{
  std::shared_ptr<const mesh::Mesh> mesh = marker.mesh();

  if (type == Type::cell)
  {
    if (mesh->topology().dim() != marker.dim())
    {
      throw std::runtime_error("Invalid MeshFunction dimension:"
                               + std::to_string(marker.dim()));
    }

    if (_cell_integral_ids.size() == 0)
      throw std::runtime_error("No cell integrals");

    // Create a reverse map
    std::map<int, int> cell_id_to_integral;
    for (unsigned int i = 0; i < _cell_integral_ids.size(); ++i)
    {
      if (_cell_integral_ids[i] != -1)
      {
        _cell_integral_domains[i].clear();
        cell_id_to_integral[_cell_integral_ids[i]] = i;
      }
    }

    for (unsigned int i = 0; i < marker.size(); ++i)
    {
      auto it = cell_id_to_integral.find(marker[i]);
      if (it != cell_id_to_integral.end())
        _cell_integral_domains[it->second].push_back(i);
    }
  }
  else if (type == Type::exterior_facet)
  {
    if (mesh->topology().dim() - 1 != marker.dim())
    {
      throw std::runtime_error("Invalid MeshFunction dimension:"
                               + std::to_string(marker.dim()));
    }

    if (_exterior_facet_integral_ids.size() == 0)
      throw std::runtime_error("No exterior facet integrals");

    // Create a reverse map
    std::map<int, int> facet_id_to_integral;
    for (unsigned int i = 0; i < _exterior_facet_integral_ids.size(); ++i)
    {
      if (_exterior_facet_integral_ids[i] != -1)
      {
        _exterior_facet_integral_domains[i].clear();
        facet_id_to_integral[_exterior_facet_integral_ids[i]] = i;
      }
    }

    for (unsigned int i = 0; i < marker.size(); ++i)
    {
      auto it = facet_id_to_integral.find(marker[i]);
      if (it != facet_id_to_integral.end())
        _exterior_facet_integral_domains[it->second].push_back(i);
    }
  }
  else
    throw std::runtime_error("FormIntegral type not supported.");
}
//-----------------------------------------------------------------------------
void FormIntegrals::set_default_domains(const mesh::Mesh& mesh)
{
  int tdim = mesh.topology().dim();

  // If there is a default integral, define it on all cells
  if (_cell_integral_ids.size() > 0 and _cell_integral_ids[0] == -1)
  {
    _cell_integral_domains[0].resize(mesh.num_entities(tdim));
    std::iota(_cell_integral_domains[0].begin(),
              _cell_integral_domains[0].end(), 0);
  }

  if (_exterior_facet_integral_ids.size() > 0
      and _exterior_facet_integral_ids[0] == -1)
  {
    // If there is a default integral, define it only on surface facets
    _exterior_facet_integral_domains[0].clear();
    for (const mesh::Facet& facet : mesh::MeshRange<mesh::Facet>(mesh))
    {
      if (facet.num_global_entities(tdim) == 1)
        _exterior_facet_integral_domains[0].push_back(facet.index());
    }
  }

  if (_interior_facet_integral_ids.size() > 0
      and _interior_facet_integral_ids[0] == -1)
  {
    // If there is a default integral, define it only on interior facets
    _interior_facet_integral_domains[0].clear();
    _interior_facet_integral_domains[0].reserve(mesh.num_entities(tdim - 1));
    for (const mesh::Facet& facet : mesh::MeshRange<mesh::Facet>(mesh))
    {
      if (facet.num_global_entities(tdim) != 1)
        _interior_facet_integral_domains[0].push_back(facet.index());
    }
  }
}
//-----------------------------------------------------------------------------
